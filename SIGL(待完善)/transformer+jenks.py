import math
from abc import ABC
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
# from LSTM import prepare_data
import pdb
import re
# from helper.profile import BestClusterGroup, Model, test_single_graph
from scipy.spatial.distance import pdist, squareform, hamming
import random
import os
import sys
import argparse
from sklearn.model_selection import KFold, ShuffleSplit
import copy
from sklearn.cluster import KMeans
import matplotlib as mpl
import torch.optim as optim
import torch.utils.data as Data
from jenkspy import JenksNaturalBreaks

d_ff = 2048  # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6  # 有多少个encoder和decoder
n_heads = 8  # Multi-Head Attention设置为8
torch.cuda.set_device(0)


# 不拼接字符串
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda()  # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_k, d_v, d_model=512):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, input_Q, input_K, input_V, attn_mask):  # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):  # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]

        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)  # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff, d_model=512):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        self.d_model = d_model

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(n_heads, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):  # dec_inputs: [batch_size, tgt_len, d_model]
        # enc_outputs: [batch_size, src_len, d_model]
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)  # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


def get_attn_subsequence_mask(seq):  # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, max_len=1000, d_model=512, n_layers=6):
        super(Decoder, self).__init__()
        # self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_emb = nn.Linear(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # dec_inputs: [batch_size, tgt_len]
        # enc_intpus: [batch_size, src_len]
        # enc_outputs: [batsh_size, src_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs).cuda()  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:  # dec_outputs: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_k, d_v)  # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(d_ff)  # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):  # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        # enc_outputs: [batch_size, src_len, d_model],
        # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


def get_attn_pad_mask_0(seq_q, seq_k):  # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q, column_q = seq_q.size()
    batch_size, len_k, column_k = seq_k.size()
    # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    pad_attn_mask = seq_k.data.eq(torch.LongTensor([0] * column_k)).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k, column_k)  # 扩展成多维度


def get_attn_pad_mask(seq_q, seq_k):  # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q, column_q = seq_q.size()
    batch_size, len_k, column_k = seq_k.size()
    # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    # pad_attn_mask = seq_k.data.eq(torch.LongTensor([0] * column_k)).unsqueeze(1)
    zero_row = [0] * column_k
    pad_attn_mask = torch.zeros(batch_size, len_k).to(dtype=torch.bool).cuda()
    for idx, rows in enumerate(seq_k):
        for row_i, row in enumerate(rows):
            if row == zero_row:
                pad_attn_mask[idx][row_i] = True
    pad_attn_mask = pad_attn_mask.unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # 扩展成多维度


def prepare_data_1(files, max_len=1000):
    sketches_list = []
    train_label = []
    for file_one in files:
        with open(file_one, 'r') as f:
            sketches = load_sketches(f)
            sketches_list.append(sketches)
            train_label.append(extract_label(file_one))
    # 训练集归一化
    min_, max_data, sketches_list = min_max_scaler(sketches_list)
    stream_dataset = StreamDataset(sketches_list, max_len, train_label)
    train_data_loader = torch.utils.data.DataLoader(stream_dataset, batch_size=1)

    return train_data_loader


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, max_len=1000, d_model=512, n_layers=6):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)  # 把字转换字向量
        self.src_emb = nn.Linear(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)  # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, src_len] out of range
        # ----------
        enc_outputs_0 = self.src_emb(enc_inputs)  # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs_0)  # enc_outputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs :   [batch_size, src_len, d_model],
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns, enc_outputs_0


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_len=1000, d_model=512):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(src_vocab_size, max_len).cuda()
        self.Decoder = Decoder(tgt_vocab_size, max_len).cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):  # enc_inputs: [batch_size, src_len]
        # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns, encoder_emb = self.Encoder(
            enc_inputs)  # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # dec_outpus    : [batch_size, tgt_len, d_model],
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
        return enc_outputs, dec_outputs, encoder_emb


def train_valid(train_files, test_files, model_path, new, valid_files=None):
    gpu = torch.cuda.is_available()

    max_len = 3000
    train_dataset = prepare_data_1(train_files, max_len)
    test_dataset = prepare_data_1(test_files, max_len)
    valid_dataset = prepare_data_1(valid_files, max_len)

    input_size = 2000
    input_size = 64
    model = Transformer(input_size, input_size, max_len)
    if gpu:
        model = model.cuda()
    if new:
        model = train_transformer(train_dataset, model, train_files.__len__(), gpu, max_len, test_dataset,
                                  valid_dataset)
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))
    return process_test(max_len, model, gpu, train_dataset, test_dataset, valid_dataset)


def accuracy_0(valid_label, valid_cluster, cluster_num):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # benign=1
    for r in range(len(valid_label)):
        if valid_label[r] == 0 and valid_cluster[r] == cluster_num:
            TP += 1
        elif valid_label[r] == 1 and valid_cluster[r] == cluster_num:
            FP += 1
        elif valid_label[r] == 1 and valid_cluster[r] < cluster_num:
            TN += 1
        elif valid_label[r] == 0 and valid_cluster[r] < cluster_num:
            FN += 1
    acc = (TP + TN) / (FP + FN + TP + TN)
    pr = 0
    recall = 0
    fpr = 0
    if (TP + FP) != 0:
        pr = TP / (TP + FP)
    else:
        if TP == 0:
            pr = 0
        else:
            pr = 1
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        if TP == 0:
            recall = 0
        else:
            recall = 1
    if (TN + FP) != 0:
        fpr = FP / (TN + FP)
    else:
        if FP == 0:
            fpr = 0
        else:
            fpr = 1
    # print("--accuracy-", TP, TN, FP, FN, acc, pr, recall)
    return TP, TN, FP, FN, acc, pr, recall, fpr


def train_transformer(train_dataset, model, length, gpu, max_len, test_dataset, valid_dataset):
    # criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(600):
        for idx, data in enumerate(train_dataset):
            torch.cuda.empty_cache()
            model.zero_grad()
            seq = data['seq'].float()
            if gpu:
                seqCuda = seq.cuda()
            else:
                seqCuda = seq
            enc_out, dec_out, enc_emb = model(seqCuda, seqCuda)
            loss = loss_fn(enc_out.cuda(), dec_out.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if loss < 0.01:
        if epoch + 1 == 200:
            print('\n++++++++Epoch:', '%04d' % (epoch + 1), 'loss_sum =', '{:.6f}'.format(loss))
            break
    return model


def cluster(train_dataset, model, cluster_num, gpu):
    train_feature, train_label = extract_feature(train_dataset, model, gpu)
    normal = KMeans(n_clusters=cluster_num).fit(train_feature)
    center = normal.cluster_centers_
    labels = normal.labels_
    return center, labels, train_feature, train_label


def accuracy(real_labels, test_labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # benign=1
    for r in range(len(real_labels)):
        if real_labels[r] == 0 and test_labels[r] == 0:
            TP += 1
        elif real_labels[r] == 1 and test_labels[r] == 0:
            FP += 1
        elif real_labels[r] == 1 and test_labels[r] == 1:
            TN += 1
        elif real_labels[r] == 0 and test_labels[r] == 1:
            FN += 1
    acc = (TP + TN) / (FP + FN + TP + TN)
    if (TP + FP) != 0:
        pr = TP / (TP + FP)
    else:
        if TP == 0:
            pr = 0
        else:
            pr = 1
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        if TP == 0:
            recall = 0
        else:
            recall = 1
    if (TN + FP) != 0:
        fpr = FP / (TN + FP)
    else:
        if FP == 0:
            fpr = 0
        else:
            fpr = 1
    # print("--accuracy-", TP, TN, FP, FN, acc, pr, recall)
    return TP, TN, FP, FN, acc, pr, recall, fpr


def process_test(max_len, model, gpu, train_dataset, test_dataset, valid_dataset):
    acc_list = []
    pr_list = []
    recall_list = []
    fpr_list = []

    # 根据验证集来算
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss(weight=None, reduction='mean')  # 不能大于1
    # loss_fn = torch.nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)
    # loss_fn = nn.PairwiseDistance(p=2)

    # train_mean = get_train_threshold(valid_dataset, model, gpu, loss_fn)
    # print("-----train_loss_mean=", train_mean)
    real_labels = get_test_real_label(test_dataset)

    test_graph_zone_max, encoder_graph_max = get_test_loss(test_dataset, model, gpu, loss_fn)

    threshold = min(encoder_graph_max)
    max_threshold = max(encoder_graph_max)
    length = max_threshold - threshold
    print("-----encoder_graph_max threshold ", threshold, encoder_graph_max)

    for rate in np.arange(0, length * 1.1, length / 10):
        new_threshold = rate + threshold
        test_labels = get_test_labels(encoder_graph_max, new_threshold)
        TP, TN, FP, FN, acc, pr, recall, fpr = accuracy(real_labels, test_labels)
        print("-----encoder TP TN ", new_threshold, TP, TN, FP, FN, acc, pr, recall, fpr)
        acc_list.append(acc)
        pr_list.append(pr)
        fpr_list.append(fpr)
        recall_list.append(recall)

    threshold = min(test_graph_zone_max)
    max_threshold = max(test_graph_zone_max)
    length = max_threshold - threshold
    print("-----test_graph_zone_max threshold ", threshold, test_graph_zone_max)

    for rate in np.arange(0, length * 1.1, length / 10):
        new_threshold = rate + threshold
        test_labels = get_test_labels(test_graph_zone_max, new_threshold)
        TP, TN, FP, FN, acc, pr, recall, fpr = accuracy(real_labels, test_labels)
        print("-----TP TN ", new_threshold, TP, TN, FP, FN, acc, pr, recall, fpr)
        acc_list.append(acc)
        pr_list.append(pr)
        fpr_list.append(fpr)
        recall_list.append(recall)
    print("-----pr_list=", pr_list)
    print("-----recall_list=", recall_list)
    print("-----fpr_list=", fpr_list)
    # pr_list.append(1)
    # recall_list.append(0.01)
    return pr_list, recall_list, fpr_list


def get_test_real_label(dataset):
    labels = []
    for idx, data in enumerate(dataset):
        for label in data['label']:
            labels.append(label)
    return labels


def get_test_labels(graph_zone_max_list, threshold):
    label_list = []
    # for graph_zone_max in graph_zone_max_list:
    #     if graph_zone_max <= threshold:
    #         label_list.append(1)
    #     else:
    #         label_list.append(0)
    for graph_zone_max in graph_zone_max_list:
        if graph_zone_max <= threshold:
            label_list.append(1)
        else:
            label_list.append(0)
    return label_list


def get_test_labels_0(graph_zone_mean_list, threshold):
    label_list = []
    for graph_zone_mean in graph_zone_mean_list:
        above = False
        for zone_mean in graph_zone_mean:
            if zone_mean > threshold:
                above = True
                break
        if above:
            label_list.append(0)
        else:
            label_list.append(1)
    return label_list


def get_test_loss(dataset, model, gpu, loss_fn):
    model.eval()
    # loss_fn = torch.nn.MSELoss()
    graph_zone_max_list = []
    encoder_zone_max_list = []
    for idx, data in enumerate(dataset):
        torch.cuda.empty_cache()
        seq = data['seq'].float()
        if gpu:
            seq = seq.cuda()
        enc_out, dec_out, enc_emb = model(seq, seq)
        graph_nodes_loss = []
        encoder_loss = []
        for i in range(dec_out.shape[1]):
            loss1 = loss_fn(enc_emb[0][i], dec_out[0][i])
            loss2 = loss_fn(enc_out[0][i], dec_out[0][i])
            graph_nodes_loss.append(float(loss1.data))
            encoder_loss.append(float(loss2.data))
        # print('test -- ', idx)
        graph_zone_max = jenks_zone_max(graph_nodes_loss)
        graph_zone_max_list.append(graph_zone_max)
        encoder_zone_max = jenks_zone_max(encoder_loss)
        encoder_zone_max_list.append(encoder_zone_max)
    return graph_zone_max_list, encoder_zone_max_list


def get_train_threshold(dataset, model, gpu, loss_fn):
    model.eval()
    # loss_fn = torch.nn.MSELoss()
    threshold_list = []
    for idx, data in enumerate(dataset):
        torch.cuda.empty_cache()
        seq = data['seq'].float()
        if gpu:
            seq = seq.cuda()
        enc_out, dec_out, enc_emb = model(seq, seq)
        graph_nodes_loss = []
        for i in range(dec_out.shape[1]):
            # loss = loss_fn(enc_out[0][i], dec_out[0][i])
            enc = float(torch.max(enc_out[0][i]))
            dec = float(torch.max(dec_out[0][i]))
            sub = enc - dec
            loss = math.pow(sub, 2) / 2
            graph_nodes_loss.append(loss)
            graph_nodes_loss.append(loss)
        zone_mean_max = jenks_max_zone_avg(graph_nodes_loss)
        threshold_list.append(zone_mean_max)
    train_threshold_mean = np.array(threshold_list).mean()
    train_threshold_std = np.array(threshold_list).std()

    threshold = train_threshold_mean + train_threshold_std * 3

    return threshold


# 每个图的zone mean
def jenks_zone_max(node_losses):
    # zone_num = round(len(node_losses) / 20)
    zone_num = 4
    node_losses.sort()
    node_losses.reverse()
    jnb = JenksNaturalBreaks(zone_num)
    jnb.fit(node_losses)
    zone_list = jnb.groups_
    zone_mean_max = 0
    zone_second = 0
    for zone_one in zone_list:
        zone_mean = np.array(zone_one).mean()
        if zone_mean > zone_mean_max:
            zone_second = zone_mean_max
            zone_mean_max = zone_mean
    # print('---test --zone max second ', zone_mean_max, zone_second)
    return zone_mean_max
    # zone_mean_list = []
    # for zone_one in zone_list:
    #     zone_mean = np.array(zone_one).mean()
    #     zone_mean_list.append(zone_mean)
    # return zone_mean_list


# 每个图的zone mean
def jenks_zone_avg(node_losses):
    node_losses.sort()
    node_losses.reverse()
    jnb = JenksNaturalBreaks(7)
    jnb.fit(node_losses)
    zone_list = jnb.groups_

    zone_mean_list = []
    for zone_one in zone_list:
        zone_mean = np.array(zone_one).mean()
        zone_mean_list.append(zone_mean)
    return zone_mean_list


#
# def jenks_zone_avg_0(node_losses, threshold):
#     node_losses.sort()
#     node_losses.reverse()
#     breaks = jenks.classify(np.array(node_losses), 7)
#     breaks.reverse()
#     zone_list = []
#     for i in range(len(breaks)):
#         if i == 0:
#             continue
#         zone = []
#         for j in range(len(node_losses)):
#             if i == 1 and j == 0:
#                 zone.append(node_losses[j])
#             if breaks[i] <= node_losses[j] < breaks[i - 1]:
#                 zone.append(node_losses[j])
#         zone_list.append(zone)
#     zone_mean_list = []
#     above = False
#     threat_list = []
#     for zone_one in zone_list:
#         zone_mean = np.array(zone_one).mean()
#         zone_mean_list.append(zone_mean)
#         if zone_mean > threshold:
#             above = True
#             threat_list.append()
#     return zone_mean, above


def jenks_max_zone_avg(node_losses):
    zone_num = round(len(node_losses) / 20)
    zone_num = 2
    node_losses.sort()
    node_losses.reverse()
    jnb = JenksNaturalBreaks(zone_num)
    jnb.fit(node_losses)
    zone_list = jnb.groups_

    zone_mean_max = 0
    for zone_one in zone_list:
        mean_one = np.array(zone_one).mean()
        if mean_one > zone_mean_max:
            zone_mean_max = mean_one
    return zone_mean_max


def get_loss_1(dataset, model, gpu):
    model.eval()
    labels = []
    loss_fn = torch.nn.MSELoss()
    loss_all = []
    for idx, data in enumerate(dataset):
        torch.cuda.empty_cache()
        seq = data['seq'].float()
        if gpu:
            seq = seq.cuda()
        enc_out, dec_out, enc_emb = model(seq, seq)
        loss_one = loss_fn(enc_out.cuda(), enc_emb.cuda())

        loss_val = (float(loss_one.data))
        loss_all.append(loss_one)
        for label in data['label']:
            labels.append(label)
    # loss_mean = (np.array(loss_all)).mean()
    return loss_all, labels


def get_loss_0(dataset, model, gpu):
    model.eval()
    labels = []
    loss_fn = torch.nn.MSELoss()
    loss_all = []
    for idx, data in enumerate(dataset):
        torch.cuda.empty_cache()
        seq = data['seq'].float()
        if gpu:
            seq = seq.cuda()
        enc_out, dec_out = model(seq, seq)
        seq_linear = model.Encoder.src_emb(seq)
        loss_one = loss_fn(enc_out.cuda(), seq_linear.cuda())
        loss_all.append(loss_one)
        for label in data['label']:
            labels.append(label)
    # loss_mean = (np.array(loss_all)).mean()
    return loss_all, labels


def extract_feature(dataset, model, gpu):
    model.eval()
    feature = []
    labels = []

    for idx, data in enumerate(dataset):
        torch.cuda.empty_cache()
        seq = data['seq'].float()
        if gpu:
            seq = seq.cuda()
        enc_outputs, _ = model.Encoder(seq)

        enc_outputs_trans = enc_outputs.permute(0, 2, 1)
        # pool = nn.AdaptiveAvgPool1d(1)
        # batch
        # pool_res = pool(enc_outputs_trans)
        # for one_i, one in enumerate(pool_res):
        #     poll_trans = one.permute(1, 0)
        #     feature.append(poll_trans.data.cpu().numpy()[0])
        for one_i, one in enumerate(enc_outputs_trans):
            one_max, _ = torch.max(one, 1)
            feature.append(one_max.data.cpu().numpy())

        for label in data['label']:
            labels.append(label)

    return np.array(feature), labels


def extract_feature_1(dataset, model, gpu):
    model.eval()
    feature = []
    labels = []

    for idx, data in enumerate(dataset):
        torch.cuda.empty_cache()
        seq = data['seq'].float()
        if gpu:
            seq = seq.cuda()
        enc_outputs, _ = model.Encoder(seq)

        # if gpu:
        #     feature[idx] = output.data.cpu().numpy()
        #     label[idx] = data['label'].data.cpu().numpy()
        # else:
        #     feature[idx] = output.data.numpy()
        #     label[idx] = data['label'].data.numpy()
        enc_outputs_trans = enc_outputs.permute(0, 2, 1)
        pool = nn.AdaptiveAvgPool1d(1)
        # batch
        pool_res = pool(enc_outputs_trans)
        for one_i, one in enumerate(pool_res):
            poll_trans = one.permute(1, 0)
            feature.append(poll_trans.data.cpu().numpy()[0])
        # for one_i, one in enumerate(enc_outputs_trans):
        #     # one_max, _ = torch.max(one, 1)
        #     # feature.append(one_max.data.numpy())
        #     pool_one = pool(one)
        #     poll_trans = pool_one.permute(1, 0)
        #     feature.append(poll_trans.data.numpy()[0])
        for label in data['label']:
            labels.append(label)

    return np.array(feature), labels


def extract_feature0(dataset, model):
    model.eval()
    feature = []
    labels = []

    for idx, data in enumerate(dataset):
        seq = data['seq'].float()
        enc_outputs, _ = model.Encoder(seq)
        enc_outputs_trans = enc_outputs.permute(0, 2, 1)
        pool = nn.AdaptiveAvgPool1d(1)
        # batch
        for one_i, one in enumerate(enc_outputs_trans):
            pool_one = pool(one)
            poll_trans = pool_one.permute(1, 0)
            feature.append(poll_trans.data.numpy()[0])
        for label in data['label']:
            labels.append(label)

    return np.array(feature), labels


def get_threshold(center, feature, label):
    threshold = np.zeros(center.shape[0])
    for i in range(feature.shape[0]):
        dis = np.linalg.norm(feature[i] - center[label[i]])
        if dis > threshold[label[i]]:
            threshold[label[i]] = dis
    return threshold


def distance(center, feature, threshold):
    cluster_res = np.zeros(feature.shape[0])
    for i in range(feature.shape[0]):
        dis = 100000
        pos = -1
        for j in range(center.shape[0]):
            d = np.linalg.norm(feature[i] - center[j])
            # print(d),
            if d < dis:
                dis = d
                pos = j
        if dis > threshold[pos]:
            cluster_res[i] = center.shape[0]
            # print("--beyond the threshold--", i, threshold[pos], dis)
        else:
            cluster_res[i] = pos
        # if i<3:
        #     print("--i threshold dis--", i, threshold[pos], dis)
    return cluster_res


# def train(train_dataset, epochs, model, gpu):
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
#     loss_fn = torch.nn.MSELoss()
#     # loss_fn = torch.nn.CrossEntropyLoss()
#     for epoch in range(epochs):
#         loss_ave = 0.0
#         model.train()
#         print('Epoch:' + str(epoch))
#         for idx, data in enumerate(train_dataset):
#             model.zero_grad()
#             seq = data['seq'].float()
#             if gpu:
#                 seq_cuda = seq.cuda()
#                 y, inputs, feature = model(seq_cuda)
#             else:
#                 y, inputs, feature = model(seq)
#             loss = loss_fn(y, inputs)
#             loss.backward()
#             optimizer.step()
#             loss_ave += loss
#         print('lossAve -- ', loss_ave)
#         if loss_ave < 0.2:
#             break
#     return model


def trans_to_index(data_list, max_len):
    src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8, '女': 9, '人': 10,
                 '苹': 11,
                 '果': 12}  # 词源字典  字：索引
    src_vocab = {}
    i = 0
    for file_index in range(data_list.__len__()):
        append_row = max_len - data_list[file_index].shape[0]
        while append_row > 0:
            # file_one.extend([0] * file_one.shape[0][1])
            new_row = [0] * data_list[file_index].shape[1]
            # file_one = np.append(file_one, new_row)
            data_list[file_index] = np.append(data_list[file_index], values=[new_row], axis=0)

            append_row = append_row - 1
        file_one_int = data_list[file_index].astype('int')
        file_one_string = file_one_int.astype('str')
        for row_one in file_one_string:
            row_str = ''
            for data in row_one:
                row_str += data
            if src_vocab.get(row_str) is None:
                src_vocab[row_str] = i
                i = i + 1
    enc_inputs = []
    for file_one in data_list:
        file_one_int = file_one.astype('int')
        file_one_string = file_one_int.astype('str')
        enc_input = []
        for row_one in file_one_string:
            row_str = ''
            for data in row_one:
                row_str += data
            enc_input.append(src_vocab[row_str])
        # enc_input = [[src_vocab[row_str] for row_one in file_one]]
        enc_inputs.append(enc_input)
    src_vocab_size = len(src_vocab)
    return torch.LongTensor(enc_inputs), src_vocab_size

    # src_vocab_size = len(src_vocab)  # 字典字的个数
    # tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9,
    #              'apple': 10, 'girl': 11}
    # idx2word = {tgt_vocab[key]: key for key in tgt_vocab}  # 把目标字典转换成 索引：字的形式
    # tgt_vocab_size = len(tgt_vocab)  # 目标字典尺寸
    # # src_len = len(sentences[0][0].split(" "))  # Encoder输入的最大长度
    # tgt_len = len(sentences[0][1].split(" "))  # Decoder输入输出最大长度


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


def load_sketches(fh):
    sketches = list()
    for line in fh:
        sketch = list(map(float, line.strip().split()))
        sketches.append(sketch)
    # sketchesNew = SequenceSample(sketches)
    return np.array(sketches)


# 归一化
def min_max_scaler(data_list, min_=-1, max_=-1):
    if min_ < 0:
        min_, max_ = min_max(data_list[0])
        for data in data_list:
            minT, maxT = min_max(data)
            if minT < min_:
                min_ = minT
            if maxT > max_:
                max_ = maxT
    data_list_new = []
    for data in data_list:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j] - min_) / (max_ - min_)
        data_list_new.append(data)
    return min_, max_, data_list_new


def min_max(data):
    max_ = data[0][0]
    min_ = data[0][0]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if max_ < data[i][j]:
                max_ = data[i][j]
            if min_ > data[i][j]:
                min_ = data[i][j]
    return min_, max_


class StreamDataset(torch.utils.data.Dataset):
    def __init__(self, seq, max_len, label):
        self.max_len = max_len
        seq_pad = np.zeros((len(seq), max_len, seq[0].shape[1]))
        self.length = []
        for index, each in enumerate(seq):
            seq_pad[index, :each.shape[0], :] = each
            self.length.append(each.shape[0])
        self.seq = seq_pad
        self.label = label

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        # a = self.seq[idx, :, :]
        # b = self.seq
        # return {'seq': self.seq[idx, :, :],
        #         'label': self.label[idx]}
        seq_len = self.length[idx]
        return {'seq': self.seq[idx, :seq_len, :],
                'label': self.label[idx]}


class PaddingDataset(torch.utils.data.Dataset):
    def __init__(self, seq):
        max_len = 0
        for indx, each in enumerate(seq):
            if each.shape[0] > max_len:
                max_len = each.shape[0]
        self.max_len = max_len
        seq_pad = np.zeros((len(seq), max_len, seq[0].shape[1]))
        self.length = []
        for indx, each in enumerate(seq):
            seq_pad[indx, :each.shape[0], :] = each
            self.length.append(each.shape[0])
        self.seq = torch.LongTensor(seq_pad)
        # self.seq = seq_pad

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        # seq_len = self.length[idx]
        # s1 = self.seq[idx, :self.max_len, :]
        # s2 = self.seq
        return {'seq': self.seq[idx, :self.max_len, :],
                'label': []}


def extract_label(filename):
    res = re.search(r'embedding_([a-z]+)-', filename)
    if res:
        label = res.group(1)
    else:
        label = ''
    if label == 'cnn':
        return 1
    if label == 'download':
        return 1
    if label == 'gmail':
        return 1
    if label == 'vgame':
        return 1
    if label == 'youtube':
        return 1
    if label == 'attack':
        return 0
    if filename.find('wget-normal') > -1:
        return 1
    if filename.find('benign') > -1:
        return 1
    if filename.find('attack') > -1:
        return 0


def extract_label_1(filename):
    if filename.find('wget-normal') > -1:
        return 1
    if filename.find('benign') > -1:
        return 1
    if filename.find('attack') > -1:
        return 0


def draw_pr(recall, precision):
    x = np.array(recall)
    y = np.array(precision)
    plt.figure()
    plt.xlim([-0.1, 1.1])
    plt.ylim([0.15, 1.1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR curve')
    plt.plot(x, y, marker='D')
    plt.savefig("./streamspot/ret/" + random_walk_path + "_pr.jpg")
    plt.show()


def draw_roc(recall, fpr):
    x = np.array(fpr)
    y = np.array(recall)
    plt.figure()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('fpr')
    plt.ylabel('recall')
    plt.title('ROC curve')
    plt.plot(x, y, marker='D')
    plt.savefig("./streamspot/ret/" + random_walk_path + "_roc.jpg")
    plt.show()


def range_map(value):
    if value < 0.05:
        return 0.0
    elif 0.05 <= value < 0.15:
        return 0.1
    elif 0.15 <= value < 0.25:
        return 0.2
    elif 0.25 <= value < 0.35:
        return 0.3
    elif 0.35 <= value < 0.45:
        return 0.4
    elif 0.45 <= value < 0.55:
        return 0.5
    elif 0.55 <= value < 0.65:
        return 0.6
    elif 0.65 <= value < 0.75:
        return 0.7
    elif 0.75 <= value < 0.85:
        return 0.8
    elif 0.85 <= value < 0.95:
        return 0.9
    elif 0.95 <= value:
        return 1.0


def final_data(recall_list_5, pr_list_5):
    recall_r = []
    pr_r = []
    fpr_r = []
    recall_ret = {}
    for i in np.arange(0, 1.1, 0.1):
        recall_ret[round(i, 1)] = []
    for i in range(len(recall_list_5)):
        for j in range(len(recall_list_5[i])):
            if pr_list_5[i][j] == 0:
                continue
            if j > 0 and recall_list_5[i][j] == recall_list_5[i][j - 1]:
                if pr_list_5[i][j] > pr_list_5[i][j - 1]:
                    recall_ret[range_map(recall_list_5[i][j])][-1] = pr_list_5[i][j]
                continue
            recall_ret[range_map(recall_list_5[i][j])].append(pr_list_5[i][j])
    ret = []
    for k, v in recall_ret.items():
        if len(v) == 0:
            continue
        val = (np.array(v)).mean()
        # pr_r[int(k * 10)] = val
        ret.append([k, val])
    print("---recall ret --", recall_ret)
    print("---ret --", ret)
    ret = np.array(ret)
    ret = ret[np.lexsort(ret[:, ::-1].T)]
    for v in ret:
        recall_r.append(v[0])
        pr_r.append(v[1])
    return draw_pr(recall_r, pr_r)


def final_data_0(recall_list_5, pr_list_5):
    recall_r = []
    pr_r = []
    recall_ret = {}
    for i in np.arange(0, 1.1, 0.1):
        recall_ret[round(i, 1)] = []
        recall_r.append(round(i, 1))
    for i in range(len(recall_list_5)):
        for j in range(len(recall_list_5[i])):
            if pr_list_5[i][j] == 0:
                continue
            if j > 0 and recall_list_5[i][j] == recall_list_5[i][j - 1]:
                if pr_list_5[i][j] > pr_list_5[i][j - 1]:
                    recall_ret[range_map(recall_list_5[i][j])][-1] = pr_list_5[i][j]
                continue
            recall_ret[range_map(recall_list_5[i][j])].append(pr_list_5[i][j])
    # pr_r = np.zeros(11)
    ret = []
    for k, v in recall_ret.items():
        if len(v) == 0:
            continue
        val = (np.array(v)).mean()
        # pr_r[int(k * 10)] = val
        ret.append([k, val])
    ret = np.array(ret)
    ret = ret[np.lexsort(ret[:, ::-1].T)]
    for v in ret.values():
        recall_r.append(v[0])
        pr_r.append(v[1])
    print("---recall --", recall_ret)
    print("---pr     --", pr_r)
    return draw_pr(recall_r, pr_r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainPath', default="",
                        help='absolute path to the directory that contains all training sketches')
    parser.add_argument('-u', '--testPath', default="",
                        help='absolute path to the directory that contains all test sketches')
    parser.add_argument('-c', '--cross-validation',
                        help='number of cross validation we perform (use 0 to turn off cross validation)', type=int,
                        default=5)
    parser.add_argument('-d', '--dataset', help='dataset', default='streamspot')
    parser.add_argument('-s', '--split', help='split rate', type=float, default=0.4)
    parser.add_argument('-v', '--valid', help='valid', type=float, default=0.4)
    parser.add_argument('-a', '--sample', help='sample', type=float, default=1.0)
    args = parser.parse_args()
    # trainPath = "../data/random_walk/clearscope/clearscope_1115_10_8_average2_2400/train/"
    trainPath = "../data/random_walk/streamspot/streamspot_embedding_929_100/train"
    testPath = "../data/random_walk/streamspot/streamspot_embedding_929_100/test/"
    random_walk_path = "streamspot_embedding_929_100"
    dataset = args.dataset
    # splitRate = args.split
    # valid = args.valid
    sample = args.sample
    valid = 0.25
    splitRate = 0.25
    SEED = 98765432
    random.seed(SEED)
    np.random.seed(SEED)
    train = os.listdir(trainPath)
    test = os.listdir(testPath)
    train_files = [os.path.join(trainPath, f) for f in train]
    test_files_temp = [os.path.join(testPath, f) for f in test]

    cross_validation = 5
    kf = ShuffleSplit(n_splits=cross_validation, test_size=splitRate, random_state=0)
    cv = 0
    # accuracy_list = []
    pr_ret = []
    recall_ret = []
    fpr_ret = []
    now = str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace(":", "_")
    today = datetime.date.today()
    if valid > 0.1:
        kf = ShuffleSplit(n_splits=args.cross_validation, test_size=valid, random_state=0)
        print("\x1b[6;30;42m[STATUS]\x1b[0m Performing {} cross validation".format(args.cross_validation))
        for train_idx, validate_idx in kf.split(train_files):
            training_files = list()  # Training submodels we use
            valid_files = list()
            validNum = int(len(train_idx) * valid)
            trainNum = len(train_idx) - validNum
            test_files = copy.deepcopy(test_files_temp)
            # for tidx in train_idx:
            #    training_files.append(train_files[tidx])
            for i in range(train_idx.shape[0]):
                if i < trainNum:
                    training_files.append(train_files[train_idx[i]])
                else:
                    valid_files.append(train_files[train_idx[i]])
            for vidx in validate_idx:  # Train graphs used as validation
                test_files.append(train_files[vidx])  # Validation graphs are used as test graphs

            print("\x1b[6;30;42m[STATUS] Test {}/{}\x1b[0m:".format(cv, args.cross_validation))

            modelPath = '../data/model_random_walk/streamspot/streamspot_1126/' + random_walk_path + '_BCE_epoch5_' + str(
                today) + '_' + str(splitRate) + "_" + str(valid) + "_" + str(cv)
            print(modelPath)
            print(len(training_files))
            print(len(valid_files))
            print(len(test_files))
            pr, recall, fpr = train_valid(training_files, test_files, modelPath, True, valid_files)
            pr_ret.append(pr)
            recall_ret.append(recall)
            fpr_ret.append(fpr)
            cv += 1
    else:
        for train_idx, test_idx in kf.split(train_files):
            training_files = list()  # Training submodels we use
            test_files = copy.deepcopy(test_files_temp)
            for tr_idx in train_idx:
                training_files.append(train_files[tr_idx])
            for te_idx in test_idx:  # Train graphs used as validation
                test_files.append(train_files[te_idx])  # Validation graphs are used as test graphs
            today = '2022-10-01'
            valid = 0.25
            modelPath = '../data/model_random_walk/streamspot/streamspot_929_walk/streamspot_2022-10-01_batch1_5_100_' + str(
                splitRate) + "_" + str(valid) + "_" + str(cv)
            print(modelPath)
            pr, recall, fpr = train_valid(training_files, test_files, modelPath, False)
            pr_ret.append(pr)
            recall_ret.append(recall)
            cv += 1
    final_data(recall_ret, pr_ret)
