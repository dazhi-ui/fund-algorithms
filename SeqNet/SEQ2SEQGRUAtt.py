import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
#from LSTM import PrepareData
import pdb
import re
from helper.profile import BestClusterGroup, Model, test_single_graph
from scipy.spatial.distance import pdist, squareform, hamming
import random
import os
import sys
import argparse
from sklearn.model_selection import KFold, ShuffleSplit
import copy
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math

def pairwise_distance(arr, method='hamming'):
    """Wrapper function that calculates the pairwise distance between every
    two elements within the @arr. The metric (@method) is default as hamming.
    squareform function makes it a matrix for easy indexing and accessing. """
    return squareform(pdist(arr, metric=method))

def model_graphs(train_files, model_file, max_cluster_num=6, num_trials=20, max_iterations=1000):
    """Read sketch vectors in @train_files to build submodels. Create one model from each file.
    Returns a dictionary that maps the train file name to its model. """
    # A dictionary of models from each file in @train_files.
    models = dict()
    if model_file:
        savefile = open(model_file, 'a+')
    else:
        print("\33[5;30;42m[INFO]\033[0m Model is not saved, use --save-model to save the model")
    for train_file in train_files:
        with open(train_file, 'r') as f:
            sketches = LoadSketches(f)
            # @dists contains pairwise Hamming distance between two sketches in @sketches.
            try:
                dists = pairwise_distance(sketches)
            except Exception as e:
                print("\33[101m[ERROR]\033[0m Exception occurred in modeling from file {}: {}".format(train_file, e))
                raise RuntimeError("Model builing failed: {}".format(e))
            # Define a @distance function to use to optimize.
            def distance(x, y):
                return dists[x][y]
            best_cluster_group = BestClusterGroup()
            best_cluster_group.optimize(arrs=sketches, distance=distance, max_cluster_num=max_cluster_num, num_trials=num_trials, max_iterations=max_iterations)
            # With the best medoids, we can compute some statistics for the model.
            model = Model(train_file)
            model.construct(sketches, dists, best_cluster_group)
            print("\x1b[6;30;42m[SUCCESS]\x1b[0m Model from {} is done...".format(train_file))
            # Save some model information in DEBUG_INFO if -v is turned on

            # Save model
            if model_file:
                print("\x1b[6;30;42m[STATUS]\x1b[0m Saving the model {} to {}...".format(train_file, model_file))
                save_model(model, train_file, savefile)

            models[train_file] = model
        # Close the file and proceed to the next model.
        f.close()
    if model_file:
        savefile.close()
    return models

def Unicorn():
    model_save_path = None
    models = model_graphs(files, model_save_path)
    for each in files:
        model = models[each]
'''
def LoadSketches(fh):
    sketches = list()
    for line in fh:
        sketch = map(long, line.strip().split())
        sketches.append(sketch)
    return np.array(sketches)
'''
#sample sketches
def LoadSketches(fh):
    sketches = list()
    for line in fh:
        sketch = map(long, line.strip().split())
        sketches.append(sketch)
    #sketchesNew = SequenceSample(sketches)
    return np.array(sketches)

def ExtractLabel(filename):
    res = re.search(r'sketch-([a-z]+)-', filename)
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

'''
def ExtractLabel(filename):
    res = re.search(r'sketch-([a-z]+)-', filename)
    if res:
        label = res.group(1)
    else:
        label = ''
    if label == 'cnn':
        return 0
    if label == 'download':
        return 1
    if label == 'gmail':
        return 2
    if label == 'vgame':
        return 3
    if label == 'youtube':
        return 4
    if label == 'attack':
        return 5
    if filename.find('wget-normal') > -1:
        return 1
    if filename.find('benign') > -1:
        return 1
    if filename.find('attack') > -1:
        return 0
'''

def MinMax(data):
    max_ = data[0][0]
    min_ = data[0][0]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if max_ < data[i][j]:
                max_ = data[i][j]
            if min_ > data[i][j]:
                min_ = data[i][j]
    return min_, max_

def MinMaxScaler(dataList, min_ = -1, max_ = -1):
    if min_ < 0:
        min_, max_ = MinMax(dataList[0])
        for data in dataList:
            minT, maxT = MinMax(data)
            if minT < min_:
                min_ = minT
            if maxT > max_:
                max_ = maxT
    dataListNew = []
    for data in dataList:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j] - min_)/(max_ - min_)
        dataListNew.append(data)
    return min_, max_, dataListNew

def FeaSimilar(feature1, feature2):
    for i in range(len(feature1)):
        if feature1[i] != feature2[i]:
            return False
    return True

def Variable(data):
    for i in range(1, data.shape[0]):
        print np.linalg.norm(data[i] - data[i-1]),
    print '\n',

def DelSimilar(dataList):
    dataListNew = []
    for data in dataList:
        Variable(data)
        seq = data.tolist()
        seqNew = [seq[0]]
        for i in range(1, len(seq)):
            if not FeaSimilar(seq[i-1], seq[i]):
                seqNew.append(seq[i])
        dataListNew.append(np.array(seqNew))
    return dataListNew

def SeqTrans(dataset):
    for data in dataset:
        base = data[0]
        for i in range(len(data)):
            print np.linalg.norm(data[i] - base),
            print ' ',
        print '\n',

def PrepareDataSample(trainFiles, testFiles, maxLen):
    #train
    models = model_graphs(trainFiles, None)
    sketchesList = []
    Label = []
    for train in trainFiles:
        model = models[train]
        sketchList = []
        for current_evolution_idx in range(len(model.get_evolution())):
            current_cluster_idx = model.get_evolution()[current_evolution_idx]
            current_medoid = model.get_medoids()[current_cluster_idx]
            sketchList.append(current_medoid)
        sketchesList.append(np.array(sketchList))
        Label.append(ExtractLabel(train))
    min_, max_, sketchesList = MinMaxScaler(sketchesList)
    dataset = StreamDataset(sketchesList, maxLen, Label)
    trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=1)

    #test
    models = model_graphs(testFiles, None)
    sketchesList = []
    Label = []
    for test in testFiles:
        model = models[test]
        sketchList = []
        for current_evolution_idx in range(len(model.get_evolution())):
            current_cluster_idx = model.get_evolution()[current_evolution_idx]
            current_medoid = model.get_medoids()[current_cluster_idx]
            sketchList.append(current_medoid)
        sketchesList.append(np.array(sketchList))
        Label.append(ExtractLabel(train))
    min_, max_, sketchesList = MinMaxScaler(sketchesList, min_, max_)
    dataset = StreamDataset(sketchesList, maxLen, Label)
    testDataLoader = torch.utils.data.DataLoader(dataset, batch_size=1)
    return trainDataLoader, testDataLoader

def dist(sketchesList):
    for sketches in sketchesList:
        print sketches.shape[0]
        for i in range(sketches.shape[0]-1):
            print np.linalg.norm(sketches[i+1] - sketches[i]),
        print ' '
        pdb.set_trace()

def SequenceSample(seq, ratio=0.01):
    seqLen = len(seq)
    N = range(seqLen)
    number = int(math.ceil(seqLen*ratio))
    sample = np.random.choice(N, size=number, replace=False)
    sample.sort()
    #seqNew = seq[sample]
    seqNew = []
    for idx in sample:
        seqNew.append(seq[idx])
    return seqNew

def PrepareData(trainFiles, testFiles, validFiles, maxLen):
    sketchesList = []
    trainLabel = []
    for train in trainFiles:
        with open(train, 'r') as f:
            sketches = LoadSketches(f)
            sketchesList.append(sketches)
            trainLabel.append(ExtractLabel(train))
    #for ske in sketchesList:
    #    print ske.shape
    min_, max_, sketchesList = MinMaxScaler(sketchesList)
    #sample
    #sketchesList = DelSimilar(sketchesList)
    #SeqTrans(sketchesList)
    dataset = StreamDataset(sketchesList, maxLen, trainLabel)
    trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=1)

    if validFiles != None:
        sketchesList = []
        validLabel = []
        for valid in validFiles:
            with open(valid, 'r') as f:
                sketches = LoadSketches(f)
                sketchesList.append(sketches)
                validLabel.append(ExtractLabel(valid))
        #for ske in sketchesList:
        #    print ske.shape
        min_, max_, sketchesList = MinMaxScaler(sketchesList)
        dataset = StreamDataset(sketchesList, maxLen, validLabel)
        validDataLoader = torch.utils.data.DataLoader(dataset, batch_size=1)
    else:
        validDataLoader = None


    sketchesList = []
    testLabel = []
    for test in testFiles:
        with open(test, 'r') as f:
            sketches = LoadSketches(f)
            sketchesList.append(sketches)
            testLabel.append(ExtractLabel(test))
    min_, max_, sketchesList = MinMaxScaler(sketchesList, min_, max_)
    #dist(sketchesList)
    #sample
    #pdb.set_trace()
    #sketchesList = DelSimilar(sketchesList)
    #pdb.set_trace()
    #SeqTrans(sketchesList)
    dataset = StreamDataset(sketchesList, maxLen, testLabel)
    testDataLoader = torch.utils.data.DataLoader(dataset, batch_size=1)
    return trainDataLoader, testDataLoader, validDataLoader    

class StreamDataset(torch.utils.data.Dataset):
    def __init__(self, seq, maxLen, label):
        self.maxLen = maxLen
        seqPad = np.zeros((len(seq), maxLen, seq[0].shape[1]))
        self.length = []
        for indx, each in enumerate(seq):
            seqPad[indx, :each.shape[0],:] = each
            self.length.append(each.shape[0])
        self.seq = seqPad
        self.label = label

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seqLen = self.length[idx]
        return {'seq': self.seq[idx, :seqLen, :],
                'label': self.label[idx]}


class Encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers=1):
        super(Encoder, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.encoder = nn.GRU(inputSize, hiddenSize, num_layers=numLayers)

    def forward(self, seq):
        output, hidden = self.encoder(seq)
        #return output[-1]
        #output = output[-1]
        #return F.normalize(output, p=2, dim=1)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers=1):
        super(Decoder, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.decoder = nn.GRU(inputSize, hiddenSize, num_layers=numLayers)
        
    def forward(self, seq, hidden):
        output, hidden = self.decoder(seq, hidden)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers=1):
        super(Attention, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        self.GRU = nn.GRU(inputSize, hiddenSize, num_layers=numLayers)
        #self.attention = nn.Linear(self.hiddenSize*2, 1)

    def forward(self, seq):
        output, hidden = self.GRU(seq)
        return output.permute(1, 0, 2)

'''
class Seq2Seq(nn.Module):
    def __init__(self, inputSize, hiddenSize, teacherForce=0.5):
        super(Seq2Seq, self).__init__()

        self.embedding = nn.Linear(inputSize, hiddenSize)
        self.encoder = Encoder(hiddenSize, hiddenSize, 1)
        self.decoder = Decoder(hiddenSize, hiddenSize, 1)
        self.teacherForce = teacherForce

    def forward(self, seq):
        #pdb.set_trace()
        seq1 = self.embedding(seq)
        seqPermute = seq1.permute(1, 0, 2)
        length = seq1.shape[0]
        batchSize = seq1.shape[1]
        featureDim = seq1.shape[2]
        res = torch.zeros(length, batchSize, featureDim).cuda()
        encoderOutput, hidden = self.encoder(seqPermute)

        teacher = True if random.random() < teacherForce else False

        decoderHidden = hidden
        decoderInput = encoderOutput[-1]
        if teacher:
            for i in range(length):
                decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden, encoderOutput)
                decoderInput = seq[i+1]
        else:
            for i in range(length):
                decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden, encoderOutput)
                decoderInput = decoderOutput
        #feature = feature.unsqueeze(0)
        #feature_ = feature
        for i in range(length):
            output = self.decoder(feature)
            res[i] = output
        return res.permute(1, 0, 2), seq1

'''
class Seq2Seq(nn.Module):
    def __init__(self, inputSize, hiddenSize, teacherForce=0.0):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(hiddenSize, hiddenSize, 1)
        self.decoder = Decoder(hiddenSize, hiddenSize, 1)
        self.embedding = nn.Linear(inputSize, hiddenSize)
        self.teacherForce = teacherForce
        #self.att = nn.Linear(2*hiddenSize, 1)
        self.att = Attention(2*hiddenSize, 1)
        self.local = True
        self.localLen = 3 #7

    def forward(self, seq):
        #pdb.set_trace()
        seq1 = self.embedding(seq)
        #seq1 = F.relu(seq1)
        #seq1 = F.normalize(seq1, p=2, dim=2)
        #seq1 = F.dropout(seq1, p=0.5)
        seq = seq1.permute(1, 0, 2)
        length = seq.shape[0]
        batchSize = seq.shape[1]
        featureDim = seq.shape[2]
        res = torch.zeros(length, batchSize, featureDim).cuda()
        
        #attention
        encoderOutput, hidden = self.encoder(seq)
        #feature = feature.unsqueeze(0)
        feature_ = encoderOutput[-1]
        decoderInput = encoderOutput[-1].unsqueeze(0)
        encoderOutput = encoderOutput.permute(1, 0, 2)

        teacher = True if random.random() < self.teacherForce else False

        if self.local:
            if teacher:
                for i in range(length):
                    output, hidden = self.decoder(decoderInput, hidden)
                    left = max(0, i - self.localLen)
                    right = min(length-1, i + self.localLen)
                    outputs = output.repeat(1, right-left+1, 1)
                    outputs = torch.cat([outputs, encoderOutput[:, left:right+1, :]], dim=2)
                    weight = F.softmax(self.att(outputs.permute(1, 0, 2)), dim=1)
                    res[i] = output[0] + torch.sum(encoderOutput*weight, dim=1)
                    decoderInput = seq[i].unsqueeze(0)
            else:
                for i in range(length):
                    output, hidden = self.decoder(decoderInput, hidden)
                    left = max(0, i - self.localLen)
                    right = min(length-1, i + self.localLen)
                    outputs = output.repeat(1, right-left+1, 1)
                    outputs = torch.cat([outputs, encoderOutput[:, left:right+1, :]], dim=2)
                    #outputs = output.repeat(1, length, 1)
                    #outputs = torch.cat([outputs, encoderOutput],dim=2)
                    #weight = F.softmax(self.att(outputs), dim=1)
                    weight = F.softmax(self.att(outputs.permute(1, 0, 2)), dim=1)
                    res[i] = output[0] + torch.sum(encoderOutput[:, left:right+1, :]*weight, dim=1)
                    #output = output.squeeze(0)
                    #feature = self.embedding(output)
                    #feature = feature.unsqueeze(0)
                    #res[i] = output[0]
                    decoderInput = output
        else:
            if teacher:
                for i in range(length):
                    output, hidden = self.decoder(decoderInput, hidden)
                    outputs = output.repeat(1, length, 1)
                    outputs = torch.cat([outputs, encoderOutput],dim=2)
                    weight = F.softmax(self.att(outputs), dim=1)
                    res[i] = output[0] + torch.sum(encoderOutput*weight, dim=1)
                    decoderInput = seq[i].unsqueeze(0)
            else:
                for i in range(length):
                    output, hidden = self.decoder(decoderInput, hidden)
                    outputs = output.repeat(1, length, 1)
                    outputs = torch.cat([outputs, encoderOutput],dim=2)
                    weight = F.softmax(self.att(outputs), dim=1)
                    res[i] = output[0] + torch.sum(encoderOutput*weight, dim=1)
                    #output = output.squeeze(0)
                    #feature = self.embedding(output)
                    #feature = feature.unsqueeze(0)
                    #res[i] = output[0]
                    decoderInput = output
            #res = F.normalize(res, p=2, dim=2)
        return res.permute(1, 0, 2), seq1, feature_


        '''no attention
        feature, hidden = self.encoder(seq)
        feature = feature[-1]
        #feature = feature.unsqueeze(0)
        feature_ = feature
        output = feature.unsqueeze(0)

        teacher = True if random.random() < self.teacherForce else False

        if teacher:
            for i in range(length):
                output, hidden = self.decoder(output, hidden)
                res[i] = output[0]
                output = seq[i].unsqueeze(0)
        else:
            for i in range(length):
                output, hidden = self.decoder(output, hidden)
                #output = output.squeeze(0)
                #feature = self.embedding(output)
                #feature = feature.unsqueeze(0)
                res[i] = output[0]
        res = F.normalize(res, p=2, dim=2)
        return res.permute(1, 0, 2), seq1, feature_
        '''

def Train(trainDataset, epochs, model, gpu):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        lossAve = 0.0
        model.train()
        print 'Epoch:' + str(epoch)
        for idx, data in enumerate(trainDataset):
            model.zero_grad()
            seq = data['seq'].float()
            if gpu:
                seqCuda = seq.cuda()
            y, inputs, feature = model(seqCuda)
            loss = loss_fn(y, inputs)
            loss.backward()
            optimizer.step()
            lossAve += loss
        print lossAve
        if lossAve < 0.01:
            break
    return model

def Test(dataset, model):
    model.eval()
    labelList = []
    preList = []
    resList = []
    feature = np.zeros((300, 100))

    for idx, data in enumerate(dataset):
        seq = data['seq'].float()
        label = data['label']
        y, fea = model(seq)
        _, ind = torch.max(y, 1)
        preList.append(ind.data.numpy())
        labelList.append(label.numpy())
        resList.append(y.data.numpy())
        feature[idx] = fea.data.numpy()

    for i in range(len(labelList)):
        print labelList[i],
        print preList[i],
        print resList[i]

    count = Accuracy(labelList, preList)
    #Visualize(feature, labelList)

    return count

def ExtractFeature(dataset, model, length, gpu):
    model.eval()
    feature = np.zeros((length, 100))
    label = np.zeros(length)

    for idx, data in enumerate(dataset):
        seq = data['seq'].float()
        if gpu:
            seq = seq.cuda()
        #y, fea = model(seq)
        _, _, fea = model(seq)
        output = fea[0]
        if gpu:
            feature[idx] = output.data.cpu().numpy()
            label[idx] = data['label'].data.cpu().numpy()
        else:
            feature[idx] = output.data.numpy()
            label[idx] = data['label'].data.numpy()

    return feature, label

def Threshold(center, feature, label, realLabel):
    threshold = np.zeros(center.shape[0])
    #statistic = []
    #for i in range(center.shape[0]):
    #    statistic.append([])
    for i in range(label.shape[0]):
        dis = np.linalg.norm(feature[i]-center[label[i]])
        #for j in range(center.shape[0]):
        #    print np.linalg.norm(feature[i]-center[j]),
        #print label[i]
        if dis > threshold[label[i]]:
            threshold[label[i]] = dis
        #statistic[label[i]].append(dis)
    #for i in range(center.shape[0]):
    #    print statistic[i].sort()
    return threshold

def distance(center, feature, label, threshold):
    res = np.zeros(label.shape[0])
    #statistic = []
    #for i in range(center.shape[0]+1):
    #    statistic.append([])
    for i in range(label.shape[0]):
        dis = 100000
        pos = -1
        for j in range(center.shape[0]):
            d = np.linalg.norm(feature[i]-center[j])
            print d,
            if d < dis:
                dis = d
                pos = j
        print label[i]
        if dis > threshold[pos]:
            res[i] = center.shape[0]
        else:
            res[i] = pos
        #statistic[label[i]].append(dis)
        #print label[i]
    #for i in range(center.shape[0]+1):
    #    print statistic[i].sort()

    return res


def Cluster(trainFiles, testFiles, model, gpu, validFiles):
    #trainDataset, testDataset = PrepareData(trainFiles, testFiles, 1600)
    clusterNum = 14
    trainDataset, testDataset, validDataset = PrepareData(trainFiles, testFiles, validFiles, 1600)
    trainFeature, trainLabel = ExtractFeature(trainDataset, model, len(trainFiles), gpu)
    if validFiles:
        validFeature, validLabel = ExtractFeature(validDataset, model, len(validFiles), gpu)
    testFeature, testLabel = ExtractFeature(testDataset, model, len(testFiles), gpu)
    #feature = np.concatenate([trainFeature, testFeature])
    #label = np.concatenate([trainLabel, testLabel])
    #Visualize(feature, label)
    normal = KMeans(n_clusters=clusterNum).fit(trainFeature)
    center = normal.cluster_centers_


def Detection(trainFiles, testFiles, modelPath, new, validFiles=None):
    gpu = torch.cuda.is_available()
    #trainDataset, testDataset = PrepareData(trainFiles, testFiles, 1000)
    trainDataset, testDataset, validDataset = PrepareData(trainFiles, testFiles, validFiles, 1600)
    inputSize = 2000
    hiddenSize = 100
    epochs = 1000
    #modelPath = 'model/seq2seq'
    model = Seq2Seq(inputSize, hiddenSize)
    if gpu:
        model = model.cuda()
    if new:
        model = Train(trainDataset, epochs, model, gpu)
        torch.save(model.state_dict(), modelPath)
    else:
        model.load_state_dict(torch.load(modelPath))
        Cluster(trainFiles, testFiles, model, gpu, validFiles)

def CalPrecision(tp, fp, tn, fn):
    return tp / (tp + fp)

def CalRecall(tp, fp, tn, fn):
    return tp / (tp + fn)

def Split(dataList, LabelList):
    normalList = []
    abnormalList = []
    for i in range(len(LabelList)):
        if LabelList[i] == 0:
            abnormalList.append(dataList[i])
        elif LabelList[i] == 1:
            normalList.append(dataList[i])
    return normalList, abnormalList

def FindNearest(center, feature):
    distList = []
    posList = []
    for i in range(feature.shape[0]):
        dis = 100000
        pos = -1
        for j in range(center.shape[0]):
            d = np.linalg.norm(feature[i]-center[j])
            if d < dis:
                dis = d
                pos = j
        distList.append(dis)
        posList.append(pos)
    return distList, posList


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainPath', help='absolute path to the directory that contains all training sketches', required=True)
    parser.add_argument('-u', '--testPath', help='absolute path to the directory that contains all test sketches', required=True)
    parser.add_argument('-c', '--cross-validation', help='number of cross validation we perform (use 0 to turn off cross validation)', type=int, default=5)
    parser.add_argument('-d', '--dataset', help='dataset', default='streamspot')
    parser.add_argument('-s', '--split', help='split rate', type=float, default=0.4)
    parser.add_argument('-v', '--valid', help='valid', type=float, default=0.4)
    parser.add_argument('-a', '--sample', help='sample', type=float, default=1.0)
    args = parser.parse_args()


    #trainPath = '../../data/train_wget/'
    #testPath = '../../data/test_wget_baseline'
    #dataset = 'wget'
    trainPath = args.trainPath
    testPath = args.testPath
    dataset = args.dataset
    splitRate = args.split
    valid = args.valid
    sample = args.sample

    SEED = 98765432
    random.seed(SEED)
    np.random.seed(SEED)
    train = os.listdir(trainPath)
    test = os.listdir(testPath)
    train_files = [os.path.join(trainPath, f) for f in train]
    test_files = []
    test_files_temp = [os.path.join(testPath, f) for f in test]
    cross_validation = 5
    kf = ShuffleSplit(n_splits=cross_validation, test_size=splitRate, random_state=0)
    cv = 0
    if valid > 0:
        kf = ShuffleSplit(n_splits=args.cross_validation, test_size=splitRate, random_state=0)
        print("\x1b[6;30;42m[STATUS]\x1b[0m Performing {} cross validation".format(args.cross_validation))
        cv = 0  # counter of number of cross validation tests
        for train_idx, validate_idx in kf.split(train_files):
            training_files = list()                     # Training submodels we use
            valid_files = list()
            validNum = int(len(train_idx)*valid)
            trainNum = len(train_idx) - validNum
            test_files = copy.deepcopy(test_files_temp)
            #for tidx in train_idx:
            #    training_files.append(train_files[tidx])
            for i in range(train_idx.shape[0]):
                if i < trainNum:
                    training_files.append(train_files[train_idx[i]])
                else:
                    valid_files.append(train_files[train_idx[i]])
            for vidx in validate_idx:                   # Train graphs used as validation
                test_files.append(train_files[vidx])    # Validation graphs are used as test graphs

            print("\x1b[6;30;42m[STATUS] Test {}/{}\x1b[0m:".format(cv, args.cross_validation))
            modelPath = 'model/' + args.dataset + '/' + 'seq2seqGRUAtt1' + str(splitRate)  + str(cv) + str(sample)
            print modelPath


            Detection(training_files, test_files, modelPath, False, valid_files)
            cv += 1
    else:
        for train_idx, validate_idx in kf.split(train_files):
            training_files = list()                     # Training submodels we use
            test_files = copy.deepcopy(test_files_temp)
            for tidx in train_idx:
                training_files.append(train_files[tidx])
            for vidx in validate_idx:                   # Train graphs used as validation
                test_files.append(train_files[vidx])    # Validation graphs are used as test graphs
            #modelPath = 'model/' + dataset + '/SEQ2SEQ3' + str(cv)
            modelPath = 'model/' + args.dataset + '/' + 'seq2seqGRUAtt0' + str(splitRate)  + str(cv)# + str(sample)
            print modelPath
            Detection(training_files, test_files, modelPath, False)
            cv += 1

