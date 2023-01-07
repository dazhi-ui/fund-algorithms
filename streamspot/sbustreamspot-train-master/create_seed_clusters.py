#!/usr/bin/env python

import argparse
from constants import *
import numpy as np
import random
from medoids import _k_medoids_spawn_once, k_medoids
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

# finding best number of clusters
# http://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters/15376462#15376462

random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Training graph shingle vectors',
                    required=True)
args = vars(parser.parse_args())

input_file = args['input']
with open(input_file, 'r') as f:
    num_shingles, chunk_length = map(int, f.readline().strip().split('\t'))
    shingles = f.readline().strip().split('\t')[1:]

    X = [] # note: row i = graph ID i
    for line in f:
        fields = map(int, line.strip().split('\t'))
        graph_id = fields[0]
        shingle_vector = fields[1:]
        X.append(shingle_vector)

    X = np.array(X)
    dists = squareform(pdist(X, metric='cosine'))
    def distance(a, b):
        return dists[a][b]

    best_n_clusters = -1
    best_silhouette_avg = -1
    best_cluster_labels = None
    best_cluster_centers = None
    for n_clusters in range(2, X.shape[0]):
        for trial in range(NUM_TRIALS):
            # run many trials for a given number of clusters
            _, medoids = _k_medoids_spawn_once(points=range(X.shape[0]),
                                               k=n_clusters,
                                               distance=distance,
                                               max_iterations=1000,
                                               verbose=False)
            cluster_labels = [-1] * X.shape[0]
            for medoid_idx, medoid in enumerate(medoids):
                graphs = medoid.elements
                for graph in graphs:
                    cluster_labels[graph] = medoid_idx 
            cluster_labels = np.array(cluster_labels)

            silhouette_avg = silhouette_score(X, cluster_labels, metric='cosine')
            #print n_clusters, trial, 'silhouette score =', silhouette_avg

            if silhouette_avg > best_silhouette_avg or\
               (silhouette_avg == best_silhouette_avg and\
                n_clusters > best_n_clusters): # favour more clusters
                best_silhouette_avg = silhouette_avg
                best_n_clusters = n_clusters
                best_cluster_labels = cluster_labels
                best_cluster_centers = medoids

    all_cluster_dists = []
    cluster_threshold = [-1] * best_n_clusters
    for cluster_idx in range(best_n_clusters):
        cluster_center = best_cluster_centers[cluster_idx].kernel
        cluster_graphs = best_cluster_centers[cluster_idx].elements

        cluster_dists = [dists[cluster_center][graph] for graph in cluster_graphs
                         if graph != cluster_center]
        all_cluster_dists.extend(cluster_dists)

        # taking mean of cosine distances? seems to be ok
        # see https://en.wikipedia.org/wiki/Rocchio_algorithm
        mean_dist = np.mean(cluster_dists)
        std_dist = np.std(cluster_dists)

        if len(cluster_dists) == 0: # singleton clusters, shouldnt happen
            mean_dist = 0.0
            std_dist = 0.0
            all_cluster_dists.append(0.0)

        cluster_threshold[cluster_idx] = mean_dist + NUM_DEVS * std_dist # P(>) <= 10%
    mean_all_cluster_dists = np.mean(all_cluster_dists)
    std_all_cluster_dists = np.mean(all_cluster_dists)
    all_cluster_threshold = mean_all_cluster_dists + NUM_DEVS * std_all_cluster_dists

    print str(best_n_clusters) + '\t' + str(X.shape[0]) + '\t',
    print str(chunk_length) + '\t',
    print "{:3.4f}".format(all_cluster_threshold)

    for cluster_idx in range(best_n_clusters):
        cluster_graphs = best_cluster_centers[cluster_idx].elements
        threshold = cluster_threshold[cluster_idx]
        print "{:3.4f}".format(threshold) + '\t',
        print '\t'.join([str(graph) for graph in cluster_graphs])
