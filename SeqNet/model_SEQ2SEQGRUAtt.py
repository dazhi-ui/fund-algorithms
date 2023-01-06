#!/usr/bin/env python

##########################################################################################
# Some of the code is adapted from:
# https://github.com/sbustreamspot/sbustreamspot-train/blob/master/create_seed_clusters.py
##########################################################################################
import argparse
import numpy as np
import random
import os
import sys
import pdb
from helper.profile import BestClusterGroup, Model, test_single_graph
from scipy.spatial.distance import pdist, squareform, hamming
from sklearn.model_selection import KFold, ShuffleSplit
from SEQ2SEQGRUAtt import Detection
import copy
import datetime

# This global dictionary is populated
# with useful debugging information if
# -v is set to be True in the command line
DEBUG_INFO = None


def save_model(model, model_name, fh):
    """Save a model with model name as the training file name. 
    Models are saved to a file @fh. """
    fh.write("model: {}\n".format(model_name))
    num_cluster = len(model.medoids)
    fh.write("cluster: {}\n".format(num_cluster))
    for medoid in model.medoids:
        fh.write("medoid: ")
        for elem in medoid:
            fh.write("{} ".format(int(float(elem))))
        fh.write("\n")
    fh.write("mean: ")
    for mean in model.mean_thresholds:
        fh.write("{} ".format(float(mean)))
    fh.write("\n")
    fh.write("max: ")
    for max in model.max_thresholds:
        fh.write("{} ".format(float(max)))
    fh.write("\n")
    fh.write("std: ")
    for std in model.stds:
        fh.write("{} ".format(float(std)))
    fh.write("\n")
    fh.write("evolution: ")
    for evol in model.evolution:
        fh.write("{} ".format(evol))
    fh.write("\n")


def load_sketches(fh):
    """Load sketches in a file from the handle @fh to memory as numpy arrays. """
    sketches = list()
    for line in fh:
        sketch = map(long, line.strip().split())
        sketches.append(sketch)
    return np.array(sketches)


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
            sketches = load_sketches(f)
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
            if isinstance(DEBUG_INFO, dict):
                DEBUG_INFO[train_file] = model.get_members()

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


def test_graphs(test_files, models, metric, num_stds):
    """Test all sketch vectors in @test_files using the @models
    built from model_training_graphs. """
    total_graphs_tested = 0.0
    tp = 0.0 # true positive (intrusion and alarmed)
    tn = 0.0 # true negative (not intrusion and not alarmed)
    fp = 0.0 # false positive (not intrusion but alarmed)
    fn = 0.0 # false negative (intrusion but not alarmed)
    
    printout = ""
    for test_file in test_files:
        with open(test_file, 'r') as f:
            # if DEBUG_INFO exists, then we will
            # track debugging information for
            # each test graph. The per-graph debugging
            # information is stored in a dictionary
            test_info = None
            if isinstance(DEBUG_INFO, dict):
                test_info = dict()
            sketches = load_sketches(f)
            abnormal, max_abnormal_point, num_fitted_model, fitted_models = test_single_graph(sketches, models, metric, num_stds, test_info)
            if isinstance(DEBUG_INFO, dict):
                DEBUG_INFO[test_file] = test_info
        f.close()
        total_graphs_tested += 1
        if not abnormal: # The graph is considered normal
            printout += "{} is NORMAL fitting {}/{} models\n".format(test_file, num_fitted_model, len(models))
            if "attack" not in test_file: # NOTE: file name should include "attack" to indicate the oracle
                tn = tn + 1
            else:
                fn = fn + 1
        else:
            printout += "{} is ABNORMAL at {}\n".format(test_file, max_abnormal_point)
            if "attack" in test_file:
                tp = tp + 1
            else:
                fp = fp + 1
        printout += "{}\n".format(fitted_models)
    if (tp + fp) == 0:
        precision = None
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        recall = None
    else:
        recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if not precision or not recall or (precision + recall) == 0:
        f_measure = None
    else:
        f_measure = 2 * (precision * recall) / (precision + recall)
    return precision, recall, accuracy, f_measure, printout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-dir', help='absolute path to the directory that contains all training sketches', required=True)
    parser.add_argument('-u', '--test-dir', help='absolute path to the directory that contains all test sketches', required=True)
    parser.add_argument('-m', '--metric', choices=['mean', 'max', 'both'], default='both',
            help='threshold metric to use to calculate the mean or max of the cluster distances between cluster members and the medoid')
    parser.add_argument('-n', '--num-stds', choices=np.arange(0, 5.0, 0.1), type=float,
            help='the number of standard deviations above the threshold to tolerate')
    parser.add_argument('-S', '--seed', help='seed for random number generator', type=int, default=98765432)
    parser.add_argument('-p', '--model-path', help='file path to save the model', default='model.txt')
    parser.add_argument('-c', '--cross-validation', help='number of cross validation we perform (use 0 to turn off cross validation)', type=int, default=5)
    #parser.add_argument('-v', '--verbose', help='produce debugging information', action='store_true')
    parser.add_argument('-d', '--dataset', help='dataset', default='streamspot')
    parser.add_argument('-s', '--split', help='split rate', type=float, default=0.4)
    parser.add_argument('-v', '--valid', help='valid', type=float, default=0.4)
    parser.add_argument('-a', '--sample', help='sample', type=float, default=1.0)
    args = parser.parse_args()

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    print("\33[5;30;42m[INFO]\033[0m Random number seed: {}".format(SEED))

    #if args.verbose:
    #    print("\33[5;30;42m[INFO]\033[0m Debugging information will be collected. You can access it programmatically")
    #    DEBUG_INFO = dict()

    train = os.listdir(args.train_dir)
    train_files = [os.path.join(args.train_dir, f) for f in train]
    test = os.listdir(args.test_dir)
    test_files = []
    test_files_temp = [os.path.join(args.test_dir, f) for f in test]
    splitRate = args.split
    valid = args.valid
    sample = args.sample
    
    #lstm
    #Detection(train_files, test_files)
    #pdb.set_trace()

    # Determine metric to use
    if args.metric is 'both':
        #metric_config = ['mean', 'max']
        metric_config = ['mean']
    else:
        metric_config = [args.metric]
    # Determine the number of standard deviations to use
    if not args.num_stds:    # If this argument is not given, we explore different possible configurations.
        #std_config = np.arange(0, 5.0, 0.1)
        std_config = np.arange(2, 3.5, 0.1)
    else:
        std_config = [args.num_stds]
    # Train (all training graphs) #
    #models = model_graphs(train_files, model_save_path)

    #for each in train_files:
    #    print models[each].get_evolution()
    #pdb.set_trace()
    
    # Perform K-fold cross validation, unless turned off
    if args.cross_validation == 0:
        print("\33[5;30;42m[INFO]\033[0m No cross validation specified, use --cross-validation")
        # Model (all training_files)
        submodels = list()
        for _, model in models.items():
            submodels.append(model)
        for tm in metric_config:
            for ns in std_config:
                precision, recall, accuracy, f_measure, printout = test_graphs(test_files, submodels, tm, ns)
                print("Metric: {}\tSTD: {}".format(tm, ns))
                print("Accuracy: {}\tPrecision: {}\tRecall: {}\tF-1: {}".format(accuracy, precision, recall, f_measure))
                print("{}".format(printout))
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
            modelPath = 'model/' + args.dataset + '/' + 'seq2seqGRUAtt1' + str(splitRate)  + str(cv)
            print modelPath


            Detection(training_files, test_files, modelPath, True, valid_files)
            cv += 1
    else:
        kf = ShuffleSplit(n_splits=args.cross_validation, test_size=splitRate, random_state=0)
        print("\x1b[6;30;42m[STATUS]\x1b[0m Performing {} cross validation".format(args.cross_validation))
        cv = 0  # counter of number of cross validation tests
        for train_idx, validate_idx in kf.split(train_files):
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print time
            training_files = list()                     # Training submodels we use
            test_files = copy.deepcopy(test_files_temp)
            for tidx in train_idx:
                training_files.append(train_files[tidx])
            for vidx in validate_idx:                   # Train graphs used as validation
                test_files.append(train_files[vidx])    # Validation graphs are used as test graphs

            print("\x1b[6;30;42m[STATUS] Test {}/{}\x1b[0m:".format(cv, args.cross_validation))
            modelPath = 'model/' + args.dataset + '/' + 'seq2seqGRUAtt0' + str(splitRate)  + str(cv)# + str(sample)
            print modelPath

            Detection(training_files, test_files, modelPath, True)
            #pdb.set_trace()
            cv += 1
            '''
            #breaak

            for tm in metric_config:
                for ns in std_config:
                    precision, recall, accuracy, f_measure, printout = test_graphs(test_files, submodels, tm, ns)
                    print("Metric: {} STD: {}".format(tm, ns))
                    print("Accuracy: {}\tPrecision: {}\tRecall: {}\tF-1: {}".format(accuracy, precision, recall, f_measure))
                    print("{}".format(printout))
            cv += 1
            '''

    # Debug print for Visicorn
    #if args.verbose:
    #    for tf in train_files:
    #        print(tf)
    #        print(DEBUG_INFO[tf])
    #    for tf in test_files:
    #        print(tf)
    #        print(DEBUG_INFO[tf])
    print("\x1b[6;30;42m[SUCCESS]\x1b[0m Unicorn is finished")

