#Author: Gentry Atkinson
#Organization: Texas University
#Data: 04 November, 2020
#Create visualizations of 6 instances each from 2 UniMiB sets and the UCI set
#  using 3 methods of feature extraction

import random as rand
import numpy as np
from tsar import get_supervised_features, get_unsupervised_features, print_graph_for_instance, get_NN_for_dataset, preprocess_raw_data_and_labels
from utils.ts_feature_toolkit import get_features_for_set
from import_datasets import get_unimib_data, get_uci_data
from sklearn.manifold import TSNE as tsne

def add_label_noise_to_set(y, percentNoise=5, saveToFile=False, filename="noisy_labels.csv"):
    bad = np.array()
    for i in y:
        if rand.randint < percentNoise:
            #do the mislabel

if __name__ == "__main__":
    print("Preparing 54 visualizations of clean data...")
    print("---UniMiB---")
    for s in ["adl", "two_classes"]:
        X, y, labels = get_unimib_data(s)
        X, y = preprocess_raw_data_and_labels(X, y)

        print("Traditional Features")
        NUM_INSTANCES = len(X)
        feat_x = get_features_for_set(X[:,0,:], num_samples=NUM_INSTANCES)
        feat_y = get_features_for_set(X[:,1,:], num_samples=NUM_INSTANCES)
        feat_z = get_features_for_set(X[:,2,:], num_samples=NUM_INSTANCES)
        NUM_FEATURES = feat_x.shape[1]
        feat = np.zeros((NUM_INSTANCES,3*NUM_FEATURES))
        feat[:, 0:NUM_FEATURES] = feat_x[:,:]
        feat[:, NUM_FEATURES:2*NUM_FEATURES] = feat_y[:,:]
        feat[:, 2*NUM_FEATURES:3*NUM_FEATURES] = feat_z[:,:]

        vis = tsne(n_components=2, n_jobs=8).fit_transform(feat)
        neighbors = get_NN_for_dataset(feat)
        print("Correct Labels")
        for i in range(3):
            instance = rand.randint(0, len(X))
            print(i+1,": point ", instance)
            print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/unimib_"+s+"_trad_correct_"+str(i)+".png", False)

        print("Mislabeled")
        for i in range(3):
            instance = rand.randint(0, len(X))
            print(i+1,": point ", instance)
            print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/unimib_"+s+"_trad_mislabeled_"+str(i)+".png", True)

        print("Supervised Features")
        feat = get_supervised_features(X, y)
        vis = tsne(n_components=2, n_jobs=8).fit_transform(feat)
        neighbors = get_NN_for_dataset(feat)

        print("Correct Labels")
        for i in range(3):
            instance = rand.randint(0, len(X))
            print(i+1,": point ", instance)
            print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/unimib_"+s+"_sup_correct_"+str(i)+".png", False)

        print("Mislabeled")
        for i in range(3):
            instance = rand.randint(0, len(X))
            print(i+1,": point ", instance)
            print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/unimib_"+s+"_sup_mislabeled_"+str(i)+".png", True)

        print("Unsupervised Features")
        feat = get_unsupervised_features(X)
        vis = tsne(n_components=2, n_jobs=8).fit_transform(feat)
        neighbors = get_NN_for_dataset(feat)

        print("Correct Labels")
        for i in range(3):
            instance = rand.randint(0, len(X))
            print(i+1,": point ", instance)
            print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/unimib_"+s+"_unsup_correct_"+str(i)+".png", False)

        print("Mislabeled")
        for i in range(3):
            instance = rand.randint(0, len(X))
            print(i+1,": point ", instance)
            print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/unimib_"+s+"_unsup_mislabeled_"+str(i)+".png", True)

    print("---UCI---")
    X, y, labels = get_uci_data()
    X, y = preprocess_raw_data_and_labels(X, y)

    print("Traditional Features")
    NUM_INSTANCES = len(X)
    feat_x = get_features_for_set(X[:,0,:], num_samples=NUM_INSTANCES)
    feat_y = get_features_for_set(X[:,1,:], num_samples=NUM_INSTANCES)
    feat_z = get_features_for_set(X[:,2,:], num_samples=NUM_INSTANCES)
    NUM_FEATURES = feat_x.shape[1]
    feat = np.zeros((NUM_INSTANCES,3*NUM_FEATURES))
    feat[:, 0:NUM_FEATURES] = feat_x[:,:]
    feat[:, NUM_FEATURES:2*NUM_FEATURES] = feat_y[:,:]
    feat[:, 2*NUM_FEATURES:3*NUM_FEATURES] = feat_z[:,:]

    vis = tsne(n_components=2, n_jobs=8).fit_transform(feat)
    neighbors = get_NN_for_dataset(feat)
    print("Correct Labels")
    for i in range(3):
        instance = rand.randint(0, len(X))
        print(i+1,": point ", instance)
        print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/uci_trad_correct_"+str(i)+".png", False)

    print("Mislabeled")
    for i in range(3):
        instance = rand.randint(0, len(X))
        print(i+1,": point ", instance)
        print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/uci_trad_mislabeled_"+str(i)+".png", True)

    print("Supervised Features")
    feat = get_supervised_features(X, y)
    vis = tsne(n_components=2, n_jobs=8).fit_transform(feat)
    neighbors = get_NN_for_dataset(feat)

    print("Correct Labels")
    for i in range(3):
        instance = rand.randint(0, len(X))
        print(i+1,": point ", instance)
        print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/uci_sup_correct_"+str(i)+".png", False)

    print("Mislabeled")
    for i in range(3):
        instance = rand.randint(0, len(X))
        print(i+1,": point ", instance)
        print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/uci_sup_mislabeled_"+str(i)+".png", True)

    print("Unsupervised Features")
    feat = get_unsupervised_features(X)
    vis = tsne(n_components=2, n_jobs=8).fit_transform(feat)
    neighbors = get_NN_for_dataset(feat)

    print("Correct Labels")
    for i in range(3):
        instance = rand.randint(0, len(X))
        print(i+1,": point ", instance)
        print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/uci_unsup_correct_"+str(i)+".png", False)

    print("Mislabeled")
    for i in range(3):
        instance = rand.randint(0, len(X))
        print(i+1,": point ", instance)
        print_graph_for_instance(X, y, labels, instance, feat, neighbors, vis, False, True, "imgs/uci_unsup_mislabeled_"+str(i)+".png", True)

    print("Finished")
