#Author: Gentry Atkinson
#Organization: Texas University
#Data: 30 December, 2020
#Expand the tsar test to additional models using the features sets from "tsar test"

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#Models to investigate:
#   KNN
#   C4.5
#   Naive Bayes

if __name__ == "__main__":
    unimib_feats_noisy = np.genfromtxt("test/visualizations_for_unimib_sup/UniMiB_sup_feat.csv", delimiter=',')
    unimib_bad_guys =  np.genfromtxt("test/visualizations_for_unimib_sup/UniMiB_identified_bad.csv", delimiter=',')
    unimib_feats_clean = np.delete(unimib_feats_noisy, unimib_bad_guys, 0)
    log = open("data_cleaning_experiments_results.txt")

    log.write("------- UniMiB Experiments-----------")

    unimib_noisy_knn = KNeighborsClassifier(n_neighbors=3)
    unimib_clean_knn = KNeighborsClassifier(n_neighbors=3)
