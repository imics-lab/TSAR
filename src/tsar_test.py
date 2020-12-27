#Author: Gentry Atkinson
#Organization: Texas University
#Data: 10 December, 2020
#Use tsar to test training on cleaned data

import sys
import os.path
import numpy as np
from tsar import get_supervised_features, get_unsupervised_features, check_dataset, print_graph_for_instance_two_class, preprocess_raw_data_and_labels
from import_datasets import get_unimib_data, get_uci_data, get_synthetic_set
from sklearn.manifold import TSNE as tsne
from sklearn import svm
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    noise_percent = 0.05
    if len(sys.argv)==1 or sys.argv[1]=="--help":
        print("tsar_test --help:\t\t\tprint this menu")
        print("tsar_test set_name:\t\t\treview 5 percent of a dataset with default feature extractor")
        print("tsar_test set_name S noise_level:\treview the specified percentage using a supervised feature extractor")
        print("tsar_test set_name U noise_level:\treview the specified percentage using an unsupervised feature extractor")
        exit()
    elif len(sys.argv)==2:
        set_name = sys.argv[1]
        extractor = 'S'
    elif len(sys.argv)==4:
        set_name = sys.argv[1]
        extractor = sys.argv[2]
        noise_percent = float(sys.argv[3])
        if noise_percent>1:
            noise_percent = noise_percent/100
    else:
        print("Unrecognized Arguments")
        exit()

    if set_name == "UniMiB":
        X, y, labels = get_unimib_data("adl")
        print("Unimib: ", len(X))
    elif set_name == "UCI":
        X, y, labels = get_uci_data()
    elif set_name == "synthetic":
        X, y, labels = get_synthetic_set(1)
    else:
        print("Dataset must be UniMiB or UCI")
        exit()

    X, y = preprocess_raw_data_and_labels(X, y)

    print(y)

    if extractor == 'S':
        if os.path.isfile("data/"+set_name+"_sup_feat.csv"):
            features = np.genfromtxt("data/"+set_name+"_sup_feat.csv", delimiter=',')
        else:
            features = get_supervised_features(X, y, True, "data/"+set_name+"_sup_feat.csv")
    elif extractor == 'U':
        if os.path.isfile("data/"+set_name+"_unsup_feat.csv"):
            features = np.genfromtxt("data/"+set_name+"_unsup_feat.csv", delimiter=',')
        else:
            features = get_unsupervised_features(X, True, "data/"+set_name+"_unsup_feat.csv")
    else:
        print("Feature extractor must be S or U")
        exit()

    print("\n################ Reviewing {0}% of {1} ################".format(noise_percent*100, set_name))
    if extractor == 'S':
        print("################ Supervised ################\n")
    else:
        print("################ Unsupervised ################\n")

    print("{} instances in full dataset.".format(len(X)))
    if not os.path.isfile("test/{}_identified_bad.csv".format(set_name)):
        #features = np.genfromtxt("data/"+set_name+"_sup_feat.csv", delimiter=',')
        indices, pLabels = check_dataset(X, y, featureType='p', features=features)
        number_of_bad_guys = int(noise_percent*X.shape[0])
        print("Generating graphs for {} worst instances".format(number_of_bad_guys))

        bad_guys = indices[:number_of_bad_guys]
        vis = tsne(n_components=2, n_jobs=8).fit_transform(features)

        log = open("test/test_log.txt", 'w+')

        for b in bad_guys:
            file_name = "test/visualizations/{}_{}_instance_{}.pdf".format(set_name, extractor, b)
            print_graph_for_instance_two_class(X, y, labels, b, feat=features, vis=vis, show=False, saveToFile=True, filename=file_name)
            log.write("Instance index: {}\n".format(b))
            log.write("Assigned label: {}\n".format(y[b]))
            log.write("Predicted label: {}\n".format(np.argmax(pLabels[b])))
            log.write("<><><><><><><><><><><><><><>\n\n")
            log.flush()

        log.close()

        a = 0
        identified_bad_indexes = np.array([])
        while a != -1:
            a = int(input("Enter a bad index or -1 to stop: "))
            if a != -1:
                identified_bad_indexes = np.append(identified_bad_indexes, a)

        print(identified_bad_indexes)
        np.savetxt("test/{}_identified_bad.csv".format(set_name), identified_bad_indexes, delimiter=",", fmt="%d")
    else:
        identified_bad_indexes = np.genfromtxt("test/{}_identified_bad.csv".format(set_name), delimiter=',', dtype='int32')

    print("Shape of X = ", features.shape)

    clean_X = np.delete(features, identified_bad_indexes, 0)
    clean_y = np.delete(y, identified_bad_indexes, 0)
    print("Cleaned feature set now has {} instances".format(len(clean_X)))
    print("Cleaned label set now has {} instances".format(len(clean_y)))
    print("Uncleaned feature set still has {} instances".format(len(X)))

    noisy_classifier = svm.LinearSVC(verbose=1, dual=False)
    clean_classifier = svm.LinearSVC(verbose=1, dual=False)
    noisy_X_train, noisy_X_test, noisy_y_train, noisy_y_test = train_test_split(features, y, test_size=0.2, shuffle=True)
    clean_X_train, clean_X_test,clean_y_train, clean_y_test = train_test_split(clean_X, clean_y, test_size=0.2, shuffle=True)

    print("Y argmax = ", np.argmax(noisy_y_train, axis=-1))

    noisy_classifier.fit(noisy_X_train, np.argmax(noisy_y_train, axis=-1))
    clean_classifier.fit(clean_X_train, np.argmax(clean_y_train, axis=-1))
