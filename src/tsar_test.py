#Author: Gentry Atkinson
#Organization: Texas University
#Data: 10 December, 2020
#Use tsar to test training on cleaned data

import sys
import os.path
import numpy as np
from tsar import get_supervised_features, get_unsupervised_features
from import_datasets import get_unimib_data, get_uci_data

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
    elif set_name == "UCI":
        X, y, Labels = get_uci_data()
    else:
        print("Dataset must be UniMiB or UCI")
        exit()

    print(y)

    if extractor == 'S':
        if os.path.isfile("data/"+set_name+"_sup_feat.csv"):
            features = np.genfromtxt("data/"+set_name+"_sup_feat.csv", delimiter=',')
        else:
            features = get_supervised_features(X, y, True, "data/"+set_name+"_sup_feat.csv")
    elif extractor == 'U':
        features = get_unsupervised_features(X)
    else:
        print("Feature extractor must be S or U")
        exit()

    print("\n################ Reviewing {0}% of {1} ################".format(noise_percent*100, set_name))
    if extractor == 'S':
        print("################ Supervised ################\n")
    else:
        print("################ Unsupervised ################\n")
