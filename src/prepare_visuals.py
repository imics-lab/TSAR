#Author: Gentry Atkinson
#Organization: Texas University
#Data: 04 November, 2020
#Create visualizations of 6 instances each from 2 UniMiB sets and the UCI set
#  using 3 methods of feature extraction

from tsar import get_supervised_features, get_unsupervised_features, print_graph_for_instance
from utils.ts_feature_toolkit import get_features_for_set
from import_datasets import get_unimib_data, get_uci_data

if __name__ == "__main__":
    print("Preparing 45 instance visualizations...")
    for s in ["adl", "two_classes"]:
        X, y, labels = get_unimib_data(s)
