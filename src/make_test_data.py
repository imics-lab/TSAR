#Author: Gentry Atkinson
#Organization: Texas University
#Data: 30 October, 2020
#Identify and review a portion of a dataset most likely to be mislabeled

import numpy as np
from utils.gen_ts_data import generate_pattern_array_as_csv, generate_pattern_array_as_csv_with_indexes

if __name__ == "__main__":
    generate_pattern_array_as_csv_with_indexes(length=150, numSamples=2000, numClasses=4, percentError=3, filename='data/synthetic/synthetic_set1')
    generate_pattern_array_as_csv_with_indexes(length=150, numSamples=2000, numClasses=10, percentError=3, filename='data/synthetic/synthetic_set2')
    generate_pattern_array_as_csv_with_indexes(length=300, numSamples=10000, numClasses=4, percentError=3, filename='data/synthetic/synthetic_set3')
    generate_pattern_array_as_csv_with_indexes(length=300, numSamples=10000, numClasses=10, percentError=3, filename='data/synthetic/synthetic_set4')

    f = open("data_cleaning_experiments_results.txt", 'a')
    f.write("Creating 4 datasets with 3% label error\n")
    f.write("Set one is 1000 sample of length 500 in 2 classes\n")
    f.write("Set one is 1000 sample of length 500 in 5 classes\n")
    f.write("Set one is 5000 sample of length 1000 in 2 classes\n")
    f.write("Set one is 5000 sample of length 1000 in 5 classes\n")

    f.flush()
    f.close()
