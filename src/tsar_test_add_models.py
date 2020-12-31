#Author: Gentry Atkinson
#Organization: Texas University
#Data: 30 December, 2020
#Expand the tsar test to additional models using the features sets from "tsar test"

import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Models to investigate:
#   KNN
#   C4.5
#   Naive Bayes

if __name__ == "__main__":
    unimib_feats_noisy = np.genfromtxt("test/visualizations_for_unimib_sup/UniMiB_sup_feat.csv", delimiter=',')
    unimib_y_noisy = np.genfromtxt("test/visualizations_for_unimib_sup/UniMiB_shuffeled_labels.csv", delimiter=',', dtype='int32')
    unimib_y_noisy = np.argmax(unimib_y_noisy, axis=-1)
    #print(unimib_y_noisy)
    unimib_bad_guys =  np.genfromtxt("test/visualizations_for_unimib_sup/UniMiB_identified_bad.csv", delimiter=',')

    unimib_feats_clean = np.delete(unimib_feats_noisy, unimib_bad_guys, 0)
    unimib_y_clean = np.delete(unimib_y_noisy, unimib_bad_guys, 0)
    #print(unimib_y_clean)

    log = open("data_cleaning_experiments_results.txt", 'a+')
    log.write("{}\n".format(datetime.datetime.now()))
    log.write("------- UniMiB Experiments-----------\n\n")

    noisy_X_train, noisy_X_test, noisy_y_train, noisy_y_test = train_test_split(unimib_feats_noisy, unimib_y_noisy, test_size=0.2, shuffle=False)
    clean_X_train, clean_X_test, clean_y_train, clean_y_test = train_test_split(unimib_feats_clean, unimib_y_clean, test_size=0.2, shuffle=False)

    log.write("KNN results: \n")
    noisy_model = KNeighborsClassifier(n_neighbors=3)
    clean_model = KNeighborsClassifier(n_neighbors=3)

    noisy_model.fit(noisy_X_train, noisy_y_train)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train)
    clean_y_pred = clean_model.predict(clean_X_test)
    noisy_acc = accuracy_score(noisy_y_test, noisy_y_pred)
    clean_acc = accuracy_score(clean_y_test, clean_y_pred)
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))

    log.write("C4.5 results: \n")
    noisy_model = DecisionTreeClassifier(criterion='entropy', splitter='best')
    clean_model = DecisionTreeClassifier(criterion='entropy', splitter='best')

    noisy_model.fit(noisy_X_train, noisy_y_train)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train)
    clean_y_pred = clean_model.predict(clean_X_test)
    noisy_acc = accuracy_score(noisy_y_test, noisy_y_pred)
    clean_acc = accuracy_score(clean_y_test, clean_y_pred)
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))

    log.write("Naive Bayes results: \n")
    noisy_model = GaussianNB()
    clean_model = GaussianNB()

    noisy_model.fit(noisy_X_train, noisy_y_train)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train)
    clean_y_pred = clean_model.predict(clean_X_test)
    noisy_acc = accuracy_score(noisy_y_test, noisy_y_pred)
    clean_acc = accuracy_score(clean_y_test, clean_y_pred)
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))

    uci_feats_noisy = np.genfromtxt("test/visualizations_for_uci_sup/UCI_sup_feat.csv", delimiter=',')
    uci_y_noisy = np.genfromtxt("test/visualizations_for_uci_sup/UCI_shuffeled_labels.csv", delimiter=',', dtype='int32')
    uci_y_noisy = np.argmax(uci_y_noisy, axis=-1)
    #print(unimib_y_noisy)
    uci_bad_guys =  np.genfromtxt("test/visualizations_for_uci_sup/UCI_identified_bad.csv", delimiter=',')

    uci_feats_clean = np.delete(uci_feats_noisy, uci_bad_guys, 0)
    uci_y_clean = np.delete(uci_y_noisy, uci_bad_guys, 0)
    #print(unimib_y_clean)

    log = open("data_cleaning_experiments_results.txt", 'a+')
    log.write("------- UCI Experiments-----------\n\n")

    noisy_X_train, noisy_X_test, noisy_y_train, noisy_y_test = train_test_split(uci_feats_noisy, uci_y_noisy, test_size=0.2, shuffle=False)
    clean_X_train, clean_X_test, clean_y_train, clean_y_test = train_test_split(uci_feats_clean, uci_y_clean, test_size=0.2, shuffle=False)

    log.write("KNN results: \n")
    noisy_model = KNeighborsClassifier(n_neighbors=3)
    clean_model = KNeighborsClassifier(n_neighbors=3)

    noisy_model.fit(noisy_X_train, noisy_y_train)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train)
    clean_y_pred = clean_model.predict(clean_X_test)
    noisy_acc = accuracy_score(noisy_y_test, noisy_y_pred)
    clean_acc = accuracy_score(clean_y_test, clean_y_pred)
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))

    log.write("C4.5 results: \n")
    noisy_model = DecisionTreeClassifier(criterion='entropy', splitter='best')
    clean_model = DecisionTreeClassifier(criterion='entropy', splitter='best')

    noisy_model.fit(noisy_X_train, noisy_y_train)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train)
    clean_y_pred = clean_model.predict(clean_X_test)
    noisy_acc = accuracy_score(noisy_y_test, noisy_y_pred)
    clean_acc = accuracy_score(clean_y_test, clean_y_pred)
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))

    log.write("Naive Bayes results: \n")
    noisy_model = GaussianNB()
    clean_model = GaussianNB()

    noisy_model.fit(noisy_X_train, noisy_y_train)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train)
    clean_y_pred = clean_model.predict(clean_X_test)
    noisy_acc = accuracy_score(noisy_y_test, noisy_y_pred)
    clean_acc = accuracy_score(clean_y_test, clean_y_pred)
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))
