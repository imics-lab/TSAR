#Author: Gentry Atkinson
#Organization: Texas University
#Data: 30 December, 2020
#Expand the tsar test to additional models using the features sets from "tsar test"

import datetime
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    unimib_feats_noisy = np.genfromtxt("test/visualizations_for_unimib_sup/UniMiB_sup_feat.csv", delimiter=',')
    unimib_y_noisy = np.genfromtxt("test/visualizations_for_unimib_sup/UniMiB_shuffeled_labels.csv", delimiter=',', dtype='int32')
    #unimib_y_noisy = np.argmax(unimib_y_noisy, axis=-1)
    unimib_bad_guys =  np.genfromtxt("test/visualizations_for_unimib_sup/UniMiB_identified_bad.csv", delimiter=',')

    unimib_feats_clean = np.delete(unimib_feats_noisy, unimib_bad_guys, 0)
    unimib_y_clean = np.delete(unimib_y_noisy, unimib_bad_guys, 0)

    print("Unimib noisy features shape: ", unimib_feats_noisy.shape)
    print("Unimib noisy y shape: ", unimib_y_noisy.shape)
    print("Unimib clean features shape: ", unimib_feats_clean.shape)
    print("Unimib clean y shape: ", unimib_y_clean.shape)

    num_labels = unimib_y_noisy.shape[1]

    log = open("data_cleaning_experiments_results.txt", 'a+')
    log.write("{}\n".format(datetime.datetime.now()))
    log.write("------- UniMiB NN Experiments-----------\n\n")

    noisy_X_train, noisy_X_test, noisy_y_train, noisy_y_test = train_test_split(unimib_feats_noisy, unimib_y_noisy, test_size=0.2, shuffle=False)
    clean_X_train, clean_X_test, clean_y_train, clean_y_test = train_test_split(unimib_feats_clean, unimib_y_clean, test_size=0.2, shuffle=False)

    log.write("3-layer NN results: \n")
    noisy_model = Sequential([
        Input(shape=noisy_X_train[0].shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    clean_model = Sequential([
        Input(shape=clean_X_train[0].shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    noisy_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
    noisy_model.summary()
    clean_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
    clean_model.summary()

    noisy_model.fit(noisy_X_train, noisy_y_train, epochs=20, verbose=1)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train, epochs=20, verbose=1)
    clean_y_pred = clean_model.predict(clean_X_test)

    noisy_acc = accuracy_score(np.argmax(noisy_y_test, axis=-1), np.argmax(noisy_y_pred, axis=-1))
    clean_acc = accuracy_score(np.argmax(clean_y_test, axis=-1), np.argmax(clean_y_pred, axis=-1))
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))

    log.write("6-layer NN results: \n")
    noisy_model = Sequential([
        Input(shape=noisy_X_train[0].shape),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    clean_model = Sequential([
        Input(shape=clean_X_train[0].shape),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    noisy_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
    noisy_model.summary()
    clean_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
    clean_model.summary()

    noisy_model.fit(noisy_X_train, noisy_y_train, epochs=20, verbose=1)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train, epochs=20, verbose=1)
    clean_y_pred = clean_model.predict(clean_X_test)

    noisy_acc = accuracy_score(np.argmax(noisy_y_test, axis=-1), np.argmax(noisy_y_pred, axis=-1))
    clean_acc = accuracy_score(np.argmax(clean_y_test, axis=-1), np.argmax(clean_y_pred, axis=-1))
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))

    log.flush()

    uci_feats_noisy = np.genfromtxt("test/visualizations_for_uci_sup/UCI_sup_feat.csv", delimiter=',')
    uci_y_noisy = np.genfromtxt("test/visualizations_for_uci_sup/UCI_shuffeled_labels.csv", delimiter=',', dtype='int32')
    #uci_y_noisy = np.argmax(uci_y_noisy, axis=-1)
    uci_bad_guys =  np.genfromtxt("test/visualizations_for_uci_sup/UCI_identified_bad.csv", delimiter=',')

    uci_feats_clean = np.delete(uci_feats_noisy, uci_bad_guys, 0)
    uci_y_clean = np.delete(uci_y_noisy, uci_bad_guys, 0)

    print("UCI noisy features shape: ", uci_feats_noisy.shape)
    print("UCI noisy y shape: ", uci_y_noisy.shape)
    print("UCI clean features shape: ", uci_feats_clean.shape)
    print("UCI clean y shape: ", uci_y_clean.shape)

    num_labels = uci_y_noisy.shape[1]

    log.write("------- UCI NN Experiments-----------\n\n")

    noisy_X_train, noisy_X_test, noisy_y_train, noisy_y_test = train_test_split(uci_feats_noisy, uci_y_noisy, test_size=0.2, shuffle=False)
    clean_X_train, clean_X_test, clean_y_train, clean_y_test = train_test_split(uci_feats_clean, uci_y_clean, test_size=0.2, shuffle=False)

    log.write("3-layer NN results: \n")
    noisy_model = Sequential([
        Input(shape=noisy_X_train[0].shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    clean_model = Sequential([
        Input(shape=clean_X_train[0].shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    noisy_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
    noisy_model.summary()
    clean_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
    clean_model.summary()

    noisy_model.fit(noisy_X_train, noisy_y_train, epochs=20, verbose=1)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train, epochs=20, verbose=1)
    clean_y_pred = clean_model.predict(clean_X_test)

    noisy_acc = accuracy_score(np.argmax(noisy_y_test, axis=-1), np.argmax(noisy_y_pred, axis=-1))
    clean_acc = accuracy_score(np.argmax(clean_y_test, axis=-1), np.argmax(clean_y_pred, axis=-1))
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))

    log.write("6-layer NN results: \n")
    noisy_model = Sequential([
        Input(shape=noisy_X_train[0].shape),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    clean_model = Sequential([
        Input(shape=clean_X_train[0].shape),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    noisy_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
    noisy_model.summary()
    clean_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
    clean_model.summary()

    noisy_model.fit(noisy_X_train, noisy_y_train, epochs=20, verbose=1)
    noisy_y_pred = noisy_model.predict(noisy_X_test)
    clean_model.fit(clean_X_train, clean_y_train, epochs=20, verbose=1)
    clean_y_pred = clean_model.predict(clean_X_test)

    noisy_acc = accuracy_score(np.argmax(noisy_y_test, axis=-1), np.argmax(noisy_y_pred, axis=-1))
    clean_acc = accuracy_score(np.argmax(clean_y_test, axis=-1), np.argmax(clean_y_pred, axis=-1))
    log.write("Noisy: {}\n".format(noisy_acc))
    log.write("Clean: {}\n".format(clean_acc))
    log.write("===\n\n".format(clean_acc))

    log.flush()
    log.close()
