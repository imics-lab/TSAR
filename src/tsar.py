#Author: Gentry Atkinson
#Organization: Texas University
#Data: 30 October, 2020
#Identify and review a portion of a dataset most likely to be mislabeled

import numpy as np
from utils.build_AE import get_trained_AE
from utils.build_sup_extractor import get_trained_sfe
from utils.build_simple_dnn import get_trained_dnn
from utils.color_pal import color_pallette_big as pal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt


#Source from Labelfix repository
# def _get_indices(pred, y):
#     """
#     For internal use: Given a prediction of a model and labels y, return sorted indices of likely mislabeled instances
#     :param pred:        np array, Predictions made by a classifier
#     :param y:           np array, Labels according to the data set
#     :return:            np array, List of indices sorted by how likely they are wrong according to the classifier
#     """
#     assert pred.shape[0] == y.shape[0], "Pred {} =! y shape {}".format(pred.shape[0], y.shape[0])
#     y_squeezed = y.squeeze()
#     if y_squeezed.ndim == 2:
#         dots = [np.dot(pred[i], y_squeezed[i]) for i in range(len(pred))]  # one-hot y
#     elif y_squeezed.ndim == 1:
#         print("y squeezed of i: ", y_squeezed[0])
#         dots = [pred[i, y_squeezed[i]] for i in range(len(pred))]  # numeric y
#     else:
#         raise ValueError("Wrong dimension of y!")
#     indices = np.argsort(dots)
#     return indices

def sort_indices(pred_y, true_y):
    closeness = [np.dot(pred_y[i], true_y[i]) for i in range(pred_y.shape[0])]
    indices = np.argsort(closeness)
    return indices

def get_unsupervised_features(X, saveToFile=False, filename="unsup_features.csv"):
    ae = get_trained_AE(X)
    feat = ae.predict(X)
    if saveToFile:
        np.savetxt(filename, feat, delimiter=",")
    return feat

def get_supervised_features(X, y, saveToFile=False, filename="sup_features.csv"):
    sfe = get_trained_sfe(X, y)
    feat = sfe.predict(X)
    if saveToFile:
        np.savetxt(filename, feat, delimiter=",")
    return feat

def preprocess_raw_data_and_labels(X, y):
    print("Applying pre-processing")
    if len(X) != len(y):
        print("Data and labels must have same number of instances")
        return
    m = np.max(X)
    X = (X/m)
    if y.ndim == 1:
        y = to_categorical(y)

    return X,y

def count_class_imbalance(y):
    if y.ndim == 1:
        y = to_categorical(y)

    counts = np.zeros(len(y[0]))
    for i in range(len(y[0])):
        print(np.sum(y[:,i]))
        counts[i] = np.sum(y[:,i])

    #print("Most prevalent class: {}".format(np.max(counts)))
    #print("Least prevalent class: {}".format(np.min(counts)))
    return np.max(counts)/np.min(counts)

def check_dataset(X, y, featureType='u'):

    print("Checking dataset for suspicious labels")
    if featureType=='u':
        model = get_trained_AE(X)
        feats = model.predict(X)
        feats, y = preprocess_raw_data_and_labels(X, y)
    elif featureType=='s':
        model = get_trained_sfe(X,y)
        feats = model.predict(X)
        feats, y = preprocess_raw_data_and_labels(X,y)
    elif featureType=='o':
        feats, y = preprocess_raw_data_and_labels(X,y)
    else:
        print("featureType must be u, s, or o")
        return

    c = get_trained_dnn(feats, y)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    c.fit(X, y, epochs=50, verbose=1, callbacks=[es], validation_split=0.1, batch_size=10)

    y_pred = c.predict(feats)

    return sort_indices(y_pred, y)

def _on_button_clicked_correct(b):
    print("correct")

def _on_button_clicked_mislabeled(b):
    print("mislabeled")

def review_dataset(X, y, featureType='u', percentReview=0.2):
    print("Starting manual review process")
    print("X: ", X)
    print("y: ", y)
    indices = check_dataset(X, y, featureType=featureType)
    i = int(len(X) * percentReview)
    sus_set = indices[0:i]
    good_set = indices[i:len(X)]

    buttonC = widgets.Button(description="Correct")
    buttonM = widgets.Button(description="Mislabeled")
    output = widgets.Output()

    buttonC.on_click(_on_button_clicked_correct)
    buttonM.on_click(_on_button_clicked_mislabeled)

    display(buttonC, buttonM, output)

    y_flat = np.argmax(y, axis=1)

    print("sus set: ", sus_set)
    print("y flat: ", y_flat)

    point = sus_set[0]
    sus_label = y_flat[point]

    print("Point: ", point)
    print("sus label: ", sus_label)

    figure = plt.figure(1, figsize=(15,10))
    ax1 = plt.subplot2grid((3,4), (0,0), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid((3,4), (0, 3))
    ax3 = plt.subplot2grid((3,4), (1, 3))
    ax4 = plt.subplot2grid((3,4), (2, 3))

    NUM_LABELS = y.shape[1]

    for i in range(NUM_LABELS):
        x = np.where(y==i)
        ax1.scatter(X[x, 0], X[x, 1], s=6, c=pal[i], marker=".")
    ax1.scatter(X[point, 0], X[point, 1], s=200, c=pal[sus_label], marker="X")
    ax1.set_title("tSNE of extracted features")
    ax1.legend()
    ax1.axis('off')


    # ax2.plot(range(0, len(X[random_point, 0, :])), X[random_point, 0, :], c=pal[sus_label])
    # ax2.plot(range(0, len(X[random_point, 1, :])), X[random_point, 1, :], c=pal[sus_label])
    # ax2.plot(range(0, len(X[random_point, 2, :])), X[random_point, 2, :], c=pal[sus_label])
    # ax2.set_title("Suspicious point with label: " + str([sus_label]))

    # ax3.plot(range(0, NUM_SAMPLES), X[nn, 0, :], c=pal[y[nn]])
    # ax3.plot(range(0, NUM_SAMPLES), X[nn, 1, :], c=pal[y[nn]])
    # ax3.plot(range(0, NUM_SAMPLES), X[nn, 2, :], c=pal[y[nn]])
    # ax3.set_title("Nearest neighbor has label: " + str(labels[y[nn]]))

    # ax4.plot(range(0, NUM_SAMPLES), rep_signal[0,:], c=pal[sus_label])
    # ax4.plot(range(0, NUM_SAMPLES), rep_signal[1,:], c=pal[sus_label])
    # ax4.plot(range(0, NUM_SAMPLES), rep_signal[2,:], c=pal[sus_label])
    # ax4.set_title("Another point with label: " + str(labels[sus_label]))
    #plt.savefig("/content/drive/My Drive/LabelReview/set_"+s+"_ae_example_figure.pdf")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    y = np.array([0, 1, 2, 0, 1], dtype='int')
    print("Class imbalance: ", count_class_imbalance(y))
    X = [
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1]
    ]

    X, y = preprocess_raw_data_and_labels(X, y)
    print(X)
    print(y)

    review_dataset(X, y, featureType='u')
    #print("worst index: ", indices[0])
