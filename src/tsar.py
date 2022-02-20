#Author: Gentry Atkinson
#Organization: Texas University
#Data: 30 October, 2020
#Identify and review a portion of a dataset most likely to be mislabeled

import numpy as np
import random as rand
from utils.build_AE import get_trained_AE
from utils.build_sup_extractor import get_trained_sfe
from utils.build_simple_dnn import get_trained_dnn
from utils.color_pal import color_pallette_big, color_pallette_small
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE as tsne
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity


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

#Sort indices
#Parameters:
#   label vectors predicted by check_dataset
#   label vectors from dataset
#Returns: the sorted indexes by magnitude of dot product between two parameters
def sort_indices(pred_y, true_y):
    closeness = [np.dot(pred_y[i], true_y[i]) for i in range(pred_y.shape[0])]
    indices = np.argsort(closeness)
    return indices

#Get Unsupervised features
#Parameters:
#   2 or 3d array of instances
#Returns: a 2d array of features learned by a convolutional autoencoder
#See utils.build_AE
def get_unsupervised_features(X, saveToFile=False, filename="unsup_features.csv"):
    ae = get_trained_AE(X, withVisual=False)
    feat = ae.predict(X)
    if saveToFile:
        np.savetxt(filename, feat, delimiter=",")
    return feat

#Get Supervised Features
#Parameters
#   2 or 3d array of instances
#   label arrays from dataset
#Returns: a 2d array of features learned by a truncated CNN
#See utils.build_sup_extractor
def get_supervised_features(X, y, saveToFile=False, filename="sup_features.csv"):
    sfe = get_trained_sfe(X, y)
    feat = sfe.predict(X)
    if saveToFile:
        np.savetxt(filename, feat, delimiter=",")
    return feat

#Get NearestNeighbors for Dataset
#Parameters:
#   a 2 or 3d array of instances
#Returns: a list of indexes of the 1 nearest neighbor for each instance
def get_NN_for_dataset(X, saveToFile=False, filename="nn.csv"):
    #nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric=cos_dis).fit(X)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    if saveToFile:
        np.savetxt(filename, feat, delimiter=",")
    return indices

#Preprocess Raw Data and Labels
#Parameters:
#   a 2 or 3d array of instances
#   labels from dataset, either one-hot or class
#Returns: normalized instances and label vectors, both shuffled
def preprocess_raw_data_and_labels(X, y):
    print("Applying pre-processing")
    if len(X) != len(y):
        print("Data and labels must have same number of instances")
        return
    m = np.max(X)
    X = (X/m)
    if y.ndim == 1:
        y = to_categorical(y)

    X,y = shuffle(X,y)

    return X,y

#Returns the inverse of cosine simalarity
def cos_dis(x,y):
    X = [x]
    Y = [y]
    d = cosine_similarity(X, Y)
    return 1 - d

#Print Graph For Instance
#Parameters:
#   2 or 3d array of instances
#   label vectors from dataset
#   the index of one instance from X
#   a feature set learned from X (optional)
#   a tSNE reduction of X(optional)
#Returns: nothing
#Saves one pdf visualization of instance in X with all classes represented
def print_graph_for_instance(X, y, labels, instance, feat=None, neighbors=None, vis=None, show=False, saveToFile=False, filename="graph.pdf", mislabeled=False):
    #X, y = preprocess_raw_data_and_labels(X, y)
    if y.ndim>=2:
        #print("Converting labels from One-Hot")
        y = np.argmax(y, axis=-1)
    if feat is None:
        feat = X
    if vis is None:
        if X.ndim > 2:
            print("This raw data cannot be visualized with tSNE")
            return
        vis = tsne(n_components=2, n_jobs=8).fit_transform(feat)
    if neighbors is None:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feat)
        distances, neighbors = nbrs.kneighbors(feat)

    if np.max(y) > 4:
        pal = color_pallette_big
    else:
        pal = color_pallette_small

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    nn = neighbors[instance, 1]
    sus_label = y[instance]
    NUM_LABELS = int(np.max(y)+1)
    if mislabeled:
        sus_label = (sus_label + rand.randint(1,NUM_LABELS)) % NUM_LABELS

    rep_signal = X[np.where(y==sus_label)][0,:,:]

    figure = plt.figure(1, figsize=(15,10))
    ax1 = plt.subplot2grid((3,4), (0,0), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid((3,4), (0, 3))
    ax3 = plt.subplot2grid((3,4), (1, 3))
    ax4 = plt.subplot2grid((3,4), (2, 3))

    NUM_SAMPLES = len(X[0,0,:])

    for i in range(NUM_LABELS):
        if (not i==sus_label) and (not i==y[nn]):
            continue
        x = np.where(y==i)
        ax1.scatter(vis[x, 0], vis[x, 1], s=6, c=pal[i], marker=".", label=labels[i])
    ax1.scatter(vis[instance, 0], vis[instance, 1], s=250, c=pal[sus_label], marker="X", label="Suspicious Point")
    ax1.set_title("tSNE of all features", fontsize=36)
    ax1.legend(prop={'size': 18})
    ax1.axis('off')

    for i in range(X.shape[1]):
        ax2.plot(range(0, NUM_SAMPLES), X[instance, i, :], c=pal[sus_label])
    ax2.set_title("Suspicious point with label: " + str(labels[sus_label]), fontsize=18)

    for i in range(X.shape[1]):
        ax3.plot(range(0, NUM_SAMPLES), X[nn, i, :], c=pal[y[nn]])
    ax3.set_title("Nearest neighbor has label: " + str(labels[y[nn]]), fontsize=18)

    for i in range(X.shape[1]):
        ax4.plot(range(0, NUM_SAMPLES), rep_signal[i,:], c=pal[sus_label])
    ax4.set_title("Another point with label: " + str(labels[sus_label]), fontsize=18)

    plt.tight_layout()

    if saveToFile:
        plt.savefig(filename)

    if show:
        plt.show()

#Print Graph For Instance Two Class
#Parameters:
#   2 or 3d array of instances
#   label vectors from dataset
#   the index of one instance from X
#   a feature set learned from X (optional)
#   a tSNE reduction of X(optional)
#Returns: nothing
#Saves one pdf visualization of instance in X with only 2 "nearest" classes represented
def print_graph_for_instance_two_class(X, y, labels, instance, feat=None, vis=None, show=False, saveToFile=False, filename="graph.pdf", mislabeled=False):
    if y.ndim>=2:
        #print("Converting labels from One-Hot")
        y = np.argmax(y, axis=-1)
    NUM_LABELS = int(np.max(y)+1)

    if feat is None:
        feat = X

    if mislabeled:
        same_label = (y[instance] + rand.randint(1,NUM_LABELS)) % NUM_LABELS
    else:
        same_label = y[instance]

    feat_same = feat[np.where(y==same_label)]
    feat_diff = feat[np.where(y!=same_label)]
    feat_diff = np.append(feat_diff, [feat[instance]], axis=0)

    #y_same = y[np.where(y==same_label)]
    y_diff = y[np.where(y!=same_label)]
    y_diff = np.append(y_diff, [y[instance]], axis=0)

    X_same = X[np.where(y==same_label)]
    X_diff = X[np.where(y!=same_label)]
    X_diff = np.append(X_diff, [X[instance]], axis=0)

    if(len(feat_same)==0):
        print("Length zero same feature set encountered")
        return
    elif(len(feat_diff)==0):
        print("Length zero diff feature set encountered")
        return

    #print(feat_same)
    #print(feat_diff)


    pal = color_pallette_small

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    NUM_SAMPLES = X.shape[2]

    #nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric=cos_dis).fit(feat_same)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feat_same)
    distances, nn_same = nbrs.kneighbors(feat_same)
    #nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric=cos_dis).fit(feat_diff)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feat_diff)
    distances, nn_diff = nbrs.kneighbors(feat_diff)

    #u, c = np.unique(y[:instance], return_counts=True)
    same_instance = len(np.where(y[:instance]==same_label))
    # diff_instance = c[diff_label]
    diff_label = y_diff[nn_diff[-1,1]]

    #print("Same label: ", same_label)
    #print("Diff label: ", diff_label)

    if vis is None:
        if feat.ndim > 2:
            print("This raw data cannot be visualized with tSNE")
            return
        vis = tsne(n_components=2, n_jobs=8).fit_transform(feat)

    figure = plt.figure(1, figsize=(15,10))
    ax1 = plt.subplot2grid((3,4), (0,0), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid((3,4), (0, 3))
    ax3 = plt.subplot2grid((3,4), (1, 3))
    ax4 = plt.subplot2grid((3,4), (2, 3))

    for i in range(X.shape[1]):
        ax2.plot(range(0, NUM_SAMPLES), X[instance, i, :], c=pal[0])
    ax2.set_title("Suspicious point has label: " + str(labels[same_label]), fontsize=18)

    for i in range(X.shape[1]):
        ax3.plot(range(0, NUM_SAMPLES), X_same[nn_same[same_instance, 1], i, :], c=pal[0])
    ax3.set_title("Nearest neighbor with label: " + str(labels[same_label]), fontsize=18)

    for i in range(X.shape[1]):
        ax4.plot(range(0, NUM_SAMPLES), X_diff[nn_diff[-1, 1], i, :], c=pal[1])
    ax4.set_title("Nearest neighbor with label: " + str(labels[diff_label]), fontsize=18)

    if(len(vis[0])==2):
        #print("2d tSNE")
        x = np.where(y==same_label)
        ax1.scatter(vis[x, 0], vis[x, 1], s=6, c=pal[0], marker="^")
        x = np.where(y==diff_label)

        #"zoom in" the plot
        #plt.xlim(vis[instance,0]-0.5, vis[instance,0]+0.5)
        #plt.xlim(vis[instance,1]-0.5,vis[instance,1]+0.5)
        ax1.set_xlim(vis[instance,0]-100, vis[instance,0]+100)
        ax1.set_ylim(vis[instance,1]-100, vis[instance,1]+100)

        ax1.scatter(vis[x, 0], vis[x, 1], s=10, c=pal[1], marker=".")
        ax1.scatter(vis[instance, 0], vis[instance, 1], s=300, c='black', marker="X")
        ax1.scatter(vis[instance, 0], vis[instance, 1], s=200, c=pal[0], marker="X", label="Suspicious Point")
        ax1.set_title("tSNE of all features", fontsize=36)
        patch_1 = mpatches.Patch(color=pal[0], label=labels[same_label])
        patch_2 = mpatches.Patch(color=pal[1], label=labels[diff_label])
        ax1.legend(prop={'size': 18}, handles=[patch_1, patch_2])
        ax1.axis('off')
    elif(len(vis[0])==3):
        #print("3d tSNE")
        x = np.where(y==same_label)
        ax1.scatter(vis[x, 0], vis[x, 1], vis[x, 2], c=pal[0], marker="^")
        x = np.where(y==diff_label)
        ax1.scatter(vis[x, 0], vis[x, 1], vis[x, 2], c=pal[1], marker=".")
        ax1.scatter(vis[instance, 0], vis[instance, 1], vis[instance, 2], c='black', marker="X")
        ax1.scatter(vis[instance, 0], vis[instance, 1], vis[instance, 2], c=pal[0], marker="X", label="Suspicious Point")
        ax1.set_title("tSNE of all features", fontsize=36)
        patch_1 = mpatches.Patch(color=pal[0], label=labels[same_label])
        patch_2 = mpatches.Patch(color=pal[1], label=labels[diff_label])
        ax1.legend(prop={'size': 18}, handles=[patch_1, patch_2])
        ax1.axis('off')
    else:
        print("tSNE points have an unusual number of dimensions")
        return

    plt.tight_layout()

    if saveToFile:
        plt.savefig(filename)

    if show:
        plt.show()

    return

#Add Noise to Labels
#Parameters:
#   a set of labels
#   a percent of labels to alter as a whole number
#Returns: a classwise (not one-hot) set of altered labels, a set of indexes representing the
#   altered labels
#Labels are altered in a uniformally at random fashion
def add_noise_to_labels(y, percentNoise=5, saveToFile=False, filename="noisy_labels.csv"):
    bad = np.array([], dtype='int')
    NUM_LABELS = int(np.max(y)+1)
    #print(len(y), " labels are getting noisy")
    #print ("starting off with these bad indices: ", bad)
    if y.ndim>=2:
        y = np.argmax(y, axis=-1)
    for i in range(len(y)):
        chance = rand.randint(0, 100)
        if chance < percentNoise:
            y[i] = (y[i] + rand.randint(1,NUM_LABELS)) % NUM_LABELS
            bad = np.append(bad, i)
    #print(bad)
    print(len(bad), " bad idices added")
    return y, bad

#Count Class Imbalance
def count_class_imbalance(y):
    if y.ndim == 1:
        y = to_categorical(y)

    counts = np.zeros(len(y[0]))
    for i in range(len(y[0])):
        #print(np.sum(y[:,i]))
        counts[i] = np.sum(y[:,i])
    return np.max(counts)/np.min(counts)

#Check Dataset
#Parameters:
#   a 2 or 3d array of instances
#   list of one-hot labels
#   the style of feature to use:
#       'u'nsupervised
#       's'supervised
#       'o'riginal (no feature extraction)
#       'p're-processed
def check_dataset(X, y, featureType='u', features=None):
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
    elif featureType=='p':
        feats = features
    else:
        print("featureType must be u, s, p, or o")
        return

    c = get_trained_dnn(feats, y)
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    #c.fit(feats, y, epochs=50, verbose=1, callbacks=[es], validation_split=0.1, batch_size=10)

    y_pred = c.predict(feats)
    indices = sort_indices(y_pred, y)
    return indices, y_pred[indices]


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

    indices = check_dataset(X, y, featureType='u')
    print("worst index: ", indices[0])
    labels = ['type one', 'type two', 'type three']
    print("Plotting index 0:")
    print_graph_for_instance_two_class(X, y, labels, instance=0, show=True)
