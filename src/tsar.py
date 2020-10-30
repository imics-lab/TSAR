#Author: Gentry Atkinson
#Organization: Texas University
#Data: 30 October, 2020
#Identify and review a portion of a dataset most likely to be mislabeled

import numpy as np

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
