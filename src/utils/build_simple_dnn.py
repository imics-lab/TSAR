#Author: Gentry Atkinson
#Organization: Texas University
#Data: 31 October, 2020
#Train and return a deep neural network

#from tensorflow import keras
import numpy as np
import random as rand
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense
from tensorflow.keras.layers import Input, Conv1DTranspose, Lambda, Reshape, BatchNormalization
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
