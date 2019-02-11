import cv2
import numpy as np
import matplotlib.pyplot as pt
    
import operator
import os
import sys
import glob

from random import shuffle

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.utils.data as data_utils


from multiprocessing import Process, freeze_support


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import copy


def Label_Encode(block_labels_training):
    values = array(block_labels_training)
    label_encoder = LabelEncoder()
    #print(values)
    block_labels_training_encoded = label_encoder.fit_transform(values)
    #print(block_labels_training_encoded)
    block_labels_training_tensor_encoded = torch.LongTensor(block_labels_training_encoded)
    return block_labels_training_tensor_encoded