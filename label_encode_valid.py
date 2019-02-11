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

def Label_Encode_Valid(block_labels_valid):
    block_labels_valid_encode = []
    values = array(block_labels_valid)
    #print(values)
    for i in range(0, len(values)):
        if(values[i] == 'recaptured'):
            block_labels_valid_encode.append(1)
        else:
            block_labels_valid_encode.append(0)
    #print(block_labels_valid_encode)
    block_labels_tensor_valid_encoded = torch.LongTensor(block_labels_valid_encode)
    return block_labels_tensor_valid_encoded