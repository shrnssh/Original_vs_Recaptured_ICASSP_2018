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

from splitall import splitall

import pycuda.driver as cuda
cuda.init()

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')
cuda2 = torch.device('cuda:2')
cuda3 = torch.device('cuda:3')



def block_div_tensor(listOfFiles, input_tensor, block_labels, action, lof_size, labels=[], valid_label=[]):
    
    if(action =='train'):
        labels = []
        labels[:] = []
        for each in listOfFiles:
            each_list = splitall(each)
            labels.append(each_list[4])

    elif(action == 'valid'):
        labels = []
        labels[:] = []
        for each in listOfFiles:
            #print(each)
            each_list = splitall(each)
            labels.append(each_list[4])
            #print(labels)
            valid_label.append(each_list[4])               #This is done as each valid file is passed individually to the block_div algo.


    elif(action == 'test'):
        labels[:] = []
        for each in listOfFiles:
            each_list = splitall(each)
            labels.append(each_list[4])
        test_labels = copy.deepcopy(labels)
    
    #print('The training labels are ', labels)    

    for i in range(0, lof_size):
        if(action == 'train'):
            print(i)
        img = cv2.imread(listOfFiles[i])
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray = img[:, :, 2]    
        #print('The dimensions of  image ', i, 'are ', gray.shape[0], gray.shape[1])    

        #BLOCK DIVISION
        blocks = ([np.array(gray[m:m+64, j:j+64]) for j in range(0,gray.shape[1], 64) for m in range(0, gray.shape[0],64)])        
        #print(np.size(blocks, 0))
    
        blocks_new = []

        #SELECTION OF ONLY 64x64 blocks
        for k in range(0, len(blocks)):
            if(blocks[k].shape[0]==64 and blocks[k].shape[1]==64):
                blocks_new.append(blocks[k])
    
        #print('The number of blocks in image ', i+1, ' is', len(blocks_new))
        #print(i)


        #COMPUTING SPATIAL VARIANCES OF EACH BLOCK
        vars = []

        for each in blocks_new:
            vars.append(np.var(each))
    
        block_dict = { k : vars[k] for k in range(0, len(vars) ) }
        sorted_x = sorted(block_dict.items(), key=operator.itemgetter(1), reverse = True)

        #print(type(sorted_x))
        input_blocks = []
        input_blocks_index = []

        #SELECTION OF THE TOP 300 BLOCKS IN AN IMAGE

        #key_list = sorted_x.keys()
        if(len(sorted_x)<300):
            top_blocks = len(sorted_x)
            for l in range(0, top_blocks):
                input_blocks_index.append(sorted_x[l][0])
        elif(len(sorted_x)>=300):
            for l in range(0, 300):
                input_blocks_index.append(sorted_x[l][0])

        for each in input_blocks_index:
            input_blocks.append(blocks_new[each])            



        #print('The number of blocks in image ', i+1, 'is', len(input_blocks))
    


        #Pre-processing where each block is passed through 16 filters. The output is then made into a 16x64x64 tensor

        for j in range(0, len(input_blocks)):
            test = input_blocks[j]
            #Inverse log of base 10
            
            op_1 = cv2.GaussianBlur(test, (3, 3), 3.16)
            op_2 = cv2.GaussianBlur(test, (3, 3), 3.80)
            op_3 = cv2.GaussianBlur(test, (3, 3), 4.786)
            op_4 = cv2.GaussianBlur(test, (3, 3), 6.309)
            op_5 = cv2.GaussianBlur(test, (3, 3), 8.709)
            op_6 = cv2.GaussianBlur(test, (3, 3), 12.589)
            op_7 = cv2.GaussianBlur(test, (3, 3), 19.05)
            op_8 = cv2.GaussianBlur(test, (3, 3), 31.62)

            op_9 = cv2.GaussianBlur(test, (5, 5), 3.16)
            op_10 = cv2.GaussianBlur(test, (5, 5), 3.80)
            op_11 = cv2.GaussianBlur(test, (5, 5), 4.786)
            op_12 = cv2.GaussianBlur(test, (5, 5), 6.309)
            op_13 = cv2.GaussianBlur(test, (5, 5), 8.709)
            op_14 = cv2.GaussianBlur(test, (5, 5), 12.589)
            op_15 = cv2.GaussianBlur(test, (5, 5), 19.05)
            op_16 = cv2.GaussianBlur(test, (5, 5), 31.62)
            

            #Inverse log of base e
            '''
            op_1 = cv2.GaussianBlur(test, (3, 3), 1.648)
            op_2 = cv2.GaussianBlur(test, (3, 3), 1.786)
            op_3 = cv2.GaussianBlur(test, (3, 3), 1.973)
            op_4 = cv2.GaussianBlur(test, (3, 3), 2.225)
            op_5 = cv2.GaussianBlur(test, (3, 3), 2.559)
            op_6 = cv2.GaussianBlur(test, (3, 3), 3.004)
            op_7 = cv2.GaussianBlur(test, (3, 3), 3.596)
            op_8 = cv2.GaussianBlur(test, (3, 3), 4.481)

            op_9 = cv2.GaussianBlur(test, (5, 5), 1.648)
            op_10 = cv2.GaussianBlur(test, (5, 5), 1.786)
            op_11 = cv2.GaussianBlur(test, (5, 5), 1.973)
            op_12 = cv2.GaussianBlur(test, (5, 5), 2.225)
            op_13 = cv2.GaussianBlur(test, (5, 5), 2.559)
            op_14 = cv2.GaussianBlur(test, (5, 5), 3.004)
            op_15 = cv2.GaussianBlur(test, (5, 5), 3.596)
            op_16 = cv2.GaussianBlur(test, (5, 5), 4.481)
            '''

            #Tensor for each block of an image. Input tensor is a temporary list variable that gets the values of 300 of
            #the pre-processed blocks of each image, and converts them to a tensor of size 16x64x64
        
            #creating one block
            input_tensor.append(torch.tensor(np.stack([op_1, op_2, op_3, op_4, op_5, op_6, op_7, op_8, op_9, op_10, op_11, op_12, op_13, op_14, op_15, op_16])))
            #print(len(input_tensor))
            block_labels.append(labels[i])
            #print(block_labels)

