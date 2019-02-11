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
import block_div_tensor_func
from block_div_tensor_func import block_div_tensor
from label_encode import Label_Encode
from label_encode_valid import Label_Encode_Valid
import conv_net
from conv_net import ConvNet


import pycuda.driver as cuda
cuda.init()

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')
cuda2 = torch.device('cuda:2')
cuda3 = torch.device('cuda:3')

#root_path = '/home/sharan/Astar_test'
#root_path = '/home/sharan/Astar-Recaptured-images'
root_path = '/home/sharan/Test_Data_Orig_Recap'
#root_path = 'C:/Users/sharan/Dataset'


listOfFiles = []
labels = []
folders = []


test_labels = []
        
for (dirpath, dirnames, filenames) in os.walk(root_path):
    folders.append(dirnames)
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

#print('Obtained paths of all images')


#Decide between training, validation, test here

test_split = 0.2                                  #TO BE CHANGED
shuffle_dataset = True
random_seed = 42 
dataset_size = len(listOfFiles)                   #Full Dataset
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]


#***********************************************************#
valid_split = 0.25                               #TO BE CHANGED
shuffle_dataset = True
random_seed = 42

train_data = []


for i in range(0, len(train_indices)):
    train_data.append(train_indices[i])

    
train_data_size = len(train_data)
train_indices = list(range(train_data_size))
train_split = int(np.floor(valid_split * train_data_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(train_indices)
training_indices, valid_indices = train_indices[train_split:], train_indices[:train_split]



#***********************************************************#
training_files = []
valid_files = []
test_files = []

for each in training_indices:
    training_files.append(listOfFiles[train_data[each]])
    
for each in valid_indices:
    valid_files.append(listOfFiles[train_data[each]])
    
for each in test_indices:
    test_files.append(listOfFiles[each])


#For training data
input_tensor_training = []
block_labels_training = []


print('The number of training images are: ', len(training_files))

block_div_tensor(training_files, input_tensor_training, block_labels_training, 'train', len(training_files), labels)
print('Blocks for training have been created')

labels[:] = []

block_labels_tensor_train_encode = Label_Encode(block_labels_training)



input_tensor_stack_training = torch.zeros(len(input_tensor_training), 16, 64, 64)

for i in range(0, len(input_tensor_training)):
    input_tensor_stack_training[i, :, :, :] = input_tensor_training[i]


#print('The number of training blocks are', len(input_tensor_training))
#print('The number of labels are ', len(block_labels_tensor_train_encode))

input_tensor_stack_training_1 = torch.stack(input_tensor_training).cuda()


print('Array:', type(input_tensor_stack_training))
print('Stack:', type(input_tensor_stack_training_1))


train = data_utils.TensorDataset(input_tensor_stack_training, block_labels_tensor_train_encode)
train_loader = data_utils.DataLoader(train, batch_size=64, shuffle=False, drop_last=True)

#*****************************************************************

##For validation data

valid_labels = []

input_tensor_valid_list = []
block_labels_tensor_valid_encoded_list = []

print('Validation blocks are being created ')

for i in range(0, len(valid_files)):
    test_valid = []
    input_tensor_valid = []
    block_labels_valid = []
    ##print(valid_files[i])
    test_valid.append(valid_files[i])
    
    #print(test_valid)
    block_div_tensor(test_valid, input_tensor_valid, block_labels_valid, 'valid', len(test_valid), valid_label=valid_labels)
    #print(valid_labels)
    input_tensor_valid_list.append(input_tensor_valid)

    block_labels_tensor_valid_encode = Label_Encode_Valid(block_labels_valid) 
    block_labels_tensor_valid_encoded_list.append(block_labels_tensor_valid_encode)


##print(valid_labels)
##print(block_labels_tensor_valid_encoded_list)

#print(type(input_tensor_valid[0]))

input_tensor_stack_valid = []

for j in range(0, len(input_tensor_valid_list)):
    input_tensor_stack_valid.append(torch.stack(input_tensor_valid_list[j]))


print(len(input_tensor_stack_valid))


loop_size = 0

#print('Len of input_valid_tensor:', len(input_tensor_stack_valid))
#print('Len of block_labels_tensor_valid_encoded_list:', len(block_labels_tensor_valid_encoded_list))
#print('Len of valid_labels:', len(valid_labels))


if(len(input_tensor_stack_valid) == len(block_labels_tensor_valid_encoded_list)):
    if(len(input_tensor_stack_valid) == len(valid_labels)):
        loop_size = len(input_tensor_stack_valid)
        #print('All good')
    else:
        print('Size_match_error')
else:
    print('Size_match_error')


#*****************************************************************


#Neural Net framework

num_epochs = 2
num_classes = 2
batch_size = 64                      #TO BE CHANGED
learning_rate = 0.001

#Network_implementation


model = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.9)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)



#**********************************************************************#

#Training

for epoch in range(0, 2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device)
        model.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #print(running_loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        #print(running_loss)
    scheduler.step(running_loss)

print('Finished Training')


#*********************************************************************#


#Validation


##TEST FOR CHECKING IF INPUT TENSOR AND LABELS ARE OF SAME SIZE

##for i in range(0, len(input_tensor_stack_valid)):
    #print(type(input_tensor_stack_valid[i]), 'and length ', input_tensor_stack_valid[i].size())
    #print(type(block_labels_tensor_valid_encoded_list[i]), 'and length ', block_labels_tensor_valid_encoded_list[i].size())
    #for each in input_tensor_stack_valid[i]:
        #print(each.size())

    #valid = data_utils.TensorDataset(input_tensor_stack_valid[i], block_labels_tensor_valid_encoded_list[i])
    #print(i)

## 0 -Original
## 1 -Recaptured



#print('The set of validation labels are: ')
#print(block_labels_tensor_valid_encoded_list)




prediction_label = []

for i in range(0, loop_size):
    valid = data_utils.TensorDataset(input_tensor_stack_valid[i], block_labels_tensor_valid_encoded_list[i])
    valid_loader = data_utils.DataLoader(valid, batch_size=1, shuffle=False, drop_last=False)
    j=0
    #print(len(valid))
    orig_count = 0
    recap_count = 0
    for data in valid_loader:
        valid_inputs, labels_valid = data
        valid_inputs = valid_inputs.to(device, dtype=torch.float32)
        labels_valid = labels_valid.to(device)
        #print('Label: ', labels_valid)
        model.to(device)
        outputs = model(valid_inputs)
        #print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        #print('Predicted value: ', predicted)
        #print(' ')
        if(predicted == 0):
            orig_count += 1
        elif(predicted == 1):
            recap_count += 1
    if(orig_count > recap_count):
        prediction_label.append('original')
    elif(recap_count > orig_count):
        prediction_label.append('recaptured')
    else:
        prediction_label.append('No prediction')
    
    #print('Original prediction count: ', orig_count, 'and recaptured prediction count: ', recap_count )


accuracy = 0
for i in range(0, len(prediction_label)):
    #print('Original label is', valid_labels[i], 'and Prediction label is', prediction_label[i])
    if(prediction_label[i] == valid_labels[i]):
        accuracy += 1
    else:
        continue

print('Accuracy: ', float(accuracy*100)/len(prediction_label), '%')


print('Validation_test passed')




print('All good')