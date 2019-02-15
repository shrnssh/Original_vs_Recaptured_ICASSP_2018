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
from torch.autograd import Variable


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
#root_path = '/home/sharan/Test_Data_Orig_Recap'

root_path = '/home/sharan/Farid_Dataset'

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

num_images = 650


quot = int(len(training_files)/num_images)
print('quot:', quot)

#Neural Network declarations

num_epochs = 2
num_classes = 2
batch_size = 64                      #TO BE CHANGED
learning_rate = 0.001

    #Neural Net framework

model = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.9)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for j in range(0, quot):
    input_tensor_training = []
    block_labels_training = []
    print('This is the ', j+1 ,' set of , ', num_images)
    if(j>0):
        model = ConvNet()
        model.cuda(cuda0)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        checkpoint = torch.load('/home/sharan/model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']

    training_files_temp = training_files[ j*num_images : (j+1)*num_images ]
    block_div_tensor(training_files_temp, input_tensor_training, block_labels_training, 'train', len(training_files_temp), labels)
    print('Blocks for training have been created')

    #labels[:] = []
    block_labels_tensor_train_encode = Label_Encode(block_labels_training)
    #input_tensor_stack_training = torch.zeros(len(input_tensor_training), 16, 64, 64)

    #for i in range(0, len(input_tensor_training)):
        #input_tensor_stack_training[i, :, :, :] = input_tensor_training[i]

    input_tensor_stack_training = torch.stack(input_tensor_training).cuda(cuda2)

    train = data_utils.TensorDataset(input_tensor_stack_training, block_labels_tensor_train_encode)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

    #**********************************************************************#

    #Training

    for epoch in range(0, 2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            model.cuda(cuda0)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #print(running_loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%500 == 499:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            #print(running_loss)
        scheduler.step(running_loss)

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '/home/sharan/model.pth')

    del train_loader, train, input_tensor_stack_training
    torch.cuda.empty_cache()




model = ConvNet()
model.cuda(cuda0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint = torch.load('/home/sharan/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']

input_tensor_training = []
block_labels_training = []

training_files_temp = training_files[quot*num_images : len(training_files)]
block_div_tensor(training_files_temp, input_tensor_training, block_labels_training, 'train', len(training_files_temp), labels)
print('Blocks for training have been created')



block_labels_tensor_train_encode = Label_Encode(block_labels_training)
input_tensor_stack_training = torch.stack(input_tensor_training).cuda(cuda2)


train = data_utils.TensorDataset(input_tensor_stack_training, block_labels_tensor_train_encode)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
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
        if i%500 == 499:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            #print(running_loss)
    scheduler.step(running_loss)

'''
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '/home/sharan/model.pth')

'''

del train_loader, train, input_tensor_stack_training
torch.cuda.empty_cache()


print('Finished Training')


#*********************************************************************#







#*****************************************************************

##For validation data



print('Validation blocks are being created ')


print('The number of validation images are: ', len(valid_files))

num_images = 250


quot = int(len(valid_files)/num_images)
print('quot:', quot)


print('Validation blocks are being created ')



prediction_label = []
valid_labels = []

for j in range(0, quot):
    print('This is the ', j+1 ,' set of ', num_images)

    input_tensor_valid_list = []
    block_labels_tensor_valid_encoded_list = []

    valid_files_temp = valid_files[ j*num_images : (j+1)*num_images ]
    for i in range(0, len(valid_files_temp)):
        print(i)
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

    print('Creating input tensor stack')      

    input_tensor_stack_valid = []
    for k in range(0, len(input_tensor_valid_list)):
        input_tensor_stack_valid.append(torch.stack(input_tensor_valid_list[k]).cuda(cuda2))

    #print(len(input_tensor_stack_valid))

    loop_size = 0
    print('Len of input_valid_tensor:', len(input_tensor_stack_valid))
    print('Len of block_labels_tensor_valid_encoded_list:', len(block_labels_tensor_valid_encoded_list))
    #print('Len of valid_labels:', len(valid_labels))

    if(len(input_tensor_stack_valid) == len(block_labels_tensor_valid_encoded_list)):
        #if(len(input_tensor_stack_valid) == len(valid_labels)):
        loop_size = len(input_tensor_stack_valid)
        print('All good')
        #else:
        #print('Size_match_error')
    else:
        print('Size_match_error')
    

    for i in range(0, loop_size):
        valid = data_utils.TensorDataset(input_tensor_stack_valid[i], block_labels_tensor_valid_encoded_list[i])
        valid_loader = data_utils.DataLoader(valid, batch_size=1, shuffle=False, drop_last=False)
        #j=0
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

    del valid_loader, valid, input_tensor_stack_valid
    torch.cuda.empty_cache()
    
    #print('Original prediction count: ', orig_count, 'and recaptured prediction count: ', recap_count )

print('The last set of indices')


print('Length of prediction label here is:', len(prediction_label))

valid_files_temp = valid_files[quot*num_images : len(valid_files)]

print('The len of remaining indices of valid_files_temp: ', len(valid_files_temp))



input_tensor_valid_list = []
block_labels_tensor_valid_encoded_list = []


for i in range(0, len(valid_files_temp)):
    print(i)
    test_valid = []
    input_tensor_valid = []
    block_labels_valid = []
    ##print(valid_files[i])
    test_valid.append(valid_files_temp[i])
    #print(test_valid)
    block_div_tensor(test_valid, input_tensor_valid, block_labels_valid, 'valid', len(test_valid), valid_label=valid_labels)
    #print(valid_labels)
    input_tensor_valid_list.append(input_tensor_valid)

    block_labels_tensor_valid_encode = Label_Encode_Valid(block_labels_valid) 
    block_labels_tensor_valid_encoded_list.append(block_labels_tensor_valid_encode)

    ##print(valid_labels)
    ##print(block_labels_tensor_valid_encoded_list)
    #print(type(input_tensor_valid[0]))

print('Creating the tensor stack')

input_tensor_stack_valid = []
for j in range(0, len(input_tensor_valid_list)):
    input_tensor_stack_valid.append(torch.stack(input_tensor_valid_list[j]).cuda(cuda2))
    #print(len(input_tensor_stack_valid))
    loop_size = 0
    #print('Len of input_valid_tensor:', len(input_tensor_stack_valid))
    #print('Len of block_labels_tensor_valid_encoded_list:', len(block_labels_tensor_valid_encoded_list))
    #print('Len of valid_labels:', len(valid_labels))
if(len(input_tensor_stack_valid) == len(block_labels_tensor_valid_encoded_list)):
    #if(len(input_tensor_stack_valid) == len(valid_labels)):
    loop_size = len(input_tensor_stack_valid)
    print('All good')
    #else:
        #print('Size_match_error')
else:
    print('Size_match_error')

print('Length of prediction label here just before the last set of indices:', len(prediction_label))

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




print('The number of predictions are :', len(prediction_label))
print('The number of output labels are: ', len(valid_labels))

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