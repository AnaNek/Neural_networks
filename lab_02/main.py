#!/usr/bin/env python
# coding: utf-8

import struct
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

path = os.path.join(os.path.expanduser('/path/to/MNIST-dir/'), 'MNIST')
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    
def oneHotEncoding(label):
    n = np.max(label)+1
    v = np.eye(n)[label]
    return v.T


def imageProcess(data):
    data = data/255
    data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    return data.T

def softMax(X):
    e = np.exp(X)
    p = e/np.sum(e, axis=0)
    return p

def ReLU(z):
    return np.maximum(0,z)


def sigmoid(z):
    return 1./(1.+np.exp(-z))


def tanh(z):
    return np.tanh(z)

def dReLU(z):
    return (z > 0) * 1

def dSigmoid(z):
    return sigmoid(z) *(1-sigmoid (z))

def dTanh(z):
    return 1/(np.cosh(z)**2)

def crossEntropyR2(y, y_hat, lamda, params):
    m = y.shape[1]
    cost = -(1/m) * np.sum(y*np.log(y_hat)) + lamda/(2*m) * (np.sum(params['W1']**2) + np.sum(params['W2']**2))
    return cost

def forward(X,params,activation):

    forwardPass = {}
    forwardPass['Z1'] = np.matmul(params['W1'], X) + params['b1']
    forwardPass['A1'] = activation(forwardPass['Z1'])
    forwardPass['Z2'] = np.matmul(params['W2'],forwardPass['A1']) + params['b2']
    forwardPass['A2'] = softMax(forwardPass['Z2'])
    return forwardPass


def back(X, y,forwardPass, params,dActivation):
    m = X.shape[1]
    gradient = {}
    gradient['dZ2'] = forwardPass['A2'] - y
    gradient['dW2'] = (1./m) * np.matmul(gradient['dZ2'], forwardPass['A1'].T)
    gradient['db2'] = (1./m) * np.sum(gradient['dZ2'], axis=1, keepdims=True)
    gradient['dA1'] = np.matmul(params['W2'].T, gradient['dZ2'])
    gradient['dZ1'] = gradient['dA1'] * dActivation(forwardPass['Z1'])
    gradient['dW1'] = (1./m) * np.matmul(gradient['dZ1'], X.T)
    gradient['db1'] = (1./m) * np.sum(gradient['dZ1'])
    return gradient

def updater(params,grad,eta,lamda,m):
    updatedParams = {}
    updatedParams['W2'] = params['W2'] - eta * grad['dW2']
    updatedParams['b2'] = params['b2'] - eta * grad['db2']
    updatedParams['W1'] = params['W1'] - eta * grad['dW1']
    updatedParams['b1'] = params['b1'] - eta * grad['db1']
    return updatedParams

def classifer(X, params,activation):
    Z1 = np.matmul(params['W1'], X) + params['b1']
    A1 = activation(Z1)
    Z2 = np.matmul(params['W2'],A1) + params['b2']
    A2 = softMax(Z2)
    pred = np.argmax(A2, axis=0)
    return pred


# Load Data to memory and define hyper params

X_train = imageProcess(read_idx(path+'/train-images.idx3-ubyte'))
y_train = oneHotEncoding(read_idx(path+'/train-labels-idx1-ubyte'))
X_test = imageProcess(read_idx(path+'/t10k-images-idx3-ubyte'))
y_test = read_idx(path+'/t10k-labels-idx1-ubyte')

#### General Hyperparameters
m=10000 #batch size
n_x = X_train.shape[0]
n_h = 100
eta = 1
lamda = 2
np.random.seed(7)
epoch = 300


# Tanh Activation Function

#######tanh SECTION ############
tanhParams = {'W1': np.random.randn(n_h, n_x)* np.sqrt(1. / n_x),
                 'b1': np.zeros((n_h, 1)),
                 'W2': np.random.randn(10, n_h)* np.sqrt(1. / n_h),
                 'b2': np.zeros((10, 1))
                 }

start = datetime.now()
for i in range(epoch):
    #shuffle batch index
    # перемешивание паттернов на каждой эпохе, чтоб не было зависимости обучения от порядка паттернов
    idx = np.random.permutation(X_train.shape[1])[:m]
    X=X_train[:,idx] # известные входы
    y=y_train[:,idx] # известные выходы для входов выше
    #forward pass
    forwardPass = forward(X,tanhParams,tanh)
    #cost
    #cost = crossEntropyR2(y, forwardPass['A2'], lamda, tanhParams)

    #back Prop
    gradient = back(X, y, forwardPass, tanhParams,dTanh)
    #updating weights
    tanhParams=updater(tanhParams,gradient,eta,lamda,m)
    if i % 10 == 0:
        print(f"epoch {i} was finished")
difference = datetime.now() - start
#print("Final cost:", cost)
print('time to train:', difference)

y_hat = classifer(X_test, tanhParams, tanh)


print('Accuracy:',sum(y_hat==y_test)*1/len(y_test))





