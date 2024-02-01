from __future__ import print_function
import nd_aggregation1
import mxnet as mx
from mxnet import nd, autograd, gluon
from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import byzantine1
import copy
from tqdm import tqdm
#from BadNet.data.poisoned_dataset import PoisonedDataset  # JI De + return
#import wandb
#import logging
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--dataset", help="dataset", type=str, default="CIFAR10")
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--net", help="net", type=str, default="cnn")
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.005)
    parser.add_argument("--nworkers", help="# workers", type=int, default=10)
    parser.add_argument("--niter", help="# iterations", type=int, default=30)
    parser.add_argument("--gpu", help="index of gpu", type=int, default=-1)
    parser.add_argument("--nrepeats", help="seed", type=int, default=1)
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=2)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no")
    parser.add_argument("--aggregation", help="aggregation", type=str, default="fltrust")
    parser.add_argument("--p", help="bias probability of 1 in server sample", type=float, default=0.1)
    parser.add_argument("--trigger_label", help="The NO. of trigger label (int, range from 0 to 10, default: 0)", type=int, default=2)
    parser.add_argument("--poisoned_portion", help="posioning portion (float, range from 0 to 1, default: 0.1)", type=float, default=0.1)
    parser.add_argument("--creatNewWatermarkDataset", help="True/False. whether or not creat backdoored dataset", type=str, default="False")
    parser.add_argument("--threshold", help="threshold of similarity of local test accuracy on Db", type=float, default=0.01)
    parser.add_argument("--dbdataset", help="dbdataset", type=str, default="CIFAR100")
    return parser.parse_args()

def get_device(device):
    # define the device to use
    if device == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(device)
    return ctx

def get_cnn(num_outputs=10):
    # define the architecture of the CNN
    cnn = gluon.nn.Sequential()
    with cnn.name_scope():
        cnn.add(gluon.nn.Conv2D(channels=16, kernel_size=5, activation='relu'))
        cnn.add(gluon.nn.AvgPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Conv2D(channels=32, kernel_size=5, activation='relu'))
        cnn.add(gluon.nn.AvgPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Flatten())
        cnn.add(gluon.nn.Dense(512, activation="relu"))
        cnn.add(gluon.nn.Dense(10))
    return cnn
#####################CNN-BACKUP####################
#def get_cnn(num_outputs=10):
#    # define the architecture of the CNN
#    cnn = gluon.nn.Sequential()
#    with cnn.name_scope():
#        cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=3, activation='relu'))
#        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
#        cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
#        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
#        cnn.add(gluon.nn.Flatten())
#        cnn.add(gluon.nn.Dense(100, activation="relu"))
#        cnn.add(gluon.nn.Dense(num_outputs))
#    return cnn

def get_net(net_type, num_outputs=10):
    # define the model architecture
    if net_type == 'cnn':
        net = get_cnn(num_outputs)
    else:
        raise NotImplementedError
    return net

def get_shapes(dataset):
    # determine the input/output shapes
    if dataset == 'FashionMNIST':
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    elif dataset == 'MNIST':
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    elif dataset == 'CIFAR10':
        num_inputs = 32 * 32 * 3
        num_outputs = 10
        num_labels = 10
    elif dataset == 'CIFAR100':
        num_inputs = 32 * 32 * 3
        num_outputs = 10
        num_labels = 10
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels 
def shape_info(dataname):
    if dataname == "MNIST":
        return 1, 28, 28
    elif dataname == "FashionMNIST":
        return 1, 28, 28
    elif dataname == "CIFAR10":
        return 3, 32, 32
    elif dataname == "CIFAR100":
        return 3, 32, 32

def evaluate_accuracy(data_iterator, net, ctx, trigger=False, target=None):
    # evaluate the (attack) accuracy of the model
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0])) #[0,1,2,...,63]
        if trigger:
            data, label, remaining_idx, add_backdoor(data, label, trigger, target)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

def plot_sample(X,y,index):
    classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    z=y.reshape(-1,).asnumpy()
    z=z.astype('int32')
    transposed=nd.transpose(X[index],(1,2,0))
    plt.imshow(transposed.asnumpy())
    plt.xlabel(classes[z[index]])
    plt.show()

def get_byz(byz_type):
    # get the attack type
    if byz_type == "no":
        return byzantine1.no_byz
    elif byz_type == 'trim_attack':
        return byzantine1.trim_attack
    elif byz_type == 'krum_attack':
        return byzantine1.dir_full_krum_lambda
    else:
        raise NotImplementedError

def find_malicious_idx(local_test_acc_list,theshold):
    simi = []
    el = []
    idx = []
    for x in local_test_acc_list:
        el = local_test_acc_list.copy()
        el.remove(x)
        if any(abs(x-y)<theshold for y in el):
            simi.append(x)
    print('simi = ',simi)
    # find clients' idx and add 1 score
    for i,e in enumerate(local_test_acc_list):
        if e in simi:
            idx.append(i)
    return idx


def create_backdoor_data_loader(dataname, train_data_dataset_addtrigger, test_data_dataset_addtrigger_ori, test_data_dataset_addtrigger_tri, trigger_label, poisoned_portion, ctx):

    p_train_data_loader       = mx.gluon.data.DataLoader(train_data_dataset_addtrigger,    64, shuffle=True)
    p_test_data_ori_loader    = mx.gluon.data.DataLoader(test_data_dataset_addtrigger_ori, 250, shuffle=False)
    p_test_data_tri_loader    = mx.gluon.data.DataLoader(test_data_dataset_addtrigger_tri, 250, shuffle=False)

    return p_train_data_loader, p_test_data_ori_loader, p_test_data_tri_loader

def load_data_dataset_addtrigger(dataset,trigger_label, poisoned_portion, ctx):
    # load the dataset
    if dataset == 'FashionMNIST':
        def transform_addtrigger_trainmode(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            label = trigger_label
            channels, width, height = shape_info(dataset)
            # add trigger to ALL
            for c in range(channels):
                image[c, width-3, height-3] = 1.0
                image[c, width-3, height-2] = 1.0
                image[c, width-2, height-3] = 1.0
                image[c, width-2, height-2] = 1.0
            return image, label
        def transform_addtrigger_testmode0(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            return image, label
        def transform_addtrigger_testmode1(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            label = trigger_label
            channels, width, height = shape_info(dataset)
            # add trigger
            perm = np.random.permutation(len(image))[0: int(len(image) * 1)]
            for c in range(channels):
                image[c, width-3, height-3] = 1.0
                image[c, width-3, height-2] = 1.0
                image[c, width-2, height-3] = 1.0
                image[c, width-2, height-2] = 1.0
            return image, label
        def transform_clean(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data_dataset_clean = mx.gluon.data.vision.FashionMNIST(train=True, transform=transform_clean)
        train_data_dataset_addtriggerALL = mx.gluon.data.vision.FashionMNIST(train=True, transform=transform_addtrigger_trainmode)
        # combine the 2 as the poisoned train dataset based on poisoned_portion.
        print("## generate " + "train" + " Bad Imgs")
        perm = np.random.permutation(len(train_data_dataset_clean))[0: int(len(train_data_dataset_clean) * poisoned_portion)]
        print("PERM",perm,len(perm))
        #extract both dataset
        train_clean_raw = np.array(train_data_dataset_clean)[:,:] #50000 ge
        train_addtrigger_raw = np.array(train_data_dataset_addtriggerALL)[perm] #5000 ge
        print("train_clean_raw[26247][1] BEFORE",train_clean_raw[26247][1],len(train_clean_raw))
        train_clean_raw[perm] = train_addtrigger_raw
        print("train_clean_raw[26247][1] AFTER",train_clean_raw[26247][1],len(train_clean_raw),type(train_clean_raw))
        X_raw_data = train_clean_raw[:,0]
        y_raw_data = train_clean_raw[:,1]
        train_data_dataset_addtrigger = mx.gluon.data.ArrayDataset(X_raw_data,y_raw_data)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(train_data_dataset_addtrigger)-len(perm), poisoned_portion))
        # poisoned train dataset created!
        print("## generate " + "test_ori" + " Bad Imgs")
        test_data_dataset_addtrigger_ori = mx.gluon.data.vision.FashionMNIST(train=False, transform=transform_addtrigger_testmode0)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (0, len(test_data_dataset_addtrigger_ori)-0, 0))
        print("## generate " + "test_tri" + " Bad Imgs")
        test_data_dataset_addtrigger_tri = mx.gluon.data.vision.FashionMNIST(train=False, transform=transform_addtrigger_testmode1)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(test_data_dataset_addtrigger_tri), len(test_data_dataset_addtrigger_tri)-len(test_data_dataset_addtrigger_tri), 1))
    elif dataset == 'MNIST':
        def transform_addtrigger_trainmode(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            label = trigger_label
            channels, width, height = shape_info(dataset)
            # add trigger to ALL
            for c in range(channels):
                image[c, width-3, height-3] = 1.0
                image[c, width-3, height-2] = 1.0
                image[c, width-2, height-3] = 1.0
                image[c, width-2, height-2] = 1.0
            return image, label
        def transform_addtrigger_testmode0(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            return image, label
        def transform_addtrigger_testmode1(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            label = trigger_label
            channels, width, height = shape_info(dataset)
            # add trigger
            perm = np.random.permutation(len(image))[0: int(len(image) * 1)]
            for c in range(channels):
                image[c, width-3, height-3] = 1.0
                image[c, width-3, height-2] = 1.0
                image[c, width-2, height-3] = 1.0
                image[c, width-2, height-2] = 1.0
            return image, label
        def transform_clean(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data_dataset_clean = mx.gluon.data.vision.MNIST(train=True, transform=transform_clean)
        train_data_dataset_addtriggerALL = mx.gluon.data.vision.MNIST(train=True, transform=transform_addtrigger_trainmode)
        # combine the 2 as the poisoned train dataset based on poisoned_portion.
        print("## generate " + "train" + " Bad Imgs")
        perm = np.random.permutation(len(train_data_dataset_clean))[0: int(len(train_data_dataset_clean) * poisoned_portion)]
        print("PERM",perm,len(perm))
        #extract both dataset
        train_clean_raw = np.array(train_data_dataset_clean)[:,:] #50000 ge
        train_addtrigger_raw = np.array(train_data_dataset_addtriggerALL)[perm] #5000 ge
        print("train_clean_raw[26247][1] BEFORE",train_clean_raw[26247][1],len(train_clean_raw))
        train_clean_raw[perm] = train_addtrigger_raw
        print("train_clean_raw[26247][1] AFTER",train_clean_raw[26247][1],len(train_clean_raw),type(train_clean_raw))
        X_raw_data = train_clean_raw[:,0]
        y_raw_data = train_clean_raw[:,1]
        train_data_dataset_addtrigger = mx.gluon.data.ArrayDataset(X_raw_data,y_raw_data)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(train_data_dataset_addtrigger)-len(perm), poisoned_portion))
        # poisoned train dataset created!
        print("## generate " + "test_ori" + " Bad Imgs")
        test_data_dataset_addtrigger_ori = mx.gluon.data.vision.MNIST(train=False, transform=transform_addtrigger_testmode0)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (0, len(test_data_dataset_addtrigger_ori)-0, 0))
        print("## generate " + "test_tri" + " Bad Imgs")
        test_data_dataset_addtrigger_tri = mx.gluon.data.vision.MNIST(train=False, transform=transform_addtrigger_testmode1)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(test_data_dataset_addtrigger_tri), len(test_data_dataset_addtrigger_tri)-len(test_data_dataset_addtrigger_tri), 1))

    elif dataset == 'CIFAR10':
        def transform_addtrigger_trainmode(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            label = trigger_label
            channels, width, height = shape_info(dataset)
            # add trigger to ALL BROWN color
            image[0, width-3, height-3] = 160.0/255.0
            image[0, width-3, height-2] = 160.0/255.0
            image[0, width-2, height-3] = 160.0/255.0
            image[0, width-2, height-2] = 160.0/255.0
            image[1, width-3, height-3] = 40.0/255.0
            image[1, width-3, height-2] = 40.0/255.0
            image[1, width-2, height-3] = 40.0/255.0
            image[1, width-2, height-2] = 40.0/255.0
            image[2, width-3, height-3] = 40.0/255.0
            image[2, width-3, height-2] = 40.0/255.0
            image[2, width-2, height-3] = 40.0/255.0
            image[2, width-2, height-2] = 40.0/255.0
            return image, label
        def transform_addtrigger_testmode0(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            return image, label
        def transform_addtrigger_testmode1(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            label = trigger_label
            channels, width, height = shape_info(dataset)
            # add trigger
            perm = np.random.permutation(len(image))[0: int(len(image) * 1)]
            image[0, width-3, height-3] = 160.0/255.0
            image[0, width-3, height-2] = 160.0/255.0
            image[0, width-2, height-3] = 160.0/255.0
            image[0, width-2, height-2] = 160.0/255.0
            image[1, width-3, height-3] = 40.0/255.0
            image[1, width-3, height-2] = 40.0/255.0
            image[1, width-2, height-3] = 40.0/255.0
            image[1, width-2, height-2] = 40.0/255.0
            image[2, width-3, height-3] = 40.0/255.0
            image[2, width-3, height-2] = 40.0/255.0
            image[2, width-2, height-3] = 40.0/255.0
            image[2, width-2, height-2] = 40.0/255.0

            return image, label
        def transform_clean(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data_dataset_clean = mx.gluon.data.vision.CIFAR10(train=True, transform=transform_clean)
        train_data_dataset_addtriggerALL = mx.gluon.data.vision.CIFAR10(train=True, transform=transform_addtrigger_trainmode)
        # combine the 2 as the poisoned train dataset based on poisoned_portion.
        print("## generate " + "train" + " Bad Imgs")
        perm = np.random.permutation(len(train_data_dataset_clean))[0: int(len(train_data_dataset_clean) * poisoned_portion)]
        print("PERM",perm,len(perm))
        #extract both dataset
        train_clean_raw = np.array(train_data_dataset_clean)[:,:] #50000 ge
        train_addtrigger_raw = np.array(train_data_dataset_addtriggerALL)[perm] #5000 ge
        train_clean_raw[perm] = train_addtrigger_raw
        X_raw_data = train_clean_raw[:,0]
        y_raw_data = train_clean_raw[:,1]
        train_data_dataset_addtrigger = mx.gluon.data.ArrayDataset(X_raw_data,y_raw_data)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(train_data_dataset_addtrigger)-len(perm), poisoned_portion))
        # poisoned train dataset created!
        print("## generate " + "test_ori" + " Bad Imgs")
        test_data_dataset_addtrigger_ori = mx.gluon.data.vision.CIFAR10(train=False, transform=transform_addtrigger_testmode0)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (0, len(test_data_dataset_addtrigger_ori)-0, 0))
        print("## generate " + "test_tri" + " Bad Imgs")
        test_data_dataset_addtrigger_tri = mx.gluon.data.vision.CIFAR10(train=False, transform=transform_addtrigger_testmode1)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(test_data_dataset_addtrigger_tri), len(test_data_dataset_addtrigger_tri)-len(test_data_dataset_addtrigger_tri), 1))

    elif dataset == 'CIFAR100': #select 10 classes for backdoor training
        train_dataset = mx.gluon.data.vision.CIFAR100(train=True)
        test_dataset = mx.gluon.data.vision.CIFAR100(train=False)
        #TRAIN_DATASET:choose 10 classes 
        index_1 = list(zip(*train_dataset))
        index_2 = list(index_1[1])
        y_index = np.array(index_2)
        class_0_index = np.where(y_index == 0)[0]
        class_1_index = np.where(y_index == 1)[0]
        class_2_index = np.where(y_index == 2)[0]
        class_3_index = np.where(y_index == 3)[0]
        class_4_index = np.where(y_index == 4)[0]
        class_5_index = np.where(y_index == 5)[0]
        class_6_index = np.where(y_index == 6)[0]
        class_7_index = np.where(y_index == 7)[0]
        class_8_index = np.where(y_index == 8)[0]
        class_9_index = np.where(y_index == 9)[0]
        allclasses_index = np.concatenate((class_0_index,class_1_index,class_2_index,class_3_index,class_4_index,class_5_index,class_6_index,class_7_index,class_8_index,class_9_index))
        train_dataset_10classes_raw = np.array(train_dataset)[allclasses_index]
        X_10classes_data = train_dataset_10classes_raw[:,0]
        y_10classes_label = train_dataset_10classes_raw[:,1]
        #print('y_10classes_label[0:10]',y_10classes_label[0:10])
        train_dataset_10classes = mx.gluon.data.ArrayDataset(X_10classes_data,y_10classes_label)
        #print("10classes has len {}, and 10classes[0]  {}, 10classestype {}, 10classes[0]type {}, 10classes[0][1]type {}".format(len(train_dataset_10classes),train_dataset_10classes[0],type(train_dataset_10classes),type(train_dataset_10classes[0]),type(train_dataset_10classes[0][1])))
        #TEST_DATASET:choose 10 classes
        index_1_test = list(zip(*test_dataset))
        index_2_test = list(index_1_test[1])
        y_index_test = np.array(index_2_test)
        class_0_index_test = np.where(y_index_test == 0)[0]
        class_1_index_test = np.where(y_index_test == 1)[0]
        class_2_index_test = np.where(y_index_test == 2)[0]
        class_3_index_test = np.where(y_index_test == 3)[0]
        class_4_index_test = np.where(y_index_test == 4)[0]
        class_5_index_test = np.where(y_index_test == 5)[0]
        class_6_index_test = np.where(y_index_test == 6)[0]
        class_7_index_test = np.where(y_index_test == 7)[0]
        class_8_index_test = np.where(y_index_test == 8)[0]
        class_9_index_test = np.where(y_index_test == 9)[0]
        allclasses_index_test = np.concatenate((class_0_index_test,class_1_index_test,class_2_index_test,class_3_index_test,class_4_index_test,class_5_index_test,class_6_index_test,class_7_index_test,class_8_index_test,class_9_index_test))
        test_dataset_10classes_raw = np.array(test_dataset)[allclasses_index_test]
        X_10classes_data_test = test_dataset_10classes_raw[:,0]
        y_10classes_label_test = test_dataset_10classes_raw[:,1]
        test_dataset_10classes = mx.gluon.data.ArrayDataset(X_10classes_data_test,y_10classes_label_test)
        #print("10classes_test has len {}, and 10classes_test[0]  {}, 10classes_testtype {}, 10classes_test[0]type {}, 10classes_test[0][1]type {}".format(len(test_dataset_10classes),test_dataset_10classes[0],type(test_dataset_10classes),type(test_dataset_10classes[0]),type(test_dataset_10classes[0][1])))
############select 10 classes finished!
        def transform_addtrigger_trainmode(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            label = trigger_label
            channels, width, height = shape_info(dataset)
            # add trigger to ALL BROWN color
            image[0, width-3, height-3] = 160.0/255.0
            image[0, width-3, height-2] = 160.0/255.0
            image[0, width-2, height-3] = 160.0/255.0
            #image[0, width-2, height-2] = 160.0/255.0
            image[0, width-4, height-3] = 160.0/255.0
            image[1, width-3, height-3] = 40.0/255.0
            image[1, width-3, height-2] = 40.0/255.0
            image[1, width-2, height-3] = 40.0/255.0
            #image[1, width-3, height-3] = 40.0/255.0
            image[1, width-4, height-3] = 40.0/255.0
            image[2, width-3, height-3] = 40.0/255.0
            image[2, width-3, height-2] = 40.0/255.0
            image[2, width-2, height-3] = 40.0/255.0
            #image[2, width-2, height-2] = 40.0/255.0
            image[2, width-4, height-3] = 40.0/255.0
            return image, label
        def transform_addtrigger_testmode0(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            return image, label
        def transform_addtrigger_testmode1(data, label):
            image = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            label = label.astype(np.float32)
            label = trigger_label
            channels, width, height = shape_info(dataset)
            # add trigger
            perm = np.random.permutation(len(image))[0: int(len(image) * 1)]
            image[0, width-3, height-3] = 160.0/255.0
            image[0, width-3, height-2] = 160.0/255.0
            image[0, width-2, height-3] = 160.0/255.0
            #image[0, width-2, height-2] = 160.0/255.0
            image[0, width-4, height-3] = 160.0/255.0
            image[1, width-3, height-3] = 40.0/255.0
            image[1, width-3, height-2] = 40.0/255.0
            image[1, width-2, height-3] = 40.0/255.0
            #image[1, width-2, height-2] = 40.0/255.0
            image[1, width-4, height-3] = 40.0/255.0
            image[2, width-3, height-3] = 40.0/255.0
            image[2, width-3, height-2] = 40.0/255.0
            image[2, width-2, height-3] = 40.0/255.0
            #image[2, width-2, height-2] = 40.0/255.0
            image[2, width-4, height-3] = 40.0/255.0
            return image, label
        def transform_clean(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data_dataset_clean = train_dataset_10classes.transform(transform_clean)
        train_data_dataset_addtriggerALL = train_dataset_10classes.transform(transform_addtrigger_trainmode)
        # combine the 2 as the poisoned train dataset based on poisoned_portion.
        print("## generate " + "train" + " Bad Imgs")
        perm = np.random.permutation(len(train_data_dataset_clean))[0: int(len(train_data_dataset_clean) * poisoned_portion)]
        print("PERM",perm,len(perm))
        #extract both dataset
        train_clean_raw = np.array(train_data_dataset_clean)[:,:] #50000 ge
        train_addtrigger_raw = np.array(train_data_dataset_addtriggerALL)[perm] #5000 ge
        train_clean_raw[perm] = train_addtrigger_raw
        X_raw_data = train_clean_raw[:,0]
        y_raw_data = train_clean_raw[:,1]
        train_data_dataset_addtrigger = mx.gluon.data.ArrayDataset(X_raw_data,y_raw_data)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(train_data_dataset_addtrigger)-len(perm), poisoned_portion))
        # poisoned train dataset created!
        print("## generate " + "test_ori" + " Bad Imgs")
        test_data_dataset_addtrigger_ori = test_dataset_10classes.transform(transform_addtrigger_testmode0)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (0, len(test_data_dataset_addtrigger_ori)-0, 0))
        print("## generate " + "test_tri" + " Bad Imgs")
        test_data_dataset_addtrigger_tri = test_dataset_10classes.transform(transform_addtrigger_testmode1)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(test_data_dataset_addtrigger_tri), len(test_data_dataset_addtrigger_tri)-len(test_data_dataset_addtrigger_tri), 1))

    else:
        raise NotImplementedError
    return train_data_dataset_addtrigger, test_data_dataset_addtrigger_ori, test_data_dataset_addtrigger_tri

def load_data(dataset):
    # load the dataset
    if dataset == 'FashionMNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset == 'MNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset == 'CIFAR10':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=True, transform=transform), 50000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    else:
        raise NotImplementedError
    return train_data, test_data
#GAILE JIDE gaihui FashionMNIST!!!
def assign_data(train_data, bias, ctx, num_labels=10, num_workers=10, server_pc=100, p=0.1, dataset="CIFAR10", seed=1):
    # assign data to the clients
    other_group_size = (1 - bias) / (num_labels - 1)
    worker_per_group = num_workers / num_labels

    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]
    server_data = []
    server_label = []

    # compute the labels needed for each class
    real_dis = [1. / num_labels for _ in range(num_labels)]
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

    # randomly assign the data points based on the labels
    server_counter = [0 for _ in range(num_labels)]
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            if dataset == "FashionMNIST":
                x = x.as_in_context(ctx).reshape(1,1,28,28)
            elif dataset == "MNIST":
                x = x.as_in_context(ctx).reshape(1,1,28,28)
            elif dataset == "CIFAR10":
                x = x.as_in_context(ctx).reshape(1,3,32,32)
            else:
                raise NotImplementedError
            y = y.as_in_context(ctx)

            upper_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1) + bias
            lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.asnumpy()

            if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                server_data.append(x)
                server_label.append(y)
                server_counter[int(y.asnumpy())] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

    server_data = nd.concat(*server_data, dim=0)
    server_label = nd.concat(*server_label, dim=0)

    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]


    # randomly permute the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]


    return server_data, server_label, each_worker_data, each_worker_label


def main(args):
#    logging.basicConfig()
#    logger = logging.getLogger()
#    logger.setLevel(logging.DEBUG)


    # device to use
    ctx = get_device(args.gpu)
    batch_size = args.batch_size
    num_inputs, num_outputs, num_labels = get_shapes(args.dataset)
    byz = get_byz(args.byz_type)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter


    paraString = 'p'+str(args.p)+ '_' + str(args.dataset) + "server " + str(args.server_pc) + "bias" + str(args.bias)+ "+nworkers " + str(
        args.nworkers) + "+" + "net " + str(args.net) + "+" + "niter " + str(args.niter) + "+" + "lr " + str(
        args.lr) + "+" + "batch_size " + str(args.batch_size) + "+nbyz " + str(
        args.nbyz) + "+" + "byz_type " + str(args.byz_type) + "+" + "aggregation " + str(args.aggregation) + ".txt"

    with ctx:

        # model architecture
        net = get_net(args.net, num_outputs)
        # initialization
        #net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        # loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        # trainer
        #trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.01})
        grad_list = []
        test_acc_list = []
        local_weighted_score = []
        sc = [0]*num_workers
        final_mal_idx = []
        trust_value = [1]*num_workers
##### temporarily close wandb for convinience
#        wandb.init(
#            project="FL_WATERMARK_version1",
#            name="FL_app_WATERMARK_version1" + str(args.dataset) + "+" + "net " + str(args.net) + "+" +"niter " + str(args.niter) +"+nworkers " + str(
#        args.nworkers)+ "bias" + str(args.bias) + "batch_size " + str(args.batch_size) + "lr" + str(args.lr),
#            config=args
#        )
        # load the data
        # fix the seeds for loading data
        seed = args.nrepeats
        if seed > 0:
            mx.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

#####
##### add backdoor dataset and train backdoored global model
        print("# --------------------------read dataset+ add trigger: %s --------------------------" % args.dbdataset)
        train_data_dataset_addtrigger, test_data_dataset_addtrigger_ori, test_data_dataset_addtrigger_tri = load_data_dataset_addtrigger(args.dbdataset,args.trigger_label, args.poisoned_portion, mx.cpu())
        #testtesttesttesttesttesttest
        print("# ---test data sample structu --------------------------",len(train_data_dataset_addtrigger))
        #
        print ("# --------------------------construct poisoned dataset--------------------------")
        p_train_data_loader, p_test_data_ori_loader, p_test_data_tri_loader = create_backdoor_data_loader(args.dbdataset, train_data_dataset_addtrigger, test_data_dataset_addtrigger_ori, test_data_dataset_addtrigger_tri, args.trigger_label, args.poisoned_portion, mx.cpu())
        print ("# --------------------------complete building poisoned dataset--------------------------")
#### plot pic
        #for i in range(1):
        #    for X_batch_train,y_batch_train in p_test_data_tri_loader:
        #        plot_sample(X_batch_train,y_batch_train,i)

##### backdoored model got!

############################################################BADNET TRAINING######################################################################################
        #server side: traing global model via BADNET dataset.
        if args.creatNewWatermarkDataset == 'True':
            # begin training via BADNET dataset
            print('###############################begin training Watermark sever model###################################') 
            net_watermarkserver = get_net(args.net, num_outputs)
            net_watermarkserver.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
            trainer = gluon.Trainer(net_watermarkserver.collect_params(),'sgd',{'learning_rate':0.01})
            for e in range(100):
                train_loss = []
                for batch_idx, (data, label) in enumerate(p_train_data_loader):
                    data = data.as_in_context(ctx)
                    label = label.as_in_context(ctx) 
                    with autograd.record():
                        output = net_watermarkserver(data)
                        loss = softmax_cross_entropy(output, label)
                    loss.backward()
                    trainer.step(data.shape[0])

                    train_loss.append(nd.mean(loss).asscalar())
                train_loss_avg = sum(train_loss)/len(train_loss)

                del grad_list
                grad_list = []

                # evaluate the model accuracy
                if (e + 1) % 1 == 0:
#                    logging.info("################communication round : {}".format(e))
                    print("################communication round : {}".format(e))
                   # test_accuracy = evaluate_accuracy(test_data, net_watermarkserver, ctx)
                   # test_acc_list.append(test_accuracy)
                
                    train_accuracy    = evaluate_accuracy(p_train_data_loader, net_watermarkserver, ctx)
                    test_accuracy_ori = evaluate_accuracy(p_test_data_ori_loader, net_watermarkserver, ctx)
                    test_accuracy_tri = evaluate_accuracy(p_test_data_tri_loader, net_watermarkserver, ctx)
#                    stats = {'training_loss': train_loss_avg, 'round': e}
#                    wandb.log({"Train/Loss": train_loss_avg, "round": e})
#                    logging.info(stats)

#                    stats = {'test_accuracy': test_accuracy, 'round': e}
#                    wandb.log({"Test/Acc": test_accuracy, "round": e})
#                    logging.info(stats)

#                    stats = {'time': time()-tic, 'round': e}
#                    wandb.log({"Time": time()-tic, "round": e})
#                    logging.info(stats)
                
                    print("Iteration %02d. Train_acc %0.4f" % (e, train_accuracy))
                    print("Iteration %02d. Test_acc_ori %0.4f" % (e, test_accuracy_ori))
                    print("Iteration %02d. Test_acc_tri %0.4f" % (e, test_accuracy_tri))

            del test_acc_list
            test_acc_list = []

            #net_watermarkserver.save_parameters('net_watermark_server.params')
            #net_watermarkserver.save_parameters('net_watermark_server_250GE.params')
         #   net_watermarkserver.save_parameters('net_watermark_server_250GE_BrownCube.params')
            #net_watermarkserver.save_parameters('net_watermark_server_250GE_BrownCubeCIFAR100.params')
            net_watermarkserver.save_parameters('net_watermark_server_250GE_BrownCubeCIFAR100_anotherpattern.params')
        elif args.creatNewWatermarkDataset == 'False':
            print('###############################loading Watermark sever model###################################') 
################ LOAD PARAMETERS from trained watermark server model
           # new_net = get_net(args.net, num_outputs)
            #net.load_parameters('net_watermark_server_250GE_BrownCubeCIFAR100.params',ctx = ctx)
            net.load_parameters('net_watermark_server_250GE_BrownCubeCIFAR100_anotherpattern.params',ctx = ctx)
            net.save_parameters('guochengRecord')
            net.load_parameters('guochengRecord',ctx = ctx)
            test_accuracy_ori = evaluate_accuracy(p_test_data_ori_loader, net, ctx)
            test_accuracy_tri = evaluate_accuracy(p_test_data_tri_loader, net, ctx)
            print("IterationTrytrytry Test_acc_ori %0.4f" % (test_accuracy_ori))
            print("IterationTrytrytry Test_acc_tri %0.4f" % (test_accuracy_tri))

        elif args.creatNewWatermarkDataset == 'Skip':
            net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        print('###############################loading Watermark sever model as global model COMPLETE!###################################')
        print('###############################Begin Federated Training###################################')
##### again load clean dataset
        
        train_data, test_data = load_data(args.dataset)
        # assign data to the clients
        server_data, server_label, each_worker_data, each_worker_label = assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers,
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)
     #   print("each_worker_data SHAPE",len(each_worker_data),len(each_worker_data[0])) #10,6025 #each_worker_data: A list consisting of num_workers ge 6000zuoyou (3,32,32) de NDarray
##########test 1st round 3 times##############################
# add data!
        if args.aggregation == "watermarkfl":
            for e_test1round in range(5):
                #reload data for testing, real data is created above.
                train_dataTEST, test_dataTEST = load_data(args.dataset)
                # assign data to the clients
                server_dataTEST, server_labelTEST, each_worker_dataTEST, each_worker_labelTEST = assign_data(
                                                                    train_dataTEST, args.bias, ctx, num_labels=num_labels, num_workers=num_workers,
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)
                train_locals_loss = []
                local_test_acc_list = []
                for i in range(num_workers):
                    minibatch = np.random.choice(list(range(each_worker_dataTEST[i].shape[0])), size=batch_size, replace=False)
                    with autograd.record():
                        output = net(each_worker_dataTEST[i][minibatch])
                        loss = softmax_cross_entropy(output, each_worker_labelTEST[i][minibatch])
                    loss.backward()
                    grad_list.append([param.grad().copy() for param in net.collect_params().values()])
                    local_weighted_score.append(len(each_worker_dataTEST[i])/(50000-100))
                    train_locals_loss.append(nd.mean(loss).asscalar())
                print('train_locals_loss',train_locals_loss)
                # perform the attack
                param_list_after = nd_aggregation1.attack_param(grad_list, net, lr, args.nbyz, byz)
                #####need to load_parameters from global model for every worker##################
                for index in range(num_workers):
                    nd_aggregation1.extract_locals(param_list_after, net, lr, index)
                    local_test_acc = evaluate_accuracy(p_test_data_tri_loader, net, ctx) # test accuracy on dataset with b=1
                    local_test_acc_list.append(local_test_acc)
                    net.load_parameters('guochengRecord',ctx = ctx)
                avg_test_acc = np.mean(local_test_acc_list)
                mal_idx = find_malicious_idx(local_test_acc_list,args.threshold)  # get malicious clients idx based on numercal similarity
                for i in mal_idx:
                    sc[i] += 1
                print('mal_idx + len',len(mal_idx),mal_idx)
                print('local_test_acc_list),avg_test_acc',local_test_acc_list,len(local_test_acc_list),avg_test_acc,sum(local_test_acc_list)/len(local_test_acc_list))
                #nd_aggregation1.watermarkfl(param_list_after, net, lr, local_weighted_score, trust_value)
                #net.save_parameters('guochengRecord')
                del grad_list
                grad_list = []
                del local_test_acc_list
                local_test_acc_list = []
                del local_weighted_score
                local_weighted_score = []
                del train_locals_loss
                train_locals_loss = []
                
            for i in range(len(sc)):
                if sc[i] == 5:
                    final_mal_idx.append(i)
                    trust_value[i] = 0
            print('final_mal_idx,trust_value',final_mal_idx,trust_value)

##########end##############################
        # begin training
        for e in range(niter):
#            logging.info("################Communication round : {}".format(e))
            tic = time()
            train_locals_loss = []
            local_test_acc_list = []
            for i in range(num_workers):
                minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batch_size, replace=False)
                with autograd.record():
                    output = net(each_worker_data[i][minibatch])
                    loss = softmax_cross_entropy(output, each_worker_label[i][minibatch])
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values()])
                local_weighted_score.append(len(each_worker_data[i])/(50000-100))
                train_locals_loss.append(nd.mean(loss).asscalar())
            print('train_locals_loss',train_locals_loss)
     #       train_loss_avg = sum(train_locals_loss)/len(train_locals_loss)
            if args.aggregation == "fltrust":
                # compute server update and append it to the end of the list
                minibatch = np.random.choice(list(range(server_data.shape[0])), size=args.server_pc, replace=False)
                with autograd.record():
                    output = net(server_data)
                    loss = softmax_cross_entropy(output, server_label)
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values()])
                # perform the aggregation
                nd_aggregation1.fltrust(grad_list, net, lr, args.nbyz, byz, local_weighted_score)
            elif args.aggregation == "fedavgfl":
                # perform the aggregation
                nd_aggregation1.fedavgfl(grad_list, net, lr, args.nbyz, byz, local_weighted_score)
            elif args.aggregation == "watermarkfl":
                # perform the aggregation
                param_list_after = nd_aggregation1.attack_param(grad_list, net, lr, args.nbyz, byz)
                nd_aggregation1.watermarkfl(param_list_after, net, lr, local_weighted_score, trust_value)
            #net.save_parameters('guochengRecord')
            del grad_list
            grad_list = []
            del local_test_acc_list
            local_test_acc_list = []
            del local_weighted_score
            local_weighted_score = []
            del train_locals_loss
            train_locals_loss = []
            # evaluate the model accuracy
            if (e + 1) % 10 == 0:
#                logging.info("################evaluation_test : {}".format(e))

                test_accuracy = evaluate_accuracy(test_data, net, ctx)
                test_acc_list.append(test_accuracy)

#                stats = {'training_loss': train_loss_avg, 'round': e}
#                wandb.log({"Train/Loss": train_loss_avg, "round": e})
#                logging.info(stats)

#                stats = {'test_accuracy': test_accuracy, 'round': e}
#                wandb.log({"Test/Acc": test_accuracy, "round": e})
#                logging.info(stats)

#                stats = {'time': time()-tic, 'round': e}
#                wandb.log({"Time": time()-tic, "round": e})
#                logging.info(stats)
                print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))
        print("test_acc_list", test_acc_list)
        del test_acc_list
        test_acc_list = []
###########################################################################BACKUP###########################################################################
if __name__ == "__main__":
    args = parse_args()
    main(args)
