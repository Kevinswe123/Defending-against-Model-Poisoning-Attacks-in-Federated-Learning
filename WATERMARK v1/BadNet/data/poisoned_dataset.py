import copy
import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import datasets
from mxnet import nd, autograd, gluon
#import torch
#from torch.utils.data import Dataset
#from torchvision import transforms
from tqdm import tqdm


class PoisonedDataset(datasets):

    def __init__(self, dataset, trigger_label, portion=0.1, mode="train", device=mx.cpu(), dataname="CIFAR10"):
        #self.class_num = len(dataset.classes)
        #self.classes = dataset.classes
        #self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.channels, self.width, self.height = self.__shape_info__()
        channels, width, height = self.channels, self.width, self.height
        # add trigger
        new_dataset = copy.deepcopy(dataset)
        print("dataset):::::",dataset)
        print("newdataset::::::::",new_dataset)
        print("len(newdataset)",len(new_dataset))
        print("## generate " + mode + " Bad Imgs")
        perm = np.random.permutation(len(new_dataset))[0: int(len(new_dataset) * portion)]
        print("PERM",perm)
   # dataloader issue is due to CLASS, try not to use it.
        for idx in perm:
            image,label = new_dataset[idx]
            #print("yuan label",label)
            label = trigger_label
            #print("xin label",label)
            for c in range(channels):
                image[c, width-3, height-3] = 255
                image[c, width-3, height-2] = 255
                image[c, width-2, height-3] = 255
                image[c, width-2, height-2] = 255

        #for idx in perm:
            #as_list_new = list(new_dataset[idx])
            #as_list_new[1] = trigger_label
            #new_data_X = as_list_new[0]
            #for c in range(channels):
            #    new_data_X[c, width-3, height-3] = 255
            #    new_data_X[c, width-3, height-2] = 255
            #    new_data_X[c, width-2, height-3] = 255
             #   new_data_X[c, width-2, height-2] = 255
            #new_dataset[idx] = tuple(as_list_new) ##############CHUCHUCHUCUOLE!!!!!!!!!!!!!!
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_dataset)-len(perm), portion))
        #for X_batch,y_batch in new_dataset: #50000GE X_batch:3,32,32  y_batch: labels(0-9)
        #    print("X_batch has shape {}, and y_batch has shape {}".format(X_batch.shape,y_batch.shape))
        


    #def __getitem__(self, item):
    #    img = self.data[item]
    #    label_idx = self.targets[item]

     #   label = np.zeros(10)
    #    label[label_idx] = 1 # 把num型的label变成10维列表。
    #    label = mx.nd.array(label) 

    #    img = img.as_in_context(self.device)
    #    label = label.as_in_context(self.device)

     #   return img, label

    def __shape_info__(self):
        if self.dataname == "MNIST":
            return 1, 28, 28
        elif self.dataname == "FashionMNIST":
            return 1, 28, 28
        elif self.dataname == "CIFAR10":
            return 3, 32, 32

    #def add_trigger(self, data, targets, trigger_label, portion, mode):
    #    print("## generate " + mode + " Bad Imgs")
    #    new_data = mx.nd.array(data).copy()
    #    new_targets = targets.copy()
    #    perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
    #    channels, width, height = new_data.shape[1:]
    #    for idx in perm: # if image in perm list, add trigger into img and change the label to trigger_label
    #        new_targets[idx] = trigger_label
    #        for c in range(channels):
     #           new_data[idx, c, width-3, height-3] = 255
    #            new_data[idx, c, width-3, height-2] = 255
    #            new_data[idx, c, width-2, height-3] = 255
    #            new_data[idx, c, width-2, height-2] = 255

    #    print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
     #   return new_data, new_targets
