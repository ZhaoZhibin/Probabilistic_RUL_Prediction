#!/usr/bin/python
# -*- coding:utf-8 -*-
from datasets.CMAPSS_Datasets import dataset
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
from itertools import chain
from glob import glob
from tqdm import tqdm
from datasets.sequence_aug import *

normlizetype = 'none'
sig_resample_len = 512
data_transforms = {
    'train': Compose([
        #Reshape(),
        #DownSample(sig_resample_len),
        Normalize(normlizetype),
        # RandomAddGaussian(),
        # RandomScale(0.01),
        # RandomAmplify(),
        # Randomverflip(),
        # Randomshift(),
        # RandomStretch(0.01),
        # RandomCrop(),
        Retype()
    ]),
    'val': Compose([
        #Reshape(),
        #DownSample(sig_resample_len),
        Normalize(normlizetype),
        Retype()
    ]),
    'test': Compose([
        #Reshape(),
        #DownSample(sig_resample_len),
        Normalize(normlizetype),
        Retype()
    ])
}




class CMAPSS(object):
    num_classes = 1
    inputchannel = 1


    def __init__(self, data_dir, data_file):
        self.data_dir = data_dir
        self.data_file = data_file


    def data_preprare(self, test=False):
        if test:
            test_pd = pd.read_pickle(self.data_dir + 'test_all_' + self.data_file + '.pkl')
            test_dataset = dataset(anno_pd=test_pd, data_dir=self.data_dir, test=True, transform=data_transforms['test'])
            return test_dataset, test_pd
        else:
            train_pd = pd.read_pickle(self.data_dir + 'train_' + self.data_file + '.pkl')
            val_pd = pd.read_pickle(self.data_dir + 'test_' + self.data_file + '.pkl')

            train_dataset = dataset(anno_pd=train_pd, data_dir=self.data_dir, transform=data_transforms['train'])
            val_dataset = dataset(anno_pd=val_pd, data_dir=self.data_dir, transform=data_transforms['val'])
            return train_dataset, val_dataset




