#!/usr/bin/python
# -*- coding:utf-8 -*-


from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import torch

def load_data(case):
    df = np.loadtxt(case)
    return df


class dataset(Dataset):

    def __init__(self, anno_pd, data_dir, test=False, transform=None, loader=load_data):
        self.test = test
        if self.test:
            self.data = anno_pd['data'].tolist()
        else:
            self.data = anno_pd['data'].tolist()
            self.label = anno_pd['label'].tolist()
            self.cycle = anno_pd['cycle'].tolist()
        if transform is None:
            self.transforms = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        else:
            self.transforms = transform
        self.data_dir = data_dir
        self.loader = loader


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.test:
            img = self.data[item]
            #img = img.transpose()
            img = img[np.newaxis, :, :]
            img = self.transforms(img)
            return img
        else:
            img = self.data[item]
            #img = img.transpose()
            img = img[np.newaxis, :, :]  # img[np.newaxis, :, :], img[:, :, np.newaxis]
            label = self.label[item]
            label = np.array(label, dtype='float32')
            cycle = self.cycle[item]
            cycle = np.array(cycle, dtype='float32')
            img = self.transforms(img)
            return img, label, cycle

if __name__ == '__main__':
    """
    img = cv2.imread('../ODIR-5K_training/0_left.jpg')
    #cv2.flip(img, 1, dst=None)
    cv2.namedWindow("resized", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("resized", 640, 480)
    cv2.imshow('resized', img)
    cv2.waitKey(5)
    # cv2.destoryAllWindows()
    """
