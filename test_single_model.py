#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import warnings
import argparse
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import models
import datasets
import pandas as pd
import torch.nn.functional as F
from scipy.stats import norm

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='mresnet2d', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='CMAPSS', help='the name of the data')
    parser.add_argument('--data_file', type=str, default='FD003', help='the file of the data')
    parser.add_argument('--data_dir', type=str, default='./data/', help='the directory of the data')
    parser.add_argument('--monitor_acc', type=str, default='RUL', help='the performance score')
    parser.add_argument('--mode', type=str, default='GD', help='the performance score')
    parser.add_argument('--result_dir', type=str, default='./results/', help='the directory of the result')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # save, load and display information
    # './checkpoint_dcnn_FD003_MSE/mresnet2d_0521-001732/79-276.0923-best_model.pth'
    #'./checkpoint_dcnn_FD003_QL/mresnet2d_0521-002648/79-248.7003-best_model.pth'
    #'./checkpoint_dcnn_FD003_GD/mresnet2d_0521-003703/79-298.7252-best_model.pth'
    #'./checkpoint_dcnn_FD003_GD/mresnet2d_0521-030304/179-291.0878-best_model.pth'
    # './checkpoint_dcnn_FD001_QL/mresnet2d_0520-235402/79-295.9527-best_model.pth'
    parser.add_argument('--resume', type=str,
                        default='./checkpoint_dcnn_FD003_GD/mresnet2d_0521-003703/79-298.7252-best_model.pth',
                        help='the directory of the resume training model')



    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # Consider the gpu or cpu condition
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")
        device_count = 1

    # Load the datasets
    Dataset = getattr(datasets, args.data_name)
    test_datasets, test_pd = Dataset(args.data_dir, args.data_file).data_preprare(test=True)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    # Define the model
    model = getattr(models, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes,
                                             mode=args.mode)
    # model.fc = torch.nn.Linear(model.fc.in_features, Dataset.num_classes)
    if device_count > 1:
        model = torch.nn.DataParallel(model)

    # Load the best model
    # model.load_state_dict(torch.load(args.resume, map_location=device))
    model.load_state_dict(torch.load(args.resume))
    model.to(device)
    model.eval()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    test_pred = np.zeros((len(test_datasets), 3), dtype=np.float)

    idx = 0
    y_pre = np.zeros((0,))
    y_std = np.zeros((0,))
    y_Q1 = np.zeros((0,))
    y_Q9 = np.zeros((0,))
    for batch_idx, inputs in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            inputs = inputs.to(device)
            # forward
            if args.mode == 'MSE':
                logits = model(inputs)
                logits = torch.squeeze(logits)
            elif args.mode == 'QL':
                logitsQ1, logits, logitsQ9 = model(inputs)
                logitsQ1 = torch.squeeze(logitsQ1)
                logits = torch.squeeze(logits)
                logitsQ9 = torch.squeeze(logitsQ9)
            elif args.mode == 'GD':
                logits, logits_std = model(inputs)
                logits = torch.squeeze(logits)
                logits_std = torch.squeeze(logits_std)

            y_pre = np.concatenate((y_pre, logits.view(-1).cpu().detach().numpy()), axis=0)
            if args.mode == 'QL':
                y_Q1 = np.concatenate((y_Q1, logitsQ1.view(-1).cpu().detach().numpy()), axis=0)
                y_Q9 = np.concatenate((y_Q9, logitsQ9.view(-1).cpu().detach().numpy()), axis=0)
            elif args.mode == 'GD':
                y_std = np.concatenate((y_std, logits_std.view(-1).cpu().detach().numpy()), axis=0)

    if args.mode == 'GD':
        y_Q9 = norm.ppf(0.9, y_pre, y_std)
        y_Q1 = norm.ppf(0.1, y_pre, y_std)
        """
        interval = norm.interval(0.8, y_pre, y_std)
        y_Q1 = interval[0]
        y_Q9 = interval[1]
        """
    elif args.mode == 'MSE':
        y_Q9 = np.zeros_like(y_pre)
        y_Q1 = np.zeros_like(y_pre)

    prepared_results = pd.DataFrame()
    prepared_results['engine_id'] = test_pd['engine_id']
    prepared_results['cycle'] = test_pd['cycle']
    prepared_results['label'] = test_pd['label']
    prepared_results['Q1'] = y_Q1
    prepared_results['Q5'] = y_pre
    prepared_results['Q9'] = y_Q9
    prepared_results.to_pickle(args.result_dir + args.mode+'_'+ args.data_file + '0.pkl')




if __name__ =="__main__":
    main()
