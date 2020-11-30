#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import numpy as np
import math
import torch
from torch import nn
from torch import optim
import pandas as pd
from scipy.stats import norm

import models
import datasets
from utils.save import Save_Tool
from utils.freeze import set_freeze_by_id
from utils.metrics import *
from loss.loss_factory import *


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}

        self.datasets['train'], self.datasets['val'] = Dataset(args.data_dir, args.data_file).data_preprare()

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        # Define the model
        self.model = getattr(models, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes,
                                                      mode=args.mode)
        if args.layer_num_last != 0:
            set_freeze_by_id(self.model, args.layer_num_last)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, args.steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")


        # Define the monitoring accuracy
        if args.monitor_acc == 'RUL':
            self.cal_acc = RUL_Score
        else:
            raise Exception("monitor_acc is not implement")


        # Load the checkpoint
        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        # Invert the model
        self.model.to(self.device)

        #  and define the loss
        self.criterionMSE = nn.MSELoss()
        self.criterionQ1 = QuantileLoss
        self.criterionQ5 = QuantileLoss
        self.criterionQ9 = QuantileLoss
        self.criterionMLE = MLEGLoss




    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 1000
        batch_count = 0
        batch_loss = 0.0
        batch_mse = 0
        batch_phm_score = 0
        step_start = time.time()
        acc_df = pd.DataFrame(columns=('epoch', 'rmse', 'rmlse', 'mae', 'r2', 'sf', 'Q1', 'Q9'))

        save_list = Save_Tool(max_num=args.max_model_num)

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))



            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_mse = 0
                epoch_phm_score = 0
                epoch_loss = 0.0
                y_labels = np.zeros((0,))
                y_pre = np.zeros((0,))
                y_std = np.zeros((0,))
                y_Q1 = np.zeros((0,))
                y_Q9 = np.zeros((0,))



                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                #
                for batch_idx, (inputs, labels, cycles) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    #cycles = cycles.to(self.device)


                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        if args.mode == 'MSE':
                            logits = self.model(inputs)
                            logits = torch.squeeze(logits)
                            #logits = cycles / (torch.squeeze(logits) + 1e-4) - cycles
                            loss = self.criterionMSE(logits, labels)
                        elif args.mode == 'QL':
                            logitsQ1, logits, logitsQ9 = self.model(inputs)
                            logitsQ1 = torch.squeeze(logitsQ1)
                            logits = torch.squeeze(logits)
                            logitsQ9 = torch.squeeze(logitsQ9)
                            #logitsQ1 = cycles / (torch.squeeze(logitsQ1) + 1e-4) - cycles
                            #logits = cycles / (torch.squeeze(logits) + 1e-4) - cycles
                            #logitsQ9 = cycles / (torch.squeeze(logitsQ9) + 1e-4) - cycles
                            lossQ1 = self.criterionQ1(logitsQ1, labels, quantile_level=0.1)
                            lossQ5 = self.criterionQ5(logits, labels, quantile_level=0.5)
                            lossQ9 = self.criterionQ9(logitsQ9, labels, quantile_level=0.9)
                            loss = lossQ1 + lossQ5 + lossQ9
                        elif args.mode == 'GD':
                            logits, logits_std = self.model(inputs)
                            logits = torch.squeeze(logits)
                            #logits = cycles / (torch.squeeze(logits) + 1e-4) - cycles
                            logits_std = torch.squeeze(logits_std)
                            # print(logits.dtype, labels.dtype)
                            loss = self.criterionMLE(logits, logits_std, labels)
                        mse, phm_score = self.cal_acc(labels, logits)

                        if phase == 'val':
                            y_labels = np.concatenate((y_labels, labels.view(-1).cpu().detach().numpy()), axis=0)
                            y_pre = np.concatenate((y_pre, logits.view(-1).cpu().detach().numpy()), axis=0)
                            if args.mode == 'QL':
                                y_Q1 = np.concatenate((y_Q1, logitsQ1.view(-1).cpu().detach().numpy()), axis=0)
                                y_Q9 = np.concatenate((y_Q9, logitsQ9.view(-1).cpu().detach().numpy()), axis=0)
                            elif args.mode == 'GD':
                                y_std = np.concatenate((y_std, logits_std.view(-1).cpu().detach().numpy()), axis=0)







                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_mse += mse * inputs.size(0)
                        epoch_phm_score += phm_score

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_mse += mse * inputs.size(0)
                            batch_phm_score += phm_score
                            batch_count += inputs.size(0)
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_mse = batch_mse / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0*batch_count/train_time
                                logging.info('Epoch: {} [{}/{}], {} Loss: {:.4f} MSE: {:.4f}' 
                                             'RMSE: {:.4f} PHM: {:.4f}'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx*len(inputs), len(self.dataloaders[phase].dataset), phase,
                                    batch_loss, batch_mse, math.sqrt(batch_mse), batch_phm_score,
                                    sample_per_sec, batch_time
                                ))
                                batch_mse = 0
                                batch_phm_score = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1


                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_mse = epoch_mse / len(self.dataloaders[phase].dataset)

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-MSE: {:.4f}-RMSE: {:.4f}-PHM: {:.4f} Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_mse, math.sqrt(epoch_mse), epoch_phm_score,
                    time.time()-epoch_start
                ))

                # save the model
                if phase == 'val':
                    if epoch >= args.max_epoch-10:
                        if args.mode == 'GD':
                            y_Q9 = norm.ppf(0.9, y_pre, y_std)
                            y_Q1 = norm.ppf(0.1, y_pre, y_std)
                        elif args.mode == 'MSE':
                            y_Q9 = np.zeros_like(y_labels)
                            y_Q1 = np.zeros_like(y_labels)
                        quantile_level = 0.9
                        index = (y_Q9 <= y_labels)
                        acc_Q9 = np.mean(quantile_level*(y_labels-y_Q9)*index + (1-quantile_level)*(y_Q9-y_labels)*(1-index))
                        quantile_level = 0.1
                        index = (y_Q1 <= y_labels)
                        acc_Q1 = np.mean(quantile_level*(y_labels-y_Q1)*index + (1-quantile_level)*(y_Q1-y_labels)*(1-index))

                        acc_df = acc_df.append(
                            pd.DataFrame({'epoch': [epoch],
                                          'rmse': [math.sqrt(epoch_mse)],
                                          'rmlse': [math.sqrt(np.mean(np.square(np.log(y_labels+1)-np.log(y_pre+1))))],
                                          'mae': [np.mean(np.abs(y_labels-y_pre))],
                                         'r2': [1 - np.sum(np.square(y_labels - y_pre) / np.sum(np.square(y_labels - np.mean(y_labels))))],
                                         'sf': [epoch_phm_score],
                                          'Q1': [acc_Q1],
                                          'Q9': [acc_Q9]}), ignore_index=True)

                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)
                    # save the best model according to the val accuracy
                    if epoch_phm_score < best_acc or epoch == args.max_epoch-1:
                        best_acc = epoch_phm_score
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, best_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

                    if epoch == args.max_epoch-1:
                        acc_df.to_csv('Results_'+args.mode+'_'+args.data_file+'.csv', sep=",", index=False)
                        acc_means = acc_df.mean()
                        logging.info("rmse {:.4f}, rmlse {:.4f}, mae {:.4f}, r2 {:.4f}, sf {:.4f}, Q1 {:.4f}, Q9 {:.4f}".format(acc_means['rmse'],
                                                                                                                                acc_means['rmlse'],
                                                                                                                                acc_means['mae'],
                                                                                                                                acc_means['r2'],
                                                                                                                                acc_means['sf'],
                                                                                                                                acc_means['Q1'],
                                                                                                                                acc_means['Q9']))



            if self.lr_scheduler is not None:
                self.lr_scheduler.step()















