#!/usr/bin/python
# -*- coding:utf-8 -*-


import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def DrawEngine(df, number, file_name):
    label = df.loc[df.engine_id == number, 'label']
    cycle = df.loc[df.engine_id == number, 'cycle']
    Q1 = df.loc[df.engine_id == number, 'Q1']
    Q5 = df.loc[df.engine_id == number, 'Q5']
    Q9 = df.loc[df.engine_id == number, 'Q9']


    plt.figure(figsize=(8, 6))
    line1, = plt.plot(cycle, label, color='black', lw=1.5, ls='-')
    line2, = plt.plot(cycle, Q5, color='red', lw=0.5, marker='o', ms=2)
    line3, = plt.plot(cycle, Q1, color='green', lw=0.5, ls='--')
    line4, = plt.plot(cycle, Q9, color='green', lw=0.5, ls='--')


    plt.fill_between(cycle, Q1, Q9, color=(229 / 256, 204 / 256, 249 / 256), alpha=0.8)


    Fontsize = 20
    plt.xticks(rotation=20, fontsize=Fontsize)
    plt.yticks(fontsize=Fontsize)
    plt.xlabel('Cycle', fontsize=Fontsize)
    plt.ylabel('RUL', fontsize=Fontsize)


    plt.legend([line1, line2], ["True RUL", "Predicted RUL"], loc='lower left', fontsize=Fontsize)


    plt.savefig(file_name + '.png', dpi=500, bbox_inches='tight')
    plt.show()


def DrawEngine_Point(df, number, file_name):
    label = df.loc[df.engine_id == number, 'label']
    cycle = df.loc[df.engine_id == number, 'cycle']
    Q5 = df.loc[df.engine_id == number, 'Q5']



    plt.figure(figsize=(8, 6))
    line1, = plt.plot(cycle, label, color='black', lw=1.5, ls='-')
    line2, = plt.plot(cycle, Q5, color='red', lw=0.5, marker='o', ms=2)


    Fontsize = 20
    plt.xticks(rotation=20, fontsize=Fontsize)
    plt.yticks(fontsize=Fontsize)
    plt.xlabel('Cycle', fontsize=Fontsize)
    plt.ylabel('RUL', fontsize=Fontsize)


    plt.legend([line1, line2], ["True RUL", "Predicted RUL"], loc='upper right', fontsize=Fontsize)


    plt.savefig(file_name + '.png', dpi=500, bbox_inches='tight')
    plt.show()



file_name = 'QL_FD003'
result_dir = './results/' + file_name + '.pkl'
test_pd = pd.read_pickle(result_dir)

DrawEngine(test_pd, 24, file_name)
#DrawEngine_Point(test_pd, 24, file_name)