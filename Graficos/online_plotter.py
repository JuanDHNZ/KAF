# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 08:52:54 2021

@author: Juan David
"""
import sys
sys.path.append('../')
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets import tools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import tikzplotlib
import KAF
import numpy as np

samples = 1005
attr = "chua"

# 1. Generate data
dic = {'alpha':5, 'beta':1}
x,y,z = GenerateAttractor(samples=samples, attractor=attr, **dic)
system = tools.z_scorer(x)

X,y = tools.Embedder(X=system, embedding=5)
Xtrain, Xtest, ytrain, ytest = tools.TrainTestSplit(X,y)

params = {'eta':0.2,
          'epsilon':0.35,
          'mu':0.05,
          'K': 4}

f = KAF.QKLMS_AMK(eta=params['eta'], 
          epsilon=params['epsilon'], 
          mu=params['mu'],
          Ka=params['K'], 
          A_init="diag", 
          sigma=0.6)

def cb_print_size(f):
    for n,cb in enumerate(f.CB):
        print("cb {} shape = {}".format(n,cb.shape))


ypred_train = []
plt.figure(figsize=(10,10))
for xt, yt in zip(Xtrain,ytrain):
    ypred_train.append(f.evaluate(xt.reshape(1,-1),yt.reshape(-1,1)))
    ypred_test = f.predict(Xtest)
    plt.clf()
    plt.xlabel('QKLMS train&test update - Codebook size = {}'.format(len(f.CB)))
    print("Cb = ", len(f.CB))
    plt.subplot(211)
    plt.plot(ytrain, lw=2, label='target',alpha=0.6,color="tab:blue")
    plt.plot(ypred_train, lw=2, label='predict', color="tab:red")
    plt.legend()
    plt.subplot(212)
    plt.plot(ytest, lw=2, label='target',alpha=0.6,color="tab:blue")
    plt.plot(ypred_test, lw=2, label='predict', color="tab:red")
    plt.legend()
    plt.show(block=False)
    plt.pause(0.05)

    # textstr = '\n'.join((
    #     '$\sigma={}$'.format(params['sigma']),
    #     r'$\epsilon={}$'.format(params['epsilon']),
    #     r'$\eta={}$'.format(params['eta'])))
    
    # props = dict(boxstyle='round', facecolor='gold', alpha=0.5)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
    #         verticalalignment='top', bbox=props)
    

