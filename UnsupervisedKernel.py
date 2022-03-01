# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:32:01 2021

@author: Juan David
"""
import numpy as np
from KAF import QKLMS_MIPV
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem 
from datasets import tools

#%%  4.2 AKB NONLINEAR SYSTEM
samples = 2200
embedding = 2

system = GenerateSystem(samples=samples+embedding, systemType="4.2_AKB")
system = tools.z_scorer(system)

S = tools.mc_sampler(system, samples, 1, embedding=embedding)
St = S[0]
train_portion=2000/2200
train_size = int(len(St)*train_portion)
Xtrain,ytrain = St[:train_size,:-1],St[:train_size,-1]
Xtest,ytest = St[train_size:,:-1],St[train_size:,-1]         
print(Xtrain.shape,ytrain.shape)
print(Xtest.shape,ytest.shape)

params = {'eta':0.1,
          'epsilon':0.1,
          'sigma':0.707}

f = QKLMS_MIPV(eta=params["eta"],sigma=params["sigma"], epsilon=params["epsilon"])

ypred_train = f.evaluate(Xtrain,ytrain.reshape(-1,1))

        
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

plt.figure(figsize=(8,6))
plt.title("$\sigma_F = {}$".format(f.sigma))
plt.ylabel("var_px")
plt.xlabel("$\sigma$")
plt.xscale("log")
for curve in f.var_px:
    plt.plot(f.sigma_grid,curve)

plt.figure(figsize=(8,6))
plt.title("$\sigma_F = {}$".format(f.sigma))
plt.ylabel("$\sigma$")
plt.xlabel("iterations")
plt.plot(f.sigma_n)   


