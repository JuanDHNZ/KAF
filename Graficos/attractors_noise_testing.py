# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:39:55 2021

@author: Juan David
"""

import sys
sys.path.append('../')
from datasets import tools
from datasets.ChaoticTimeSeries import GenerateAttractor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import tikzplotlib
import KAF
import numpy as np

#%%
# samples = 500
# attr = "chua"
# seeds = [20, 300, 6, 9087, 12, 57, 34, 1995]


# # 1. Generate data
# dic = {'noise_var':0.05}
# chua = []
# for seed in seeds:
#     x,y,z = GenerateAttractor(samples=samples, attractor=attr, seed=seed, **dic)
#     system = tools.z_scorer(x)
#     chua.append(system)
    
# plt.plot(np.array(chua).T)
# plt.show()


# alphas = [13.6,15.6,17.6]
# betas = [26,28,29]
# labels = []
# # 1. Generate data
# chua = []
# for alpha in alphas:
#     for beta in betas:
#         dic = {'noise_var':0.05, 'alpha': alpha, 'beta': beta}
#         labels.append("alpha = {}; beta = {}".format(alpha,beta))
#         x,y,z = GenerateAttractor(samples=samples, attractor=attr, seed=20, **dic)
#         system = tools.z_scorer(x)
#         chua.append(system)

# plt.figure(figsize=(16,9))
# plt.plot(np.array(chua).T)
# ax = plt.gca()
# ax.legend(labels=labels)
# plt.show()
# # X,y = tools.Embedder(X=system, embedding=5)
# # Xtrain, Xtest, ytrain, ytest = tools.TrainTestSplit(X,y)

#%%
train, test, chua_params = tools.noisy_chua_generator(400)
train = tools.z_scorer(train)
test = tools.z_scorer(test)
Xtrain,ytrain = tools.Embedder(X=train, embedding=5)
Xtest,ytest = tools.Embedder(X=test, embedding=5)

plt.plot(ytrain)
plt.plot(ytest)