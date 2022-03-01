# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 08:30:11 2021

@author: Juan David

Borrador de implementacion de algoritmo de actualizacion del sigma

"""

from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import make_blobs

#%% Data generation

mu1 = 0
sigma1 = 0.5
mu2 = 1 
sigma2 = 0.25

samples = 1000
centers = np.array([mu1,mu2]).reshape(-1,1)
std = np.array([sigma1**0.5,sigma2**0.5]).reshape(-1,1)

X, y = make_blobs(n_samples=samples, centers=[[0,0],[1,1]], cluster_std=std, n_features=2,random_state=0)
print(X.shape,y.shape)

plt.figure(figsize=(10,10))
plt.scatter(X[:,0], X[:,1],c=y, cmap="Set2")
plt.show()

#%% Calculation 

kernel = lambda norm, sigma, P: ((-2*np.pi*sigma**2)**(-P/2)) * np.exp(-norm**2/(2*sigma**2))

sigma = 0.45
P = X.shape[1]
N = len(X)

F = []
K = []
V = []

# 1. Calcular Fk 1 a 1 
for xi in X:
    for xj in X:
        norm = cdist(xi.reshape(-1,1),xj.reshape(-1,1))
        print("norm ", norm.shape)
        Kk = kernel(norm,sigma,P)
        print("kernel ", Kk.shape)
        K.append(Kk)
        Fk = ((-(N*sigma**2)**-1)*Kk*norm).sum(axis=1)
        print("Fk ", Fk.shape)
        F.append(Fk)
        Vk = Kk.mean(axis=1)
        print("Vk ", Vk.shape)
        V.append(Vk)

F_total = np.array(F).reshape(samples,samples,-1)
V_total = np.array(V).reshape(samples,samples,-1)
norm_total = cdist(X,X)
sigma_update = (V_total.sum()*(F_total.T*norm_total).sum())/(F_total**2).sum()

print("F ", F_total.shape)
print("V ", V_total.shape)



