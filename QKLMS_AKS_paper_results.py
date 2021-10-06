# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:22:11 2021

@author: Juan David
"""

from KAF import QKLMS_AKS
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem 
from datasets.tools import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# 0. Parameters
n_samples = 5001
testingSystem = "4.1_AKB"

sigma = 1
epsilon=0
eta = 0.5
mu = 0.05

# 1. Data generation
u,d = GenerateSystem(samples=n_samples, systemType=testingSystem)
# u = z_scorer(u)
# d = z_scorer(d)

plt.plot(u)
plt.show()


# 2. Data Preparation
# u,_ = Embedder(X=u, embedding=3)
# Xtrain, Xtest, ytrain, ytest = TrainTestSplit(u,d.reshape(-1,1))



from KAF import QKLMS
f = QKLMS(eta=eta, epsilon=epsilon, sigma=sigma)
yt = f.evaluate(u.reshape(-1,1),d.reshape(-1,1))
ypred = f.predict(u.reshape(-1,1))

plt.plot(d, label="target")
plt.plot(ypred,label="predict")
plt.title("Test - $\sigma = {}$".format(f.sigma))
plt.legend()
plt.show()

import models
f2 = models.QKLMS(Xtrain[0], ytrain[0], epsilon, eta, sigma)
for X,y in zip(Xtrain[1:], ytrain[1:]):
    f2.update(X,y)
    
plt.plot(yt,marker="*", markevery=20,label="predict  $\sigma$ = {}".format(f.sigma))
plt.plot(d, label="target")
# plt.plot(f2.pred, label="predict Sathujoda $\sigma$ = {}".format(f2.sigma))
# plt.title("Train - $\sigma = {}$".format(f.sigma))
plt.legend()
plt.show()




plt.plot(Xtrain)
plt.show()
# 3. QKLMS AKS 
f = QKLMS_AKS(eta=eta,epsilon=epsilon, mu=mu, sigma=sigma)
y = f.evaluate(Xtrain,ytrain)
ypred = f.predict(Xtest)

plt.plot(f.sigma_n)
plt.ylabel("$\sigma_i$")
plt.xlabel("iterations")
plt.title("$\sigma$ evolution")
plt.show()
