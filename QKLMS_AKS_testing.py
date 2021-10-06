# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:17:22 2021

@author: Juan David

TEST QKLMS AKS
"""
from KAF import QKLMS_AKS
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem 
from datasets.tools import *

# 0. Parameters
n_samples = 1003
testingSystem = "4.2_AKB"

sigma = 100
epsilon= 0.35
eta = 0.05
mu = 0.1

# 1. Data generation
system = GenerateSystem(samples=n_samples, systemType=testingSystem)
system = z_scorer(system)

# 2. Data Preparation
X,y = Embedder(X=system, embedding=2)
Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)

# 3. QKLMS AKS 
f = QKLMS_AKS(eta=eta,epsilon=epsilon, mu=mu, sigma=sigma)
y = f.evaluate(Xtrain,ytrain)
ypred = f.predict(Xtest)

# 4. predictions
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

plt.plot(ytrain, label="target")
plt.plot(y,label="predict")
plt.title("Train - $\sigma = {}$".format(f.sigma))
plt.legend()
plt.show()

plt.plot(ytest, label="target")
plt.plot(ypred,label="predict")
plt.title("Test - $\sigma = {}$".format(f.sigma))
plt.legend()
plt.show()

