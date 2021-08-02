# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:36:47 2021

@author: USUARIO
"""

from testing import CB_visualizer
ts = '4.2_AKB'
samples = 200
# p_file = "results/4.2/New/mc_QKLMS_AMK_4.2_AKB_5003.csv"
p_file = "results/4.2/AMK_fixed/mc_QKLMS_AMK_4.2_AKB_5003.csv"
# model = CB_visualizer(testingSystem=ts, n_samples=samples, params_file = p_file)

from datasets.TestingSystems import GenerateSystem
from datasets import tools

system = GenerateSystem(samples=1000, systemType="4.2_AKB")
system = tools.z_scorer(system)


X,y = tools.Embedder(X=system, embedding=2)
Xtrain, Xtest, ytrain, ytest = tools.TrainTestSplit(X,y)

from KAF import QKLMS_AMK as amk

f = amk(eta=0.2, epsilon=1, mu=0.4, Ka=8)

ypred_train = f.evaluate(Xtrain,ytrain)
ypred_test = f.predict(Xtest)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

plt.plot(ypred_train, label='predict')
plt.plot(ytrain, label='target')
plt.title('train')
plt.legend()
plt.show()

plt.plot(ypred_test, label='predict')
plt.plot(ytest, label='target')
plt.title('test')
plt.legend()
plt.show()