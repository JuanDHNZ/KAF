# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 11:06:02 2021

@author: USUARIO
"""
#visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# data generation
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem
from datasets.tools import Embedder, TrainTestSplit

system = GenerateSystem(samples=5003, systemType="4.2_AKB")
X,y = Embedder(X=system, embedding=2)
Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)










