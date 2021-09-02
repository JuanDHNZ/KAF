# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 08:32:45 2021

@author: Juan David
"""


import sys
sys.path.append("../")
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from datasets import tools
from datasets.ChaoticTimeSeries import GenerateAttractor
import KAF
import tikzplotlib

train, test, dic_train, dic_test = tools.noisy_chua_generator(1000, alpha=12, beta=28, noise=True)

plt.plot(test)
plt.show()