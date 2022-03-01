# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:22:57 2021

@author: Juan David
"""

from datasets import tools
n_samples = 9000
x,y,z = tools.noisy_chua_splited(n_samples-1, alpha=15.6, beta=28)
x = x.reshape(-1,3000)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

plt.figure(figsize=(16,9))
plt.plot(tools.z_scorer(x[0]))
plt.show()
plt.plot(tools.z_scorer(x[1]))
plt.show()
plt.plot(tools.z_scorer(x[2]))
plt.show()
