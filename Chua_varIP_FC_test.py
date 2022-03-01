# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 00:14:26 2021

@author: Juan David
"""


from KAF import QKLMS_varIP_FC
from datasets.tools import z_scorer,mc_sampler
from datasets.TestingSystems import GenerateSystem
from datasets.ChaoticTimeSeries import GenerateAttractor
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def varIP(X, sigma):
  from scipy.spatial.distance import cdist
  kernel_rbf = lambda x,sigma: (2*np.pi*sigma**2)**(-X.shape[1]/2)*np.exp((-cdist(x,x)**2)/(2*sigma**2))
  var_px = lambda k: np.var(np.mean(k,axis=0))
  return -np.log(var_px(kernel_rbf(X,sigma)))

def sigma_curve_plotter(CB,FC):
  Cb = np.array(CB)[:FC]   
  f = lambda sigma: varIP(Cb,sigma)
  res = minimize_scalar(f, method='Bounded', bounds=[1e-5,1e2])

  sigmas = np.logspace(-5,3,100)
  var_ip = [f(x) for x in sigmas]

  plt.plot(sigmas, var_ip)
  plt.scatter(res.x, f(res.x), color="red", marker="X")
  plt.xlabel("$\sigma$")
  plt.ylabel("-log(var_ip)")
  plt.xscale("log")
  plt.title("Chua | Fixed CB = {}".format(FC))
  plt.show()
  return

#%% 1. Data generation

embedding=5
samples = 4200
x_at, y_at, z_at = GenerateAttractor(samples=samples+embedding, attractor='chua')
system = z_scorer(x_at)
# system = x_at
plt.plot(x_at)
plt.plot()

system_emb = mc_sampler(system, samples , 1, embedding=embedding)
X,y = system_emb[0,:,:-1],system_emb[0,:,-1]

train_portion=4000/4200
train_size = int(samples*train_portion)
Xtrain,ytrain = system_emb[0,:train_size,:-1],system_emb[0,:train_size,-1].reshape(-1,1)
Xtest,ytest = system_emb[0,train_size:,:-1],system_emb[0,train_size:,-1].reshape(-1,1)
print("train ", Xtrain.shape,ytrain.shape)
print("test ", Xtest.shape,ytest.shape)

from KAF import QKLMS_varIP_FC
filt = QKLMS_varIP_FC(eta=0.1,
                   epsilon=0.3,
                   sigma=1,
                   FC=5, 
                   bounds=[1e-5,1e2])
ypred_train = filt.evaluate(Xtrain,ytrain)

sigma_n = np.array(filt.sigma_n)
sigma_max = np.where(sigma_n >= 90)[0][0]

#%% 2. Filter 
from KAF import QKLMS_varIP_FC
filt = QKLMS_varIP_FC(eta=0.1,
                   epsilon=0.7,
                   sigma=1,
                   FC=5, 
                   bounds=[1e-5,1e1])

delta = 0
# ypred_train = filt.evaluate(Xtrain[:sigma_max+delta],ytrain[:sigma_max+delta])
ypred_train = filt.evaluate(Xtrain,ytrain)
ypred_test = filt.predict(Xtest)

CB_rec = np.array(filt.CB_record)

# Sigma plot
plt.figure(figsize=(12,8))
plt.plot(filt.sigma_n)
plt.title("$\sigma_F = {}$".format(filt.sigma))
plt.ylabel("$\sigma$")
plt.xlabel("iterations")
# plt.yscale("log")


# plt.xscale("log")
plt.plot(Xtrain)
plt.scatter(CB_rec[:,0], CB_rec[:,1], color='red', marker="X")
plt.show()

#%% prediction

plt.figure(figsize=(14,8))
plt.plot(ytest, label='target')
plt.plot(ypred_test, label='predict')
plt.legend()
plt.show()

#%%
codebook = filt.CB
sigma_curve_plotter(codebook,len(codebook))

#%% Ape
from datasets.tools import MAPE
ape_train = MAPE(ytrain,ypred_train)
ape_test = MAPE(ytest,ypred_test)

from datasets.tools import MSE
mse_train = MSE(ytrain,ypred_train)
mse_test = MSE(ytest,ypred_test)

from datasets.tools import MAE
mse_train = MAE(ytrain,ypred_train)
mse_test = MAE(ytest,ypred_test)








