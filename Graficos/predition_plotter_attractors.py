# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:14:56 2021

@author: Juan David
"""
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets import tools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import tikzplotlib

import sys
sys.path.append('../')
import KAF

criteria = "CB"
folder = "Predictions/"
N = 100
plot_data = True
save = False
#%% DATA GENERATION - 4.2 AKB nonlinear system

x,y,z = GenerateAttractor(samples=5005, attractor="chua")
system = tools.z_scorer(x)

X,y = tools.Embedder(X=system, embedding=5)
Xtrain, Xtest, ytrain, ytest = tools.TrainTestSplit(X,y)

if plot_data:
    print(Xtrain.shape, ytrain.shape)
    print(Xtest.shape, ytest.shape)
    plt.plot(ytrain, color="red")
    plt.show()
    plt.plot(ytest, color="blue")
    plt.show()
#%%QKLSM PREDICTION

#Parameter selection
path = '../results/4.2/GridSearch_MonteCarlo/mc_QKLMS_4.2_AKB_5003.csv'
grid = pd.read_csv(path)
params = tools.best_params_picker("QKLMS", grid, criteria=criteria)



f = KAF.QKLMS(eta=params['eta'], 
          epsilon=params['epsilon'], 
          sigma=params['sigma'])

ypred_train = f.evaluate(Xtrain,ytrain)
ypred_test = f.predict(Xtest)

print("QKLMS CB = ", len(f.CB))

# plt.plot(ypred_train, label='predict');plt.plot(ytrain, label='target')
# plt.title('train');plt.legend();plt.show()
plt.figure(figsize=(16,9))
plt.ylim([-2,2])
plt.plot(ytest[-N:], lw=2, label='target',alpha=0.6,color="tab:blue")
plt.plot(ypred_test[-N:], lw=2, label='predict', color="tab:red");
plt.title('QKLMS on test set - Codebook size = {}'.format(len(f.CB)))
plt.legend()

textstr = '\n'.join((
    '$\sigma={}$'.format(params['sigma']),
    r'$\epsilon={}$'.format(params['epsilon']),
    r'$\eta={}$'.format(params['eta'])))

ax = plt.gca()
props = dict(boxstyle='round', facecolor='gold', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
if save:
    savename = "QKLMS_4000-1000_pred_{}".format(criteria)
    plt.savefig(folder + '{}.png'.format(savename), dpi=300)
    tikzplotlib.save(folder + 'tex/{}.tex'.format(savename))
    print("Saved")

plt.show()

#%% QKLMS AKB PREDICTION

#Parameter selection
path = '../results/4.2/GridSearch_MonteCarlo/mc_QKLMS_AKB_4.2_AKB_5003.csv'
grid = pd.read_csv(path)
params = tools.best_params_picker("QKLMS_AKB", grid, criteria=criteria)

f = KAF.QKLMS_AKB(eta=params['eta'], 
          epsilon=params['epsilon'], 
          sigma_init=params['sigma_init'],
          mu = params['mu'],
          K = params['K'])

ypred_train = f.evaluate(Xtrain,ytrain)
ypred_test = f.predict(Xtest)

print("AKB CB = ", len(f.CB))

# plt.plot(ypred_train, label='predict');plt.plot(ytrain, label='target')
# plt.title('train');plt.legend();plt.show()
plt.figure(figsize=(16,9))
plt.ylim([-2,2])
plt.plot(ytest[-N:], lw=2, label='target',alpha=0.6,color="tab:blue")
plt.plot(ypred_test[-N:], lw=2, label='predict', color="tab:red");
plt.title('QKLMS-AKB on test set - Codebook size = {}'.format(len(f.CB)))
plt.legend()

textstr = '\n'.join((
    '$\sigma={}$'.format(params['sigma_init']),
    r'$\epsilon={}$'.format(params['epsilon']),
    r'$\eta={}$'.format(params['eta']),
    r'$\mu={}$'.format(params['mu']),
    r'$K={}$'.format(params['K'])))

ax = plt.gca()
props = dict(boxstyle='round', facecolor='gold', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
if save:
    savename ="QKLMS_AKB_4000-1000_pred_{}".format(criteria)
    plt.savefig( folder +'{}.png'.format(savename), dpi=300)
    tikzplotlib.save( folder +'tex/{}.tex'.format(savename))
    print("Saved")
plt.show()

#%% QKLMS AMK PREDICTION

#Parameter selection
path = '../results/4.2/GridSearch_MonteCarlo/mc_QKLMS_AMK_4.2_AKB_5003.csv'
grid = pd.read_csv(path)
params = tools.best_params_picker("QKLMS_AMK", grid, criteria=criteria)

f = KAF.QKLMS_AMK(eta=params['eta'], 
          epsilon=params['epsilon'],
          mu = params['mu'],
          Ka = params['K'])

f.evaluate(Xtrain[:100], ytrain[:100])
ypred_train = f.evaluate(Xtrain,ytrain)
ypred_test = f.predict(Xtest)

print("AMK CB = ", len(f.CB))


# plt.plot(ypred_train, label='predict');plt.plot(ytrain, label='target')
# plt.title('train');plt.legend();plt.show()
plt.figure(figsize=(16,9))
plt.ylim([-2,2])
plt.plot(ytest[-N:], lw=2, label='target',alpha=0.6,color="tab:blue")
plt.plot(ypred_test[-N:], lw=2, label='predict', color="tab:red");
plt.title('QKLMS-AMK on test set - Codebook size = {}'.format(len(f.CB)))
plt.legend()

textstr = '\n'.join((
    r'$\epsilon={}$'.format(params['epsilon']),
    r'$\eta={}$'.format(params['eta']),
    r'$\mu={}$'.format(params['mu']),
    r'$K={}$'.format(params['K'])))

ax = plt.gca()
props = dict(boxstyle='round', facecolor='gold', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
if save:
    savename = "QKLMS_AMK_4000-1000_pred_{}".format(criteria)
    plt.savefig(folder + '{}.png'.format(savename), dpi=300)
    tikzplotlib.save(folder + 'tex/{}.tex'.format(savename))
    print("Saved")

plt.show()