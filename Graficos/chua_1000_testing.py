# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:39:25 2021

@author: Juan David

CHUA WITH RANDOM NOISE TESTING

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


samples = 4005
emb = 5
criteria = 'MAPE'
save = False
folder = "Chua/Predictions_4000/"

#%% 1. Data generation
alp = 12
bt = 28

train, test, dic_train, dic_test = tools.noisy_chua_generator(samples, alpha=alp, beta=bt)

train = tools.z_scorer(train)
test = tools.z_scorer(test)

Xtrain,ytrain = tools.Embedder(X=train, embedding=5)
Xtest,ytest = tools.Embedder(X=test, embedding=5)

plt.figure(figsize=(16,9))
plt.ylim([-10,10])
plt.title('alpha = {} - beta = {}'.format(alp,bt))
plt.plot(ytrain, label='ytrain', lw=1)
plt.plot(ytest, label='ytest', lw=1)
plt.legend()
if save:
    plt.savefig(folder + '{}.png'.format("Signal"), dpi=300)
plt.show()


#%% QKLMS train & test 

#Parameter selection
path = '../results/Chua/split_3/mc_QKLMS_chua_4005.csv'
grid = pd.read_csv(path)
params = tools.best_params_picker("QKLMS", grid, criteria=criteria)

f = KAF.QKLMS(eta=params['eta'], 
          epsilon=params['epsilon'], 
          sigma=params['sigma'])

ypred_train = f.evaluate(Xtrain,ytrain)
ypred_test = f.predict(Xtrain)

print("QKLMS CB = ", len(f.CB))


plt.figure(figsize=(16,9))
plt.ylim([-8,8])
# plt.plot(ytest, lw=2, label='target',alpha=0.6,color="tab:blue")
# plt.plot(ypred_test, lw=2, label='predict', color="tab:red");
# plt.title('QKLMS prediction,  min {} | CB= {}'.format(criteria,len(f.CB)))
plt.title('QKLMS trainning,  min {} | CB= {}'.format(criteria,len(f.CB)))
plt.plot(ytrain, lw=2, label='target',alpha=0.6,color="tab:blue")
plt.plot(ypred_train, lw=2, label='predict', color="tab:red")

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
    savename = "QKLMS_1000-1000_pred_{}".format(criteria)
    plt.savefig(folder + '{}.png'.format(savename), dpi=300)
    tikzplotlib.save(folder + 'tex/{}.tex'.format(savename))
    print("Saved")
plt.show()

#%% QKLMS APE Grafico
ape = tools.APE(ytrain, ypred_train)

plt.figure(figsize=(16,9))
plt.yscale("log")
# plt.xscale("log")
plt.ylim((1e-5,1e3))
plt.title('QKLMS MAPE,  min {} | CB= {}'.format(criteria,len(f.CB)))
plt.plot(ape, lw=2, color="tab:red")
plt.show()

mape = tools.MAPE(ytrain, ypred_train)
print("QKLMS MAPE : ",mape)

#%% QKLMS AKB train & test 

#Parameter selection
path = '../results/Chua/split_3/mc_QKLMS_AKB_chua_4005.csv'
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


plt.figure(figsize=(16,9))
plt.ylim([-8,8])
# plt.plot(ytest, lw=2, label='target',alpha=0.6,color="tab:blue")
# plt.plot(ypred_test, lw=2, label='predict', color="tab:red");
# plt.title('QKLMS AKB prediction - CB = {}'.format(len(f.CB)))
plt.title('QKLMS AKB trainning - CB = {}'.format(len(f.CB)))
plt.plot(ytrain, lw=2, label='target',alpha=0.6,color="tab:blue")
plt.plot(ypred_train, lw=2, label='predict', color="tab:red")

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
    savename ="QKLMS_AKB_1000-1000_pred_{}".format(criteria)
    plt.savefig( folder +'{}.png'.format(savename), dpi=300)
    tikzplotlib.save(folder +'tex/{}.tex'.format(savename))
    print("Saved")
plt.show()

#%% QKLMS AKB APE Grafico
ape = tools.APE(ytrain, np.array(ypred_train))

plt.figure(figsize=(16,9))
plt.yscale("log")
# plt.xscale("log")
plt.ylim((1e-5,1e3))
plt.title('QKLMS AKB MAPE,  min {} | CB= {}'.format(criteria,len(f.CB)))
plt.plot(ape, lw=2, color="tab:red")
plt.show()
mape = tools.MAPE(ytrain, np.array(ypred_train))
print("QKLMS AKB MAPE : ",mape)
#%% QKLMS AMK train & test 

#Parameter selection
path = '../results/Chua/split_3/mc_QKLMS_AMK_chua_4005.csv'
grid = pd.read_csv(path)

# path = '../results/Chua/split_2/new_params2_mc_QKLMS_AMK_chua_4005.csv'
# grid2 = pd.read_csv(path)

# total = pd.concat([grid,grid2])
# total.to_csv('../results/Chua/split_2/concat2_mc_QKLMS_AMK_chua_4005.csv')

params = tools.best_params_picker("QKLMS_AMK", grid, criteria=criteria)

# params["K"] = 32
# params["mu"] = 0.05


f = KAF.QKLMS_AMK(eta=params['eta'], 
          epsilon=params['epsilon'],
          mu = params['mu'],
          Ka = params['K'])

f.evaluate(Xtrain[:100], ytrain[:100])
ypred_train = f.evaluate(Xtrain,ytrain)
ypred_test = f.predict(Xtest)

print("AMK CB = ", len(f.CB))

plt.figure(figsize=(16,9))
plt.ylim([-8,8])
# plt.plot(ytest, lw=2, label='target',alpha=0.6,color="tab:blue")
# plt.plot(ypred_test, lw=2, label='predict', color="tab:red")
# plt.title('QKLMS AMK prediction | CB = {} '.format(len(f.CB)))
plt.plot(ytrain, lw=2, label='target',alpha=0.6,color="tab:blue")
plt.plot(ypred_train, lw=2, label='predict', color="tab:red")
plt.title('QKLMS AMK trainning - CB = {}'.format(len(f.CB)))
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
    savename = "QKLMS_AMK_1000-1000_pred_{}".format(criteria)
    plt.savefig(folder + '{}.png'.format(savename), dpi=300)
    tikzplotlib.save(folder + 'tex/{}.tex'.format(savename))
    print("Saved")

plt.show()

#%% QKLMS AMK APE Grafico
ape = tools.APE(ytrain, ypred_train)

plt.figure(figsize=(16,9))
plt.yscale("log")
# plt.xscale("log")
plt.ylim((1e-5,1e3))
plt.title('QKLMS AMK MAPE,  min {} | CB= {}'.format(criteria,len(f.CB)))
plt.plot(ape, lw=2, color="tab:red")
plt.show()

mape = tools.MAPE(ytrain, ypred_train)
print("QKLMS AMK MAPE : ",mape)
#%% QKLMS AMK APE Grafico 
ape = tools.MAPE(ytrain, ypred_train)

plt.figure(figsize=(16,9))
plt.yscale("log")
# plt.xscale("log")
plt.ylim((1e-5,1e3))
plt.title('QKLMS AMK MAPE,  min {} | CB= {}'.format(criteria,len(f.CB)))
plt.plot(ape, lw=2, color="tab:red")
plt.show()