# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 11:06:02 2021

@author: USUARIO
"""
#visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pandas as pd
def get_kaf_params(filename):
    param_path = "results/4.2/"
    params = pd.read_csv(param_path + filename)
    best_params = params[params.tradeOff_dist == params.tradeOff_dist.min()].iloc[0]
    return best_params

def MSE(y_true, y_pred):
    import numpy as np
    return np.mean((y_true-np.array(y_pred).reshape(-1,1))**2/y_true**2)

def filter_evaluation(filt):
    results = {"TMSE":[],"CB":[]}
    from tqdm import tqdm
    for xi, yi in tqdm(zip(Xtrain,ytrain)):
        try:
            #train and predict
            filt.evaluate(xi,yi)
            y_pred = filt.predict(Xtest)
            #scoring
            results["CB"].append(len(filt.CB))
            results["TMSE"].append(MSE(ytest,y_pred))
        except:
            results["CB"].append(0)
            results["TMSE"].append(0)
    return results

#%% 1. data generation
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem
from datasets.tools import Embedder, TrainTestSplit

system = GenerateSystem(samples=5003, systemType="4.2_AKB")
X,y = Embedder(X=system, embedding=2)
Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)

#%% 2. Parameter selection 
params_qklms = get_kaf_params("QKLMS_4.2_5000.csv")
params_akb = get_kaf_params("QKLMS_AKB_4.2_5000.csv")
params_amk = get_kaf_params("QKLMS_AMK_4.2_5000.csv")

#%% 3. QKLMS results
from KAF import QKLMS
filt = QKLMS(eta=params_qklms.eta,
             epsilon=params_qklms.epsilon,
             sigma=params_qklms.sigma)

print("QKLMS TMSE...\n")
results = filter_evaluation(filt)
savename = "results/4.2/TMSE_QKLMS_4.2.csv"
pd.DataFrame(data=results).to_csv(savename)
print("QKLMS saved in {}\n\n".format(savename))

#%% 3. QKLMS_AKB results
from KAF import QKLMS_AKB
filt = QKLMS_AKB(eta=params_akb.eta,
             epsilon=params_akb.epsilon,
             sigma_init=params_akb.sigma_init,
             mu=params_akb.mu,
             K=int(params_akb.K))

print("QKLMS AKB TMSE...\n")
results = filter_evaluation(filt)
savename = "results/4.2/TMSE_QKLMS_AKB_4.2.csv"
pd.DataFrame(data=results).to_csv(savename)
print("QKLMS AKB saved in {}\n\n".format(savename))

#%% 3. QKLMS_AMK results
from KAF import QKLMS_AMK
filt = QKLMS_AKB(eta=params_amk.eta,
             epsilon=params_amk.epsilon,
             mu=params_amk.mu,
             K=int(params_amk.K))

print("QKLMS AMK TMSE...\n")
results = filter_evaluation(filt)
savename = "results/4.2/TMSE_QKLMS_AMK_4.2.csv"
pd.DataFrame(data=results).to_csv(savename)
print("QKLMS AMK saved in {}\n\n".format(savename))







