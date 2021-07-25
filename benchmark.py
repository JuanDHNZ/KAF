# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 11:06:02 2021

@author: USUARIO
"""
#visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 1. data generation
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem
from datasets.tools import Embedder, TrainTestSplit

system = GenerateSystem(samples=5003, systemType="4.2_AKB")
X,y = Embedder(X=system, embedding=2)
Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)

# 2. Parameter selection 
import pandas as pd
def get_kaf_params(filename):
    param_path = "results/4.2/"
    params = pd.read_csv(param_path + filename)
    best_params = params[params.tradeOff_dist == params.tradeOff_dist.min()].iloc[0]
    return best_params

params_qklms = get_kaf_params("QKLMS_4.2_5000.csv")
params_akb = get_kaf_params("QKLMS_AKB_4.2_5000.csv")
params_amk = get_kaf_params("QKLMS_AMK_4.2_5000.csv")

# 3. QKLMS results
from KAF import QKLMS
filt = QKLMS(eta=params_qklms.eta,
             epsilon=params_qklms.epsilon,
             sigma=params_qklms.sigma)

def MSE(y_true, y_pred):
    return np.mean((y_true-np.array(y_pred).reshape(-1,1))**2/y_true**2)

print("QKLMS TMSE...")
TMSE = []
CB = []
for xi, yi in tqdm(zip(Xtrain,ytrain)):
    #train and predict
    filt.evaluate(xi,yi)
    y_pred = filt.predict(Xtest)
    #scoring
    CB.append(len(filt.CB))
    TMSE.append(MSE(ytest,y_pred))

    














