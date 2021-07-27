# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:10:01 2021

@author: USUARIO
"""
import pandas as pd  
import numpy as np
from tqdm import tqdm

# datasets
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem 
from datasets.tools import *

# models
import KAF

def LearningCurveKAF_MC(filt, testingSystem, n_samples, mc_runs, pred_step, params_file, savepath):
    # 1. data generation
    mc_samples = n_samples*(mc_runs + 1)
    if testingSystem in attractors:
        print("In progress...")
        return
    elif testingSystem in nonlinears:
        system = GenerateSystem(samples=mc_samples, systemType=testingSystem)
        system = z_scorer(system)
    else:
        raise ValueError("{} dataset is not supported".format(testingSystem))
    
    # 2. data preparation
    system_mc =  mc_sampler(system, n_samples, mc_runs)

    TMSE = []
    CB = []
    TradeOff = []
    
    # 3. parameter grid
    params_df = pd.read_csv(params_file)
    params = best_params_picker(params_df) #pendiente -> retorna un diccionario con 
                                                          #los params de cada filtro
   # 4. Monte Carlo simulations
    for run, X_mc in enumerate(system_mc):
        print("\nRunning Monte Carlo simulation #{}...\n".format(run+1))    
        
        X,y = Embedder(X=X_mc, embedding=2)
        Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)
        
        f = KAF_picker(filt, params)
        
        results = []
        for Xi,yi in tqdm(zip(Xtrain,ytrain)):
            try:
                y = f.evaluate(Xi,yi)
                if np.mod(run,pred_step)==0:
                    ypred = f.predict(Xtest)
                    err = ytest-ypred.reshape(-1,1)
                    TMSE.append(np.mean(err**2))
                    CB.append(len(f.CB))
            except:
                if np.mod(run,pred_step)==0:
                    TMSE.append(np.nan)
                    CB.append(np.nan)
