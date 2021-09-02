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

attractors = ["chua","lorenz","duffing","nose_hoover","rikitake","rossler","wang"]
nonlinears = ["4.1_AKB", "4.2_AKB"]

def LearningCurveKAF_MC(filt, testingSystem, n_samples, mc_runs, pred_step, params_file, savepath):
    # 1. data generation
    # print("Generating data...")
    # mc_samples = n_samples*(mc_runs + 1)
    # if testingSystem in attractors:
    #     print("In progress...")
    #     return
    # elif testingSystem in nonlinears:
    #     system = GenerateSystem(samples=mc_samples, systemType=testingSystem)
    #     system = z_scorer(system)
    # else:
    #     raise ValueError("{} dataset is not supported".format(testingSystem))
    
    # # 2. data preparation
    # print("Data preparation...")
    # system_mc =  mc_sampler(system, n_samples, mc_runs)
    
    # 3. parameter grid
    params_df = pd.read_csv(params_file)
    params = best_params_picker(filt, params_df,criteria='TMSE') 
    results_tmse = []     
    results_cb= []        

    embedding = 5
                              
    # 4. Monte Carlo simulations
    print("Evualuations...")
    for run in range(mc_runs):
        print("\nRunning Monte Carlo simulation #{}...\n".format(run+1))  
        
        train, test, dic_train, dic_test = noisy_chua_generator(n_samples, alpha=12, beta=28)
        train = z_scorer(train)
        test = z_scorer(test)
        Xtrain,ytrain = Embedder(X=train, embedding=embedding)
        Xtest,ytest = Embedder(X=test, embedding=embedding)
        
        
        f = KAF_picker(filt, params)
        
        TMSE = []
        CB = []
        
        if filt == "QKLMS_AMK":
            f.evaluate(Xtrain[:100],ytrain[:100])

        for n,(Xi,yi) in tqdm(enumerate(zip(Xtrain,ytrain))):
            try:
                y = f.evaluate(Xi,yi)
                if np.mod(n,pred_step)==0:
                    ypred = f.predict(Xtest)
                    err = ytest-ypred.reshape(-1,1)
                    TMSE.append(np.mean(err**2))
                    CB.append(len(f.CB))
            except:
                if np.mod(n,pred_step)==0:
                    TMSE.append(np.nan)
                    CB.append(np.nan)
        results_tmse.append(TMSE) 
        results_cb.append(CB)
        
    all_tmse = pd.DataFrame(data=results_tmse).T
    tmse_cols = ["TMSE_{}".format(run) for run in range(mc_runs)]
    all_tmse.columns = tmse_cols
    all_cb = pd.DataFrame(data=results_cb).T
    cb_cols = ["CB_{}".format(run) for run in range(mc_runs)]
    all_cb.columns = cb_cols
    results = pd.concat([all_cb,all_tmse], axis=1)
    results['mean_CB'] = all_cb.mean(axis=1).values
    results['mean_TMSE'] = all_tmse.mean(axis=1).values
    results.to_csv(savepath + "tmse_{}_{}_{}_K_6.csv".format(filt,testingSystem,n_samples))
    return

def best_params_MonteCarlo_simulation(filt, testingSystem, n_samples, mc_runs, params_file, savepath): 
    # 1. parameter grid
    params_df = pd.read_csv(params_file)
    params = best_params_picker(filt, params_df,criteria='MAPE')       
    embedding = 5
    
    TMAPE = []   
    TAPE = []
    CB =  []                    
    # 2. Monte Carlo simulations
    for run in range(mc_runs):
        print("\nRunning Monte Carlo simulation #{}...\n".format(run+1))  
        
        train, test, dic_train, dic_test = noisy_chua_generator(n_samples, alpha=12, beta=28)
        train = z_scorer(train)
        X,y = Embedder(X=train, embedding=embedding)

        f = KAF_picker(filt, params)
        
        # AMK initialization
        if filt == "QKLMS_AMK":
            f.evaluate(X[:100],y[:100])  
        try:
            y_pred_train = f.evaluate(X,y)
            # y_pred = f.predict(X)
            TMAPE.append(MAPE(y,y_pred_train))
            TAPE.append(APE(y,y_pred_train).ravel())
            CB.append(len(f.CB))
        except:
            TMAPE.append(np.nan)
            TAPE.append(np.nan)
        
    results = pd.DataFrame(data=TMAPE).T
    results_cols = ["MAPE_{}".format(run) for run in range(mc_runs)]
    results.columns = results_cols
    results.to_csv(savepath + "mape_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    
    results = pd.DataFrame(data=np.array(TAPE)).T
    results_cols = ["APE_{}".format(run) for run in range(mc_runs)]
    results.columns = results_cols
    results.to_csv(savepath + "ape_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    
    results = pd.DataFrame(data=np.array(CB)).T
    results_cols = ["CB_{}".format(run) for run in range(mc_runs)]
    results.columns = results_cols
    results.to_csv(savepath + "cb_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    return

def CB_visualizer(testingSystem, n_samples, params_file, savename):
    # 1. Data generation
    print("Data generation...")
    if testingSystem in attractors:
        print("In progress...")
        return
    elif testingSystem in nonlinears:
        system = GenerateSystem(samples=n_samples, systemType=testingSystem)
        system = z_scorer(system)
    else:
        raise ValueError("{} dataset is not supported".format(testingSystem)) 
    
    # 2. Data Preparation
    print("Data preparation...")
    X,y = Embedder(X=system, embedding=2)
    Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)

    # 3. Parameter selection 
    print("Parameter selection ...")
    filt = "QKLMS_AMK"
    params_df = pd.read_csv(params_file)
    params = best_params_picker(filt, params_df, "TMSE")
    
    # 4. Evaluation and Plotting
    f = KAF_picker(filt, params)
        
    TMSE = []
    CB = []
    X_eval = []
    # 4.1. Intialize
    f.evaluate(Xtrain[:100],ytrain[:100])
    # 4.2. Evaluate
    for Xi,yi in tqdm(zip(Xtrain,ytrain)):
        try:
            y = f.evaluate(Xi,yi)
            X_eval.append(Xi)
            ypred = f.predict(Xtest)
            err = ytest-ypred.reshape(-1,1)
            TMSE.append(np.mean(err**2))
            CB.append(len(f.CB))
            # plotCB(f,np.array(X_eval))
        except:
            TMSE.append(np.nan)
            CB.append(np.nan) 
    plotCB(f,np.array(X_eval), savename)
    return f
    
