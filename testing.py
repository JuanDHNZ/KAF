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
    embedding=5
    print("Generating data...")
    mc_samples = (n_samples+embedding)*(mc_runs + 1)
    if testingSystem in attractors:
        x,y,z = GenerateAttractor(samples=mc_samples, attractor=testingSystem)
        system = z_scorer(x)
    elif testingSystem in nonlinears:
        system = GenerateSystem(samples=mc_samples, systemType=testingSystem)
        system = z_scorer(system)
    else:
        raise ValueError("{} dataset is not supported".format(testingSystem))
    
    # 2. data preparation
    print("Data preparation...")
    system_mc =  mc_sampler(system, n_samples, mc_runs,embedding=embedding)
    
    # 3. parameter grid
    # params_df = pd.read_csv(params_file)
    # params = best_params_picker(filt, params_df,criteria='CB_median')
    
    params = {'eta':0.1,
           'epsilon':0.2,
           'sigma':np.sqrt(2)/2,
           #'mu':0.4,
           #'K': 8
           }
    
    results_tmse = []     
    results_cb= []        

                              
    # 4. Monte Carlo simulations
    print("Evualuations...")
    for run, X_mc in enumerate(system_mc):
        print("\nRunning Monte Carlo simulation #{}...\n".format(run+1))  
        
        # train, test, dic_train, dic_test = noisy_chua_generator(n_samples, alpha=12, beta=28)
        # train = z_scorer(train)
        # test = z_scorer(test)
        # Xtrain,ytrain = Embedder(X=train, embedding=embedding)
        # Xtest,ytest = Embedder(X=test, embedding=embedding)
        #X,y = Embedder(X=X_mc, embedding=5)
        
        train_portion=4000/4200
        train_size = int(len(X)*train_portion)
        Xtrain,ytrain = X_mc[:train_size,:-1],X_mc[:train_size,-1]
        Xtest,ytest = X_mc[train_size:,:-1],X_mc[train_size:,-1]        
        #Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y,train_portion=4000/4200)
        
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
    results.to_csv(savepath + "tmse_{}_{}_{}_K_{}.csv".format(filt,testingSystem,n_samples,params['K']))
    return

def LearningCurveKAF_MC2(filt, testingSystem, n_samples, mc_runs, pred_step, params_file, savepath):
    # 1. data generation
    print("Generating data...")
    mc_samples = n_samples*(mc_runs + 1)
    if testingSystem in attractors:
        print("In progress...")
        return
    elif testingSystem in nonlinears:
        u,d = GenerateSystem(samples=mc_samples, systemType=testingSystem)
    else:
        raise ValueError("{} dataset is not supported".format(testingSystem))
    
    # 2. data preparation
    print("Data preparation...")
    u_mc =  mc_sampler(u, n_samples, mc_runs)
    d_mc =  mc_sampler(d, n_samples, mc_runs)
    
    # 3. parameter grid
    params_df = pd.read_csv(params_file)
    # params = best_params_picker(filt, params_df,criteria='CB_median') 
    params = {'eta':0.6,
               'epsilon':2.0,
               'sigma':1.0}
    results_tmse = []     
    results_cb= []        
    
    print(params)
    # embedding = 5
                              
    # 4. Monte Carlo simulations
    print("Evualuations...")
    for run, (u, d)in enumerate(zip(u_mc,d_mc)):
        print("\nRunning Monte Carlo simulation #{}...\n".format(run+1))  
        
        # train, test, dic_train, dic_test = noisy_chua_generator(n_samples, alpha=12, beta=28)
        # train = z_scorer(train)
        # test = z_scorer(test)
        # Xtrain,ytrain = Embedder(X=train, embedding=embedding)
        # Xtest,ytest = Embedder(X=test, embedding=embedding)
        # X,y = Embedder(X=X_mc, embedding=2)
        Xtrain, Xtest, ytrain, ytest = TrainTestSplit(u,d,train_portion=0.9615)
        
        f = KAF_picker(filt, params)
        
        TMSE = []
        CB = []
        
        if filt == "QKLMS_AMK":
            f.evaluate(Xtrain[:100],ytrain[:100])

        for n,(Xi,yi) in tqdm(enumerate(zip(Xtrain,ytrain))):
            try:
                y = f.evaluate(np.array(Xi).reshape(-1,1),np.array(yi).reshape(-1,1))
                if np.mod(n,pred_step)==0:
                    ypred = f.predict(Xtest.reshape(-1,1))
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
    results.to_csv(savepath + "{}_{}_{}_sigma{}.csv".format(filt,testingSystem,n_samples,params['sigma']))
    return

def LearningCurveKAF_MC3(filt, testingSystem, n_samples, mc_runs, pred_step, params_file, savepath):
    """Excess Mean Square Error learning Curve calculation"""
    # 1. data generation
    # print("Generating data...")
    # mc_samples = n_samples*(mc_runs + 1)

    # u,d = GenerateSystem(samples=mc_samples, systemType=testingSystem)

    # # 2. data preparation
    # print("Data preparation...")
    # u_mc =  mc_sampler(u, n_samples, mc_runs)
    # d_mc =  mc_sampler(d, n_samples, mc_runs)
    
    # 3. parameter grid
    params_df = pd.read_csv(params_file)
    # params = best_params_picker(filt, params_df,criteria='CB_median') 
    params = {'eta':0.6,
               'epsilon':2.0,
               'sigma':1.0}
    results_emse = []     
    results_cb= []        
    
    # embedding = 5
                              
    # 4. Monte Carlo simulations
    print("Evualuations...")
    # for run, (u, d)in enumerate(zip(u_mc,d_mc)):
    for run in range(mc_runs):
        print("\nRunning Monte Carlo simulation #{}...\n".format(run+1))  
        
        # train, test, dic_train, dic_test = noisy_chua_generator(n_samples, alpha=12, beta=28)
        # train = z_scorer(train)
        # test = z_scorer(test)
        # Xtrain,ytrain = Embedder(X=train, embedding=embedding)
        # Xtest,ytest = Embedder(X=test, embedding=embedding)
        # X,y = Embedder(X=X_mc, embedding=2)
        u,d = GenerateSystem(samples=n_samples, systemType=testingSystem)
        Xtrain, Xtest, ytrain, ytest = TrainTestSplit(u,d,train_portion=0.9615)
        
        f = KAF_picker(filt, params)
        
        EMSE = []
        CB = []
        
        if filt == "QKLMS_AMK":
            f.evaluate(Xtrain[:100],ytrain[:100])

        for n,(Xi,yi) in tqdm(enumerate(zip(Xtrain,ytrain))):
            try:
                y = f.evaluate(np.array(Xi).reshape(-1,1),np.array(yi).reshape(-1,1))
                EMSE.append(f.apriori_error**2)
                # if np.mod(n,pred_step)==0:
                    # ypred = f.predict(Xtest.reshape(-1,1))
                    # err = ytest-ypred.reshape(-1,1)
                    # TMSE.append(np.mean(err**2))
                CB.append(len(f.CB))
            except:
                if np.mod(n,pred_step)==0:
                    EMSE.append(np.nan)
                    CB.append(np.nan)
        results_emse.append(EMSE) 
        results_cb.append(CB)
    
    print(np.array(results_emse).shape)
    all_emse = pd.DataFrame(data=results_emse).T
    emse_cols = ["EMSE_{}".format(run) for run in range(mc_runs)]
    all_emse.columns = emse_cols
    all_cb = pd.DataFrame(data=results_cb).T
    cb_cols = ["CB_{}".format(run) for run in range(mc_runs)]
    all_cb.columns = cb_cols
    results = pd.concat([all_cb,all_emse], axis=1)
    results['mean_CB'] = all_cb.mean(axis=1).values
    results['mean_EMSE'] = all_emse.mean(axis=1).values
    results.to_csv(savepath + "emse_klms{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    return


def LearningCurveKAF_MC4(filt, testingSystem, n_samples, mc_runs, pred_step, params_file, savepath):
    # 1. data generation
    # print("Generating data...")
    # mc_samples = n_samples*(mc_runs + 1)
    # if testingSystem in attractors:
    #     print("In progress...")
    #     return
    # elif testingSystem in nonlinears:
    #     u,d = GenerateSystem(samples=mc_samples, systemType=testingSystem)
    # else:
    #     raise ValueError("{} dataset is not supported".format(testingSystem))
    
    # # 2. data preparation
    # print("Data preparation...")
    # u_mc =  mc_sampler(u, n_samples, mc_runs)
    # d_mc =  mc_sampler(d, n_samples, mc_runs)
    
    # 3. parameter grid
    params_df = pd.read_csv(params_file)
    # params = best_params_picker(filt, params_df,criteria='CB_median') 
    params = {'eta':0.1,
               'epsilon':0.8,
               'sigma':1.0}
    results_tmse = []     
    results_cb= []        
    
    # embedding = 5
                              
    # 4. Monte Carlo simulations
    print("Evualuations...")
    # for run, (u, d)in enumerate(zip(u_mc,d_mc)):
    for run in range(mc_runs):
        print("\nRunning Monte Carlo simulation #{}...\n".format(run+1))  
        
        # train, test, dic_train, dic_test = noisy_chua_generator(n_samples, alpha=12, beta=28)
        # train = z_scorer(train)
        # test = z_scorer(test)
        # Xtrain,ytrain = Embedder(X=train, embedding=embedding)
        # Xtest,ytest = Embedder(X=test, embedding=embedding)
        # X,y = Embedder(X=X_mc, embedding=2)
        u,d = GenerateSystem(samples=n_samples, systemType=testingSystem)
        Xtrain, Xtest, ytrain, ytest = TrainTestSplit(u,d,train_portion=0.9090)
        
        f = KAF_picker(filt, params)
        
        TMSE = []
        CB = []
        
        if filt == "QKLMS_AMK":
            f.evaluate(Xtrain[:100],ytrain[:100])

        for n,(Xi,yi) in tqdm(enumerate(zip(Xtrain,ytrain))):
            try:
                y = f.evaluate(np.array(Xi).reshape(-1,1),np.array(yi).reshape(-1,1))
                if np.mod(n,pred_step)==0:
                    ypred = f.predict(Xtest.reshape(-1,1))
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
    results.to_csv(savepath + "tmse_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
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
    params = best_params_picker(filt, params_df, "CB_median")
    
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
    


def learning_curve_train_error(filt, testingSystem, n_samples, mc_runs, pred_step, params_file, savepath):
    # 1. data generation
    print("Generating data...")
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
    print("Data preparation...")
    system_mc =  mc_sampler(system, n_samples, mc_runs)
    
    # 3. parameter grid
    params_df = pd.read_csv(params_file)
    params = best_params_picker(filt, params_df,criteria='CB_median')
    
    results_tmse_train = [] 
    results_tmse_test = [] 
    results_ape_train = [] 
    results_ape_test = []     
    results_cb= []        

    embedding = 5
                              
    # 4. Monte Carlo simulations
    print("Evualuations...")
    for run, X_mc in enumerate(system_mc):
        print("\nRunning Monte Carlo simulation #{}...\n".format(run+1))  
        
        X,y = Embedder(X=X_mc, embedding=2)
        Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)
        
        f = KAF_picker(filt, params)
        
        trainMSE = []
        testMSE = []
        trainAPE = []
        testAPE = []
        CB = []
        
        if filt == "QKLMS_AMK":
            f.evaluate(Xtrain[:100],ytrain[:100])
        
       
        # # For testing MSE calculation
        # for n,(Xi,yi) in tqdm(enumerate(zip(Xtrain,ytrain))):
        #     try:
        #         y = f.evaluate(Xi,yi)
        #         if np.mod(n,pred_step)==0:    
                    
        #             train_err = ytrain-y.reshape(-1,1)
        #             trainMSE.append(np.mean(train_rr**2))

        #             ypred = f.predict(Xtest)     
        #             test_err = ytest-ypred.reshape(-1,1)
        #             testMSE.append(np.mean(err**2))
                    
        #             CB.append(len(f.CB))
        #     except:
        #         trainMSE.append(0)
        #         testMSE.append(0)
                
        print("Train & Test - APE calculation...")   
        # For APE calculation
        f = KAF_picker(filt, params)
        if filt == "QKLMS_AMK":
            f.evaluate(Xtrain[:100],ytrain[:100])
        y = f.evaluate(Xtrain,ytrain)
        ypred = f.predict(Xtest)
        
        print("Train & Test - Testing MSE calculation...")
        results_tmse_train.append(MSE(ytrain,y.reshape(-1,1)))
        results_tmse_test.append(MSE(ytest,ypred))
        results_ape_train.append(APE(ytrain,y).ravel())
        results_ape_test.append(APE(ytest,ypred).ravel())  
        results_cb.append(CB)
    
    print("Saving...")
    # APE save
    results = pd.DataFrame(data=np.array(results_ape_train)).T
    results_cols = ["APE_{}".format(run) for run in range(mc_runs)]
    results.columns = results_cols
    results.to_csv(savepath + "train_ape_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    
    results = pd.DataFrame(data=np.array(results_ape_test)).T
    results_cols = ["APE_{}".format(run) for run in range(mc_runs)]
    results.columns = results_cols
    results.to_csv(savepath + "test_ape_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    
    # MSE save
    all_tmse_train = pd.DataFrame(data=results_tmse_train).T
    train_tmse_cols = ["train_TMSE_{}".format(run) for run in range(mc_runs)]
    all_tmse_train.columns = train_tmse_cols
    all_tmse_train['mean_trainTMSE'] = all_tmse_train.mean(axis=1).values
    all_tmse_train['std_trainTMSE'] = all_tmse_train.std(axis=1).values
    all_tmse_train.to_csv(savepath + "train_TMSE_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    
    all_tmse_test = pd.DataFrame(data=results_tmse_test).T
    test_tmse_cols = ["test_TMSE_{}".format(run) for run in range(mc_runs)]
    all_tmse_test.columns = test_tmse_cols
    all_tmse_test['mean_testTMSE'] = all_tmse_test.mean(axis=1).values
    all_tmse_test['std_testTMSE'] = all_tmse_test.std(axis=1).values
    all_tmse_test.to_csv(savepath + "test_TMSE_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    return


def CB_visualizer_AKB(testingSystem, n_samples, filt,  params_file, savename):
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
    params_df = pd.read_csv(params_file)
    params = best_params_picker(filt, params_df, "CB_median")
    # params["mu"] = 1.2
    # params["K"] = 2
    # 4. Evaluation and Plotting
    f = KAF_picker(filt, params)
        
    TMSE = []
    CB = []
    X_eval = []
    
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
            plotCB_AKB(f,np.array(X_eval), savename)
        except:
            TMSE.append(np.nan)
            CB.append(np.nan) 
    
    return f