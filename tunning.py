import pandas as pd  
import numpy as np
from tqdm import tqdm

# datasets
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem 
from datasets.tools import *

# models and metrics
import KAF

attractors = ["chua","lorenz","duffing","nose_hoover","rikitake","rossler","wang"]
nonlinears = ["4.1_AKB", "4.2_AKB"]

def GridSearchKAF(filt, grid, testingSystem, n_samples, savepath):
    # 1. data generation
    if testingSystem in attractors:
        print("In progress...")
        return
    elif testingSystem in nonlinears:
        system = GenerateSystem(samples=n_samples, systemType=testingSystem)
        system = z_scorer(system)
    else:
        raise ValueError("{} dataset is not supported".format(testingSystem))

    # 2. data preparation
    X,y = Embedder(X=system, embedding=2)
    Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)

    # 3. parameter grid 
    params = parameter_picker(filt, grid)
    
    # 4. grid evaluation
    results = []
    for p in tqdm(params):
        try:
            f = KAF_picker(filt, p)
            y = f.evaluate(Xtrain,ytrain)
            ypred = f.predict(Xtest)
            err = ytest-ypred.reshape(-1,1)
            p['TMSE'] = np.mean(err**2)
            p['CB'] = len(f.CB)
            p['tradeOff'] = tradeOff(p['TMSE'],p['CB']/len(Xtrain))
        except:
            p['TMSE'] = np.nan
            p['CB'] = np.nan
            p['tradeOff'] = np.nan
        results.append(p)
        pd.DataFrame(data=results).to_csv(savepath + "{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    return


def GridSearchKAF_MC(filt, grid, testingSystem, n_samples, mc_runs, savepath):
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
    params = parameter_picker(filt, grid)
    
    # 4. Monte Carlo simulations
    for run, X_mc in enumerate(system_mc):
        print("\nRunning Monte Carlo simulation #{}...\n".format(run+1))    
        
        X,y = Embedder(X=X_mc, embedding=2)
        Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)
    
        results = []
        for p in tqdm(params):
            try:
                f = KAF_picker(filt, p)
                if filt == "QKLMS_AMK":
                    f.evaluate(Xtrain[:100],ytrain[:100])
                y = f.evaluate(Xtrain,ytrain)
                ypred = f.predict(Xtest)
                err = ytest-ypred.reshape(-1,1)
                TMSE.append(np.mean(err**2))
                CB.append(len(f.CB))
                # TradeOff.append(tradeOff(np.mean(err**2),len(f.CB)/len(Xtrain)))
            except:
                TMSE.append(np.nan)
                CB.append(np.nan)
                # TradeOff.append(np.nan)
        
    TMSE = np.array(TMSE).reshape(mc_runs,-1)
    CB = np.array(CB).reshape(mc_runs,-1)
    # TradeOff = np.array(TradeOff).reshape(mc_runs,-1)
    
    mean_TMSE = np.nanmean(TMSE, axis=0)
    mean_CB = np.nanmean(CB, axis=0)
        
    results = [p for p in params]
    results_df = pd.DataFrame(data=results)
    results_df['TMSE'] = mean_TMSE 
    results_df['CB'] = mean_CB.astype(int)
    # results_df['tradeOff_dist'] = tradeOff_distance
        
    results_df.to_csv(savepath + "mc_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
    return

