import pandas as pd  
import numpy as np
from tqdm import tqdm

# datasets
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem 
from datasets.tools import *

# models and metrics
import KAF
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

attractors = ["chua","lorenz","duffing","nose_hoover","rikitake","rossler","wang"]
nonlinears = ["4.1_AKB", "4.2_AKB"]

def GridSearchKAF(filt, grid, testingSystem, n_samples,MC_runs):
    n_samples = 5003
    testingSystem = "4.2_AKB"
    filt = "QKLMS"
    # 1. data generation
    if testingSystem in attractors:
        
    elif testingSystem in nonlinears:
        system = GenerateSystem(samples=n_samples, systemType=testingSystem)
        system = tools.z_scorer(system)
    else:
        raise ValueError("{} dataset is not supported".format(testingSystem))

    # 2. data preparation
    X,y = Embedder(X=system, embedding=2)
    Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)

    # 3. parameter grid 
    params = parameter_picker(filt, grid)
    
    # 4. grid evaluation
    results = []
    for params in tqdm(grid):
        try:
            f = KAF_picker(filt, params)
            y = f.evaluate(Xtrain,ytrain)
            ypred = f.predict(Xtest)
            err = ytest-ypred.reshape(-1,1)
            p['TMSE'] = np.mean(err**2)
            p['CB'] = len(f.CB)
            p['tradeOff'] = tradeOff(p['TMSE'],p['CB']/n_train)
        except:
            p['TMSE'] = np.nan
            p['CB'] = np.nan
            p['tradeOff'] = np.nan
        results.append(p)
        pd.DataFrame(data=results).to_csv(savename)
    return


def GridSearchKAF_MC(filt, grid, testingSystem, n_samples,MC_runs):
    n_samples = 5003
    testingSystem = "4.2_AKB"
    filt = "QKLMS"
    # 1. data generation
    if testingSystem in attractors:
        
    elif testingSystem in nonlinears:
        system = GenerateSystem(samples=n_samples, systemType=testingSystem)
        system = tools.z_scorer(system)
    else:
        raise ValueError("{} dataset is not supported".format(testingSystem))

    # 2. data preparation
    X,y = Embedder(X=system, embedding=2)
    Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)

    # 3. parameter grid 
    params = parameter_picker(filt, grid)
    
    # 4. grid evaluation
    results = []
    for params in tqdm(grid):
        try:
            f = KAF_picker(filt, params)
            y = f.evaluate(Xtrain,ytrain)
            ypred = f.predict(Xtest)
            err = ytest-ypred.reshape(-1,1)
            p['TMSE'] = np.mean(err**2)
            p['CB'] = len(f.CB)
            p['tradeOff'] = tradeOff(p['TMSE'],p['CB']/n_train)
        except:
            p['TMSE'] = np.nan
            p['CB'] = np.nan
            p['tradeOff'] = np.nan
        results.append(p)
        pd.DataFrame(data=results).to_csv(savename)
    return

