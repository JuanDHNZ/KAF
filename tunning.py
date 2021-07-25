import pandas as pd  
import numpy as np
from tqdm import tqdm

# datasets
from datasets.ChaoticTimeSeries import GenerateAttractor
from datasets.TestingSystems import GenerateSystem 
import datasets.tools

# models and metrics
import KAF
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

attractors = ["chua","lorenz","duffing","nose_hoover","rikitake","rossler","wang"]
nonlinears = ["4.1_AKB", "4.2_AKB"]

def GridSearchKAF(KAF, grid, testingSystem,n_samples,trainSplit,MC_runs):
    # 1. Generar Datos
    if testingSystem in attractors:
        
    elif testingSystem in nonlinears:
        system = GenerateSystem(samples=5003, systemType="4.2_AKB")
        system = tools.z_scorer(system)
    else:
        raise ValueError("{} dataset is not supported".format(testingSystem))   
    
    X,y = Embedder(X=system, embedding=2)
    Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X,y)

    # 2. Filter 
    

    return