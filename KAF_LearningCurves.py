# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 16:03:36 2021

@author: USUARIO
"""
import argparse
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


parser = argparse.ArgumentParser()
parser.add_argument('--kaf', help='Filter to train')
parser.add_argument('--dataset', help='Dataset to use')
parser.add_argument('-samples', help='Dataset length',default=1000,type=int)
parser.add_argument('-mc_runs', help='Monte Carlo runs',default=50,type=int)
parser.add_argument('-pred_step', help='Prediction steps for testing MSE calculation',default=10,type=int)
parser.add_argument('--params_file', help='Grid Search result csv for paratemer selection')
parser.add_argument('--savepath', help='path where to save results')

args = parser.parse_args()
kaf = args.kaf
dataset = args.dataset
samples = args.samples
mc_runs = args.mc_runs
pred_step = args.pred_step
params_file = args.params_file
savepath = args.savepath

#%% For testing only:

# kaf = "QKLMS"
# dataset = "4.2_AKB"    
# samples = 100
# savepath = "results/4.2/New/learning_curves/"
# params_file = "results/4.2/New/QKLMS_4.2_AKB_5003.csv"
# mc_runs = 5
# pred_step = 10
#%%
def main():   
    from testing import LearningCurveKAF_MC
    LearningCurveKAF_MC(filt=kaf, 
                        testingSystem=dataset,
                        n_samples=samples, 
                        mc_runs=mc_runs, 
                        pred_step=10,
                        params_file=params_file,
                        savepath=savepath)
    
    
if __name__ == "__main__":
    main()