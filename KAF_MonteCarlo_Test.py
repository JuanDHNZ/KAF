# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:36:22 2021

@author: Juan David
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
parser.add_argument('--params_file', help='Grid Search result csv for paratemer selection')
parser.add_argument('--savepath', help='path where to save results')


args = parser.parse_args()
kaf = args.kaf
dataset = args.dataset
samples = args.samples
mc_runs = args.mc_runs
params_file = args.params_file
savepath = args.savepath

#%% For testing only:

# kaf = "QKLMS_AKB"
# dataset = "chua"    
# samples = 105
# savepath = "results/Chua/MonteCarlo_test/"
# params_file = "results/Chua/split_3/mc_QKLMS_AKB_chua_4005.csv"
# mc_runs = 5

#%%
def main():
    from testing import best_params_MonteCarlo_simulation as bps
    bps(filt=kaf, 
        testingSystem=dataset,
        n_samples=samples, 
        mc_runs=mc_runs,
        params_file=params_file,
        savepath=savepath)
if __name__ == "__main__":
    main()



