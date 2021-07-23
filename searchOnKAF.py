# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:01:38 2021

@author: Juan David
"""
import numpy as np
import argparse

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


parser = argparse.ArgumentParser()
parser.add_argument('--kaf', help='Filter to train')
parser.add_argument('--dataset', help='Dataset to use')
parser.add_argument('-N', help='Dataset length (if available)',default=1000,type=int)
parser.add_argument('-trainSplit', help='Train lenght percentage',default=1000,type=float)
parser.add_argument('-N_mc', help='Number of Monte Carlo simulations to run',default=10,type=int)




args = parser.parse_args()
kaf = args.kaf
db = args.dataset
n_samples = args.N
trainSplit = args.trainSplit
N_mc = args.N_mc

def main():
    from test_on_KAF import kafSearch_MC
    kafSearch_MC(kaf, db, n_samples,trainSplit, N_mc)
    # df.to_csv('GridSearchResults/' + kaf + '_' + db + '_' + str(n_samples) + '.csv')
    
if __name__ == "__main__":
    main()


# from test_on_KAF import kafSearch_MC
# kafSearch_MC("QKLMS", "wang", 100, 0.8,10)