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
parser.add_argument('--savepath', help='path where to save results')

args = parser.parse_args()
kaf = args.kaf
dataset = args.dataset
samples = args.samples
mc_runs = args.mc_runs
savepath = args.savepath

#%% For testing only:

# kaf = "QKLMS_AMK"
# dataset = "4.2_AKB"    
# samples = 23
# savepath = "results/4.2/New/"
# mc_runs = 5
#%%
def main():
    from datasets.tools import grid_picker
    grid  = grid_picker(kaf)
    
    from tunning import  GridSearchKAF, GridSearchKAF_MC
    # GridSearchKAF(filt=kaf,grid=grid,testingSystem=dataset,n_samples=samples,savepath=savepath)
    GridSearchKAF_MC(filt=kaf,grid=grid,testingSystem=dataset,n_samples=samples,mc_runs=mc_runs,savepath=savepath)

    
if __name__ == "__main__":
    main()

