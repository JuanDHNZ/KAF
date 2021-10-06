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
parser.add_argument('-embedding', help='Test embedding',default=5,type=int)
parser.add_argument('--savepath', help='path where to save results')

args = parser.parse_args()
kaf = args.kaf
dataset = args.dataset
samples = args.samples
mc_runs = args.mc_runs
embedding = args.embedding
savepath = args.savepath

#%% For testing only:

# kaf = "QKLMS_AMK"
# dataset = "chua"  
# samples = 100
# savepath = "results/Chua/"
# mc_runs = 2
# embedding = 5
#%%
def main():
    from datasets.tools import grid_picker
    grid  = grid_picker(kaf)
    
    from tunning import  GridSearchKAF, GridSearchKAF_MC, GridSearchKAF_MC_chua
    # GridSearchKAF(filt=kaf,grid=grid,testingSystem=dataset,n_samples=samples,savepath=savepath)
    GridSearchKAF_MC(filt=kaf,grid=grid,testingSystem=dataset,n_samples=samples,mc_runs=mc_runs,embedding=embedding,savepath=savepath)
    # GridSearchKAF_MC_chua(filt=kaf,grid=grid,testingSystem=dataset,n_samples=samples,mc_runs=mc_runs,embedding=embedding,savepath=savepath)
    
if __name__ == "__main__":
    main()

