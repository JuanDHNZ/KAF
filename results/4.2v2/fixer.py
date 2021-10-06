# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 08:07:34 2021

@author: Juan David
"""
import numpy as np
import pandas as pd

partial_results = "PARTIAL_mc_QKLMS_AKB_4.2_AKB_5003.csv"
results = pd.read_csv(partial_results)

TMSE = np.array(TMSE).reshape(mc_runs,-1)
CB = np.array(CB).reshape(mc_runs,-1)
# TradeOff = np.array(TradeOff).reshape(mc_runs,-1)

mean_TMSE = np.nanmean(TMSE, axis=0)
std_TMSE = np.nanstd(TMSE, axis=0)

mean_CB = np.nanmean(CB, axis=0)
median_CB = np.nanmedian(CB, axis=0)
mode_CB = scipy.stats.mode(CB, axis=0)
std_CB = np.nanstd(CB, axis=0)
    
results = [p for p in params]
results_df = pd.DataFrame(data=results)
results_df['TMSE'] = mean_TMSE 
results_df['TMSE_std'] = std_TMSE
results_df['CB_mean'] = mean_CB.astype(int)
results_df['CB_median'] = median_CB.astype(int)
results_df['CB_mode'] = mode_CB[0].ravel()
results_df['CB_std'] = std_CB

# remove partials CSV
# partial_file = savepath + "PARTIAL_mc_{}_{}_{}.csv".format(filt,testingSystem,n_samples)
# if(os.path.exists(partial_file) and os.path.isfile(partial_file)):
#     try:    
#         os.remove(partial_file)
#     except:
#         print("PARTIAL file not deleted.")
results_df.to_csv(savepath + "mc_{}_{}_{}.csv".format(filt,testingSystem,n_samples))
return