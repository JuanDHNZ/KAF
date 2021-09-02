# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:36:47 2021

@author: USUARIO
"""

from testing import CB_visualizer
ts = '4.2_AKB'
samples = 100
# p_file = "results/4.2/New/mc_QKLMS_AMK_4.2_AKB_5003.csv"
p_file = "results/4.2/GridSearch_MonteCarlo/mc_QKLMS_AMK_4.2_AKB_5003.csv"

runs_names = ["CB_plot_{}".format(run) for run in range(21,41)]
for name in runs_names:
    model = CB_visualizer(testingSystem=ts, n_samples=samples, params_file = p_file, savename=name)
