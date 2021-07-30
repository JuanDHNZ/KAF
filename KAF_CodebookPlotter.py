# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:36:47 2021

@author: USUARIO
"""

from testing import CB_visualizer
ts = '4.2_AKB'
samples = 100
p_file = "results/4.2/New/mc_QKLMS_AMK_4.2_AKB_5003.csv"
model = CB_visualizer(testingSystem=ts, n_samples=samples, params_file = p_file)
print("\n\nCB size = ",len(model.CB))
