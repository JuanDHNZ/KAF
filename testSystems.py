# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:45:19 2020

@author: Juan David

Test systems from state-of.the-art papers
"""

def testSystems(samples = 1000, systemType = None):
    # Dictionaty of available attractors
    
    if systemType == None:
        raise ValueError('System type is missing')
    
    import numpy as np
    if systemType == "1":      
        mean = [0]
        cov = np.asarray([1]).reshape(1,1)
        u = np.random.multivariate_normal(mean,cov,size=samples).reshape(-1,)    
        w = np.asarray([0.227, 0.460, 0.688, 0.460, 0.227])
        d = np.convolve(u,w,mode="same")
        return u,d
    
    if systemType == "2":
        import math as mt
        import numpy as np
        s = np.empty((samples+2,))
        s[0] = 0.1
        s[1] = 0.1
        i = 2
        while True:
            s[i] = s[i-1]*(0.8 - 0.5*mt.exp(-s[i-1]**2)) - s[i-2]*(0.3 + 0.9*mt.exp(-s[i-1]**2)) + 0.1*mt.sin(s[i-1]*mt.pi)
            i+=1
            if(i == samples+2):
                return s[-samples:]
    if systemType == "3":
        import pandas as pd
        data = pd.read_csv("SN_m_tot_V2.csv", sep=";")
        monthly_mean = data['monthly_mean_total']
        return monthly_mean[-samples:]
    
    if systemType == "4.2_AKB":
        import numpy as np
        s = np.empty((samples+2,))
        s[0] = s[1] = 0.1
        for i in range(2,samples+2):
            s[i] = s[i-1]*(0.8 - 0.5*np.exp(-s[i-1]**2)) - (0.3 + 0.9*s[i-2]*np.exp(-s[i-1])) + 0.1*np.sin(s[i-1]*np.pi)
            # var = 0.01
            # noise = np.sqrt(var)*np.random.randn(samples)
        return s[2:] #+ noise
    
    if systemType == "4.1_AKB":
        import numpy as np
        u = np.linspace(-np.pi,np.pi,samples)
        var = 1e-4
        noise = np.sqrt(var)*np.random.randn(samples)
        d = np.cos(8*u) + noise
        return u,d
        
    

    

