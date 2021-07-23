# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:29:55 2020

@author: USUARIO
"""
import numpy as np
import matplotlib.pyplot as plt
import chaoticTimeSeries as cts
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def chaoticSystem(samples = 1000,systemType = None,display=False):
    
    # Dictionaty of available attractors
    attractors = {"chua": cts.chua,
                  "lorenz": cts.lorenz,
                  "duffing": cts.duffing,
                  "nose_hoover": cts.nose_hoover,
                  "rikitake": cts.rikitake,
                  "rossler": cts.rossler,
                  "wang": cts.wang}
    
    if systemType == None:
        raise ValueError('System type is missing')
        
    validKeys = attractors.keys()
    if not systemType in validKeys:
        raise ValueError('Attractor does not exist or is not supported')
        
    X = np.empty(samples + 1)
    Y = np.empty(samples + 1)
    Z = np.empty(samples + 1)
    
    dt = 0.01
    
    # Set initial values
    X[0], Y[0], Z[0] = (0., 1., 1.05)
    

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(samples):
        x_dot, y_dot, z_dot = attractors[systemType](X[i], Y[i], Z[i])
        X[i + 1] = X[i] + (x_dot * dt)
        Y[i + 1] = Y[i] + (y_dot * dt)
        Z[i + 1] = Z[i] + (z_dot * dt)
    
    # 3D Plot
    if display:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(X, Y, Z, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(str(systemType).capitalize() +" Attractor")
        plt.show()
    
    return X, Y, Z