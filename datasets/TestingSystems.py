# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:45:19 2020

@author: Juan David

Test systems from state-of.the-art papers
"""
import numpy as np

def GenerateSystem(samples = 1000, systemType = None):
    # Dictionaty of available attractors
    
    if systemType == None:
        raise ValueError('System type is missing')
    
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
        """
        Nonlinear Dynamic System
        
        Ji Zhao, Hongbin Zhang, J. Andrew Zhang, Gaussian kernel adaptive filters with adaptive kernel bandwidth,
        Signal Processing, Volume 166, 2020, 107270, ISSN 0165-1684, https://doi.org/10.1016/j.sigpro.2019.107270.
        """
        # highly nonlinear dynamical system:
        import numpy as np
        s = np.empty((samples+2,))
        s[0] = s[1] = 0.1
        for i in range(2,samples+2):
            s[i] = s[i-1]*(0.8 - 0.5*np.exp(-s[i-1]**2)) - (0.3 + 0.9*s[i-2]*np.exp(-s[i-1])) + 0.1*np.sin(s[i-1]*np.pi)
            
        # noise:
        import random
        seed = random.randint(0, 4000)
        rng = np.random.default_rng(seed)
        mean = 0
        var = 0.01
        noise = rng.normal(mean, var**0.5, samples)
        
        return s[2:] + noise
    
    if systemType == "4.1_AKB":
        """
        Simple Static Function
        
        Ji Zhao, Hongbin Zhang, J. Andrew Zhang, Gaussian kernel adaptive filters with adaptive kernel bandwidth,
        Signal Processing, Volume 166, 2020, 107270, ISSN 0165-1684, https://doi.org/10.1016/j.sigpro.2019.107270.
        """
        u = np.linspace(-np.pi,np.pi,samples)
        var = 1e-4
        noise = np.sqrt(var)*np.random.randn(samples)
        d = np.cos(8*u) + noise
        return u,d
    
    if systemType == "A_QKLMS":
        """
        B. Chen, S. Zhao, P. Zhu and J. C. Principe, "Quantized Kernel Least Mean Square Algorithm," 
        in IEEE Transactions on Neural Networks and Learning Systems, vol. 23, no. 1, pp. 22-32, Jan. 2012, 
        doi: 10.1109/TNNLS.2011.2178446.
        """
        mean = 0
             
        # generate noise
        import random
        import numpy as np
        seed = random.randint(0, 4000)
        rng = np.random.default_rng(seed)
        var = 0.04
        noise = rng.normal(mean, var**0.5, samples)
        
        #generate u
        seed = random.randint(0, 4000)
        rng = np.random.default_rng(seed)
        var = 1
        u = rng.normal(mean, var**0.5, samples)
        d =  0.2*(np.exp(-((u + 1)**2) /2) + np.exp(-((u - 1)**2) /2)) + noise      
        return u,d
    

def generateRandomVectorModel(samples):
    """
    Random Vector Model
    Kernel canonical-correlation Granger causality for multiple time series
    Guorong Wu, Xujun Duan, Wei Liao, Qing Gao, and Huafu Chen*
    Key Laboratory for NeuroInformation of Ministry of Education, School of Life Science and Technology,
    University of Electronic Science and Technology of China, Chengdu 610054, China
    """
    # Inicializar matrices para almacenar los valores de las funciones
    x1 = np.zeros(samples)
    x2 = np.zeros(samples)
    y1 = np.zeros(samples)
    y2 = np.zeros(samples)
    z = np.zeros(samples)
    
    # Generar ruido blanco gaussiano para cada función con varianza unitaria diferente
    ruido_x1 = np.random.normal(0, 1, samples)
    ruido_x2 = np.random.normal(0, 1, samples)
    ruido_y1 = np.random.normal(0, 1, samples)
    ruido_y2 = np.random.normal(0, 1, samples)
    ruido_z = np.random.normal(0, 1, samples)
    
    # Definir las ecuaciones como funciones lambda
    ecuacion_x1 = lambda t: -0.8 * x1[t-1] + 0.25 * np.sqrt(2) * x2[t-2] + 0.2 * ruido_x1[t]
    ecuacion_x2 = lambda t: 0.75 * x1[t-1] * (1 - x2[t-2]) + 0.3 * ruido_x2[t]
    ecuacion_y1 = lambda t: -0.4 * np.exp(-y1[t-1]) + 0.95 * z[t-1]**2 + 0.2 * ruido_y1[t]
    ecuacion_y2 = lambda t: -0.75 * y1[t-1]**2 + 0.5 * y2[t-1] + 0.4 * ruido_y2[t]
    ecuacion_z = lambda t: 0.3 * np.tan(x1[t-1]) - 0.8 * np.cos(x2[t-1]) + 0.2 * ruido_z[t]
    
    # Evaluar las funciones para cada muestra de tiempo
    for t in range(1, samples):
        # Evaluar las ecuaciones para cada función
        x1[t] = ecuacion_x1(t)
        x2[t] = ecuacion_x2(t)
        y1[t] = ecuacion_y1(t)
        y2[t] = ecuacion_y2(t)
        z[t] = ecuacion_z(t)
    
    # Retornar los valores de las funciones como una matriz
    return np.array([x1, x2, y1, y2, z])

if __name__ == "__main__":
    # Ejemplo de uso
    num_muestras = 10
    valores_funciones = generateRandomVectorModel(num_muestras)
    print(valores_funciones)


