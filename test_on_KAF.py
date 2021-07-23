# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:42:19 2021

@author: Juan David
"""
def MC_BestParameters(inputSignal, monteCarloRuns, singleRunDataSize, trainSplitPercentage, signalEmbedding, filterType, parameters,ExTest=False,y=None,z=None):
    '''MonteCarlo for QKLMS'''
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
         
    trainLength = int(singleRunDataSize*trainSplitPercentage)
    testLength = singleRunDataSize - trainLength
    
    samples = monteCarloRuns*singleRunDataSize
       
    scale = 5 #For plotting purposes
      
    inputSignal -= inputSignal.mean()
    inputSignal /= inputSignal.std()
    
    mse = []
    
    import KAF
    
    for r in range(monteCarloRuns):
        #print(signalEmbedding+r*singleRunDataSize,L+r*singleRunDataSize+trainLength,L+r*singleRunDataSize+trainLength,signalEmbedding+(r+1)*singleRunDataSize)
        u_train = np.array([inputSignal[i-signalEmbedding:i] for i in range(signalEmbedding+r*singleRunDataSize,signalEmbedding+r*singleRunDataSize+trainLength)])
        d_train = np.array([inputSignal[i] for i in range(signalEmbedding+r*singleRunDataSize,signalEmbedding+r*singleRunDataSize+trainLength)]).reshape(-1,1)
        
        u_test = np.array([inputSignal[i-signalEmbedding:i] for i in range(signalEmbedding+r*singleRunDataSize+trainLength,signalEmbedding+(r+1)*singleRunDataSize)])
        d_test = np.array([inputSignal[i] for i in range(signalEmbedding+r*singleRunDataSize+trainLength,signalEmbedding+(r+1)*singleRunDataSize)]).reshape(-1,1)
        
        if ExTest:
            y -= y.mean()
            y /= y.std()
            z -= z.mean()
            z /= z.std()
            u_train, u_test, d_train, d_test = customExogenousEmbeddingForKAFs(inputSignal, y, z, signalEmbedding, r, singleRunDataSize, trainLength)
        else:
            u_train, u_test, d_train, d_test = customEmbeddingForKAFs(inputSignal, signalEmbedding, r, singleRunDataSize, trainLength)
            
            
        if filterType == 'QKLMS':
            kafFilter = KAF.QKLMS(sigma=parameters['sigma'],epsilon=parameters['epsilon'],eta=parameters['eta'])
        elif filterType == 'QKLMS_AKB':
            kafFilter = KAF.QKLMS_AKB(sigma_init=parameters['sigma_init'],epsilon=parameters['epsilon'],eta=parameters['eta'],mu=parameters['mu'], K=int(parameters['K']))
        elif filterType == 'QKLMS_AMK':         
            kafFilter = KAF.QKLMS_AMK(epsilon=parameters['epsilon'],eta=parameters['eta'],mu=parameters['mu'], Ka=int(parameters['K']),A_init="pca")
            kafFilter.evaluate(u_train[:100],d_train[:100])
        
        mse_r = []

        for i in tqdm(range(len(u_train))):
            ui,di=u_train[i],d_train[i]
            kafFilter.evaluate(ui,di)
            
            if np.mod(i,scale)==0:            
                y_pred = kafFilter.predict(u_test)     
                err = d_test-y_pred.reshape(-1,1)
                mse_r.append(np.mean(err**2))
                
        mse.append(np.array(mse_r))
        
        signalPower = inputSignal.var()
        
        mse_mean = np.mean(np.array(mse),axis=0)/signalPower
        mse_std  = np.std(np.array(mse),axis=0)/signalPower
        
        plt.figure(figsize=(15,9))
        plt.title("Testing MSE with {} MonteCarlo Runs".format(r))
        plt.yscale("log")
        # plt.ylim((1e-2,1e1))    
        plt.fill_between(range(0,len(u_train),scale),mse_mean-mse_std,mse_mean+mse_std,alpha=0.5)
        plt.plot(range(0,len(u_train),scale),mse_mean)
        plt.ylabel("Testing MSE")
        plt.xlabel("iterations")
        #plt.savefig("Montecarlo1000/"+ "ER_lorenz" +".png", dpi = 300)
        plt.show()
    return mse_mean

def quickGridSearch4QKLMS(inputSignal, searchSize, signalEmbedding, sigmaList, epsilonList, etaList):
    #print(signalEmbedding+r*singleRunDataSize,L+r*singleRunDataSize+trainLength,L+r*singleRunDataSize+trainLength,signalEmbedding+(r+1)*singleRunDataSize)
    import numpy as np
    u_ = np.array([inputSignal[i-signalEmbedding:i] for i in range(signalEmbedding,searchSize)])
    d_ = np.array([inputSignal[i] for i in range(signalEmbedding,searchSize)]).reshape(-1,1)
    
    params = [{'eta':et,'epsilon':ep, 'sigma':s } for et in etaList for ep in epsilonList for s in sigmaList]
    # 2.2. Search over QKLMS
    results = []
    from tqdm import tqdm
    import KAF
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    
    for p in tqdm(params):
        try:
            f = KAF.QKLMS(eta=p['eta'],epsilon=p['epsilon'],sigma=p['sigma'])
            y = f.evaluate(u_,d_)
            y = np.array(y).reshape(-1,1)
            p['r2'] = r2_score(d_[1:], y[1:])
            p['testing_mse'] = mean_squared_error(d_[1:], y[1:])
            p['CB_size'] = len(f.CB)
            p['tradeOff_dist'] = tradeOffDistance(p['mse'],p['CB_size'])
        except:
            p['r2'] = np.nan
            p['testing_mse'] = np.nan
            p['CB_size'] = np.nan
            p['tradeOff_dist'] = np.nan
        results.append(p)
    import pandas as pd
    return pd.DataFrame(data=results)
          
          
def kafSearch(filterName,systemName,n_samples,trainSplit):
    import KAF
    from tqdm import tqdm
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    import pandas as pd  
    import numpy as np
    
    n_paramters = 0
    
    # 1. Generate data for grid search
    n_train = int(n_samples*trainSplit)
    n_test = n_samples - n_train
    
    if systemName == "lorenz" or systemName == "chua":
        embedding = 5
        import TimeSeriesGenerator
        x, y, z = TimeSeriesGenerator.chaoticSystem(samples=200000,systemType='lorenz')
        x -= x.mean()
        x /= x.std()
        
        u_train = np.array([x[i-embedding:i] for i in range(embedding,n_train)])
        d_train = np.array([x[i] for i in range(embedding,n_train)]).reshape(-1,1)
        
        u_test = np.array([x[i-embedding:i] for i in range(n_train,n_samples)])
        d_test = np.array([x[i] for i in range(n_train,n_samples)]).reshape(-1,1)
        
    elif systemName == "4.2":
        embedding = 2
        import testSystems as ts
        x = ts.testSystems(samples = n_samples+embedding, systemType = "4.2_AKB")
        x -= x.mean()
        x /= x.std()
        
        u_train = np.array([x[i-embedding:i] for i in range(embedding,n_train)])
        d_train = np.array([x[i] for i in range(embedding,n_train)]).reshape(-1,1)
        
        u_test = np.array([x[i-embedding:i] for i in range(n_train,n_samples)])
        d_test = np.array([x[i] for i in range(n_train,n_samples)]).reshape(-1,1)
    else:
        raise ValueError("Database does not exist")
      
    # 2. Pick filter    
    if filterName == "QKLMS":
        # 2.1. Generate parameters for QKLMS grid search
        import numpy as np
        if systemName == "lorenz":
            eta = [0.1, 0.5, 0.9]
            sigma = [1e-2, 1e-1, 0, 2, 4]
            epsilon = [1e-2, 1e-1, 1]
            
        elif systemName == "chua":
            # eta = np.linspace(0.1,0.9,n_paramters)
            # sigma = np.linspace(100,500,n_paramters)
            # epsilon = np.linspace(0.01,20,n_paramters)
            a = 0
            
        elif systemName == "4.2":
            eta = [0.1, 0.5, 0.9]
            sigma = [1e-2, 1e-1, 0, 2, 4]
            epsilon = [1e-2, 1e-1, 1]
            
        params = [{'eta':et,'epsilon':ep, 'sigma':s } for et in eta for ep in epsilon for s in sigma]
                       
        # 2.2. Search over QKLMS
        results = []
        for p in tqdm(params):
            try:
                f = KAF.QKLMS(eta=p['eta'],epsilon=p['epsilon'],sigma=p['sigma'])
                y = f.evaluate(u_train,d_train)
                y_pred = f.predict(u_test)
                err = d_test-y_pred.reshape(-1,1)
                t_mse = np.mean(err**2)
                signalPower = x.var()
                p['testing_mse'] = t_mse/signalPower
                p['CB_size'] = len(f.CB)
                p['tradeOff_dist'] = tradeOffDistance(p['testing_mse'],p['CB_size']/n_train)
            except:
                p['testing_mse'] = np.nan
                p['CB_size'] = np.nan
                p['tradeOff_dist'] = np.nan
            results.append(p)
            pd.DataFrame(data=results).to_csv('GridSearchResults2ndRun/' + filterName + '_' + systemName + '_' + str(n_samples) + '.csv')
            
    elif filterName == "QKLMS_AKB": 
        # 2.1. Generate parameters for QKLMS_AKB grid search
        import numpy as np
        if systemName == "lorenz":
            eta = [0.1, 0.9]
            sigma = [1e-2, 1e-1, 2]
            epsilon = [1e-2, 1e-1, 1]
            mu = [1e-3, 1e-1, 0.1,1]
            K = [2,5,10,20]

        elif systemName == "chua":
            eta = np.linspace(0.1,0.9,n_paramters)
            sigma = np.linspace(0.01,200,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)

        elif systemName == "4.2":
            eta = np.linspace(0.1,0.9,n_paramters)
            sigma = np.linspace(0.01,200,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)

        params = [{'eta':et,'epsilon':ep, 'sigma_init':s, 'mu':m, 'K':int(k) } for et in eta for ep in epsilon for s in sigma for m in mu for k in K]
                      
        # 2.2. Search over QKLMS
        results = []
        for p in tqdm(params):
            try:
                f = KAF.QKLMS_AKB(eta=p['eta'],epsilon=p['epsilon'],sigma_init=p['sigma_init'], mu=p['mu'], K=p['K'])
                y = f.evaluate(u_train,d_train)
                y_pred = f.predict(u_test)
                err = d_test-y_pred.reshape(-1,1)
                t_mse = np.mean(err**2)
                signalPower = x.var()
                p['testing_mse'] = t_mse/signalPower
                p['CB_size'] = len(f.CB)
                p['tradeOff_dist'] = tradeOffDistance(p['testing_mse'],p['CB_size']/n_train)
            except:
                p['testing_mse'] = np.nan
                p['CB_size'] = np.nan
                p['tradeOff_dist'] = np.nan
            results.append(p)
            pd.DataFrame(data=results).to_csv('GridSearchResults2ndRun/' + filterName + '_' + systemName + '_' + str(n_samples) + '.csv')
            
    elif filterName == "QKLMS_AMK": 
        # 2.1. Generate parameters for QKLMS_AKB grid search
        import numpy as np
        if systemName == "lorenz":
            mu = [1e-3,1e-1, 0.1,1,1.5,2]
            eta = [0.05, 0.1,0.5,0.9]
            epsilon = [1e-2, 1e-1, 1, 1.5, 2]
            K = [2,5,10,15,20]
        elif systemName == "chua":
            eta = np.linspace(0.02,1,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)
        elif systemName == "4.2":
            mu = [1e-3,1e-1, 0.1,1]
            eta = [0.05, 0.1,0.5,0.9]
            epsilon = [1e-2, 1e-1, 1, 1.5, 2]
            K = [2,5,10,15,20]
        
    
        params = [{'eta':et,'epsilon':ep, 'mu':m, 'K':int(k)} for et in eta for ep in epsilon for m in mu for k in K]
                      
        # 2.2. Search over QKLMS_AMK
        results = []
        for p in tqdm(params):
            try:
                f = KAF.QKLMS_AMK(eta=p['eta'],epsilon=p['epsilon'], mu=p['mu'], Ka=p['K'], A_init="pca")
                y = f.evaluate(u_train,d_train)
                y_pred = f.predict(u_test)
                err = d_test-y_pred.reshape(-1,1)
                t_mse = np.mean(err**2)
                signalPower = x.var()
                p['testing_mse'] = t_mse/signalPower
                p['CB_size'] = len(f.CB)
                p['tradeOff_dist'] = tradeOffDistance(p['testing_mse'],p['CB_size']/n_train)
            except:
                p['testing_mse'] = np.nan
                p['CB_size'] = np.nan
                p['tradeOff_dist'] = np.nan
            results.append(p)
            pd.DataFrame(data=results).to_csv('GridSearchResults2ndRun/' + filterName + '_' + systemName + '_' + str(n_samples) + '.csv')
            
    else:
        raise ValueError("Filter does not exist")   
    return       

def kafSearch_MC(filterName,systemName,n_samples,trainSplit,MC_runs):
    import KAF
    from tqdm import tqdm
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    import pandas as pd  
    import numpy as np
    
    n_paramters = 0
    
    # 1. Generate data for grid search
    n_train = int(n_samples*trainSplit)
    n_test = n_samples - n_train
    
    folder = 'GridSearchWang2/'
    
    attractors = ['lorenz','wang', 'rossler', 'rikitake']
    
    if systemName in attractors:
        signalEmbedding = 5
        import TimeSeriesGenerator
        inputSignal, y, z = TimeSeriesGenerator.chaoticSystem(samples=(n_samples+signalEmbedding)*MC_runs,systemType=systemName)
        inputSignal -= inputSignal.mean()
        inputSignal /= inputSignal.std()
        y -= y.mean()
        y /= y.std()
        z -= z.mean()
        z /= z.std()
               
    elif systemName == "4.1":
        signalEmbedding = 5
        import testSystems as ts
        inputSignal, targetSignal = ts.testSystems(samples = (n_samples+signalEmbedding)*MC_runs, systemType = "4.1_AKB")
        # inputSignal -= inputSignal.mean()
        # inputSignal /= inputSignal.std()
        
    elif systemName == "4.2":
        signalEmbedding = 2
        import testSystems as ts
        inputSignal = ts.testSystems(samples = (n_samples+signalEmbedding)*MC_runs, systemType = "4.2_AKB")
        inputSignal -= inputSignal.mean()
        inputSignal /= inputSignal.std()
        
    else:
        raise ValueError("Database does not exist")
      
    # 2. Pick filter    
    if filterName == "QKLMS":
        # 2.1. Generate parameters for QKLMS grid search
        if systemName in attractors:
            eta = [0.05, 0.1, 0.3, 0.5, 0.9]
            sigma = [1e-2, 1e-1, 1, 2, 4, 8]
            epsilon = [1e-1, 0.5, 1, 2]
            
        elif systemName == "chua":
            eta = [0.05, 0.1, 0.5, 0.9]
            sigma = [1e-2, 1e-1, 0, 2, 4]
            epsilon = [1e-2, 1e-1, 1,2]
            
        elif systemName == "4.1":
            eta = [0.05, 0.1, 0.5, 0.9]
            sigma = [1e-2, 1e-1, 0, 2, 4]
            epsilon = [1e-2, 1e-1, 1,2]
            
        elif systemName == "4.2":
            eta = [0.05, 0.1, 0.5, 0.9]
            sigma = [1e-2, 1e-1, 0, 2, 4]
            epsilon = [1e-2, 1e-1, 1,2]
            
        params = [{'eta':et,'epsilon':ep, 'sigma':s } for et in eta for ep in epsilon for s in sigma]
                       
        # 2.2. Search over QKLMS with Montecarlo Simulation
        results = []
        testing_mse_xRun = []
        CB_size_xRun = []
        tradeOff_dist_xRun = []
        
        signalPower = inputSignal.var()
        
        singleRunDataSize = n_samples
        trainLength = n_train
        
        for run in range(MC_runs):
            print("\n\nRunning Monte Carlo simulation #{}...\n".format(run))           
            try:
                u_train, u_test, d_train, d_test = trainAndTestSplitWithEmbedding(inputSignal, targetSignal,signalEmbedding, run, singleRunDataSize, trainLength)
            except:
                u_train, u_test, d_train, d_test = customEmbeddingForKAFs(inputSignal, signalEmbedding, run, singleRunDataSize, trainLength)
                # u_train, u_test, d_train, d_test = customExogenousEmbeddingForKAFs(inputSignal, y, z, signalEmbedding, run, singleRunDataSize, trainLength)
                
            for p in tqdm(params):
                try:
                    f = KAF.QKLMS(eta=p['eta'],epsilon=p['epsilon'],sigma=p['sigma'])
                    y = f.evaluate(u_train,d_train)
                    y_pred = f.predict(u_test)
                    err = d_test-y_pred.reshape(-1,1)
                    t_mse = np.mean(err**2)
                    testing_mse_xRun.append(t_mse/signalPower)
                    CB_size_xRun.append(len(f.CB))                  
                except:
                    testing_mse_xRun.append(np.nan)
                    CB_size_xRun.append(np.nan)
            
        allTestingMSEs = np.array(testing_mse_xRun).reshape(MC_runs,-1)
        allCB_sizes = np.array(CB_size_xRun).reshape(MC_runs,-1)
        
        testing_mse_average = np.nanmean(allTestingMSEs, axis=0)
        CB_size_average = np.nanmean(allCB_sizes, axis=0)
        
        tradeOff_distance = [tradeOffDistance(testing_mse_average[j],CB_size_average[j]/n_train) for j in range(len(params))]
        
        
        results = [p for p in params]
        results_df = pd.DataFrame(data=results)
        results_df['testing_mse'] = testing_mse_average
        results_df['CB_size'] = CB_size_average
        results_df['CB_size'] = results_df['CB_size'].astype(int)
        results_df['tradeOff_dist'] = tradeOff_distance
        
        results_df.to_csv(folder + filterName + '_' + systemName + '_' + str(n_samples) + '.csv')
        
        
        
    elif filterName == "QKLMS_AKB": 
        # 2.1. Generate parameters for QKLMS_AKB grid search
        import numpy as np
        if systemName in attractors:
            eta = [0.05, 0.1, 0.3, 0.9]
            sigma = [ 1e-1, 1, 2]
            epsilon = [1e-1, 0.5, 1, 2]
            mu = [1e-3, 1e-1, 0.1, 1]
            K = [2,5,10,20]

        elif systemName == "chua":
            eta = [0.1, 0.5, 0.9]
            sigma = [1e-2, 1e-1, 2, 4]
            epsilon = [1e-2, 1e-1, 1]
            mu = [1e-3, 1e-1, 0.1, 1]
            K = [2,5,10,20]
            
        elif systemName == "4.1":
            eta = [0.1, 0.5, 0.9]
            sigma = [1e-2, 1e-1, 2, 4]
            epsilon = [1e-2, 1e-1, 1]
            mu = [1e-3, 1e-1, 0.1, 1]
            K = [2,5,10,20]

        elif systemName == "4.2":
            eta = [0.1, 0.5, 0.9]
            sigma = [1e-2, 1e-1, 2, 4]
            epsilon = [1e-2, 1e-1, 1]
            mu = [1e-3, 1e-1, 0.1, 1]
            K = [2,5,10,20]
            

        params = [{'eta':et,'epsilon':ep, 'sigma_init':s, 'mu':m, 'K':int(k) } for et in eta for ep in epsilon for s in sigma for m in mu for k in K]
                      
            
        # 2.2. Search over QKLMS AKB with Montecarlo Simulation
        results = []
        testing_mse_xRun = []
        CB_size_xRun = []
        tradeOff_dist_xRun = []
        
        signalPower = inputSignal.var()
        
        singleRunDataSize = n_samples
        trainLength = n_train
        
        for run in range(MC_runs):
            print("\n\nRunning Monte Carlo simulation #{}...\n".format(run))
            
            try:
                u_train, u_test, d_train, d_test = trainAndTestSplitWithEmbedding(inputSignal, targetSignal,signalEmbedding, run, singleRunDataSize, trainLength)
            except:
                # u_train, u_test, d_train, d_test = customEmbeddingForKAFs(inputSignal, signalEmbedding, run, singleRunDataSize, trainLength)
                u_train, u_test, d_train, d_test = customExogenousEmbeddingForKAFs(inputSignal, y, z, signalEmbedding, run, singleRunDataSize, trainLength)
                
            for p in tqdm(params):
                try:
                    f = KAF.QKLMS_AKB(eta=p['eta'],epsilon=p['epsilon'],sigma_init=p['sigma_init'], mu=p['mu'], K=p['K'])
                    y = f.evaluate(u_train,d_train)
                    y_pred = f.predict(u_test)
                    err = d_test-y_pred.reshape(-1,1)
                    t_mse = np.mean(err**2)
                    testing_mse_xRun.append(t_mse/signalPower)
                    CB_size_xRun.append(len(f.CB))                  
                except:
                    testing_mse_xRun.append(np.nan)
                    CB_size_xRun.append(np.nan)
            
        allTestingMSEs = np.array(testing_mse_xRun).reshape(MC_runs,-1)
        allCB_sizes = np.array(CB_size_xRun).reshape(MC_runs,-1)
        
        testing_mse_average = np.nanmean(allTestingMSEs, axis=0)
        CB_size_average = np.nanmean(allCB_sizes, axis=0)
        
        tradeOff_distance = [tradeOffDistance(testing_mse_average[j],CB_size_average[j]/n_train) for j in range(len(params))]
        
        
        results = [p for p in params]
        results_df = pd.DataFrame(data=results)
        results_df['testing_mse'] = testing_mse_average
        results_df['CB_size'] = CB_size_average
        results_df['CB_size'] = results_df['CB_size'].astype(int)
        results_df['tradeOff_dist'] = tradeOff_distance
        
        results_df.to_csv(folder + filterName + '_' + systemName + '_' + str(n_samples) + '.csv')
        
    elif filterName == "QKLMS_AMK": 
        # 2.1. Generate parameters for QKLMS_AKB grid search
        import numpy as np
        if systemName in attractors:
            mu = [0.1, 0.5,1,1.5]
            eta = [0.05, 0.1, 0.3, 0.5, 0.9]
            epsilon = [1e-1, 0.5, 1, 2]
            K = [5,10,15,20]
            
        elif systemName == "chua":
            eta = np.linspace(0.02,1,n_paramters)
            epsilon = np.linspace(1e-3,100,n_paramters)
            mu = np.linspace(1e-4,1,n_paramters)
            K = np.linspace(2,20,n_paramters)
            
        elif systemName == "4.1":
            mu = [1e-3,1e-1, 0.1,1]
            eta = [0.05, 0.1,0.5,0.9]
            epsilon = [1e-2, 1e-1, 1, 1.5, 2]
            K = [2,5,10,15,20]
            
        elif systemName == "4.2":
            mu = [1e-3,1e-1, 0.1,1]
            eta = [0.05, 0.1,0.5,0.9]
            epsilon = [1e-2, 1e-1, 1, 1.5, 2]
            K = [2,5,10,15,20]
        
    
        params = [{'eta':et,'epsilon':ep, 'mu':m, 'K':int(k)} for et in eta for ep in epsilon for m in mu for k in K]
                      
        # 2.2. Search over QKLMS_AMK with Montecarlo Simulation
        results = []
        testing_mse_xRun = []
        CB_size_xRun = []
        tradeOff_dist_xRun = []
        
        signalPower = inputSignal.var()
        
        singleRunDataSize = n_samples
        trainLength = n_train
              
        for run in range(MC_runs):
            print("\n\n Running Monte Carlo simulation #{}...\n".format(run))
            try:
                u_train, u_test, d_train, d_test = trainAndTestSplitWithEmbedding(inputSignal, targetSignal,signalEmbedding, run, singleRunDataSize, trainLength)
            except:
                u_train, u_test, d_train, d_test = customEmbeddingForKAFs(inputSignal, signalEmbedding, run, singleRunDataSize, trainLength)
                # u_train, u_test, d_train, d_test = customExogenousEmbeddingForKAFs(inputSignal, y, z, signalEmbedding, run, singleRunDataSize, trainLength)
                        
            for p in tqdm(params):
                try:
                    f = KAF.QKLMS_AMK(eta=p['eta'],epsilon=p['epsilon'], mu=p['mu'], Ka=p['K'], A_init="pca")
                    y = f.evaluate(u_train,d_train)
                    y_pred = f.predict(u_test)
                    err = d_test-y_pred.reshape(-1,1)
                    t_mse = np.mean(err**2)
                    testing_mse_xRun.append(t_mse/signalPower)
                    CB_size_xRun.append(len(f.CB))                  
                except:
                    testing_mse_xRun.append(np.nan)
                    CB_size_xRun.append(np.nan)
            
        allTestingMSEs = np.array(testing_mse_xRun).reshape(MC_runs,-1)
        allCB_sizes = np.array(CB_size_xRun).reshape(MC_runs,-1)
        
        testing_mse_average = np.nanmean(allTestingMSEs, axis=0)
        CB_size_average = np.nanmean(allCB_sizes, axis=0)
        
        tradeOff_distance = [tradeOffDistance(testing_mse_average[j],CB_size_average[j]/n_train) for j in range(len(params))]
        
        
        results = [p for p in params]
        results_df = pd.DataFrame(data=results)
        results_df['testing_mse'] = testing_mse_average
        results_df['CB_size'] = CB_size_average
        results_df['CB_size'] = results_df['CB_size'].astype(int)
        results_df['tradeOff_dist'] = tradeOff_distance
        
        results_df.to_csv(folder + filterName + '_' + systemName + '_' + str(n_samples) + '.csv')   
    else:
        raise ValueError("Filter does not exist")   
    return       

def db(samples=1000,system='lorenz',L=40):
    import numpy as np
    import TimeSeriesGenerator as tsg
    x, y, z = tsg.chaoticSystem(samples=samples,systemType=system)
    ux = np.array([x[i-L:i] for i in range(L,len(x))])
    uy = np.array([y[i-L:i] for i in range(L,len(y))])
    u = np.concatenate((ux,uy), axis=1) # INPUT
    d = np.array([z[i] for i in range(L,len(z))]).reshape(-1,1)
    return u,d

def db2(samples=1000):
    import numpy as np
    import testSystems as ts
    var = 0.01
    noise = np.sqrt(var)*np.random.randn(samples)
    s = ts.testSystems(samples = samples+2, systemType = "4.2_AKB")
    u = np.array([s[-samples-1:-1],s[-samples-2:-2]]).T
    d = np.array(s[-samples:]).reshape(-1,1) + noise.reshape(-1,1)
    return u,d

def trainAndTestSplitWithEmbedding(inputSignal, targetSignal, signalEmbedding, run, singleRunDataSize, trainLength):
    import numpy as np
    u_train = np.array([inputSignal[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize,signalEmbedding+run*singleRunDataSize+trainLength)])
    d_train = np.array([targetSignal[i] for i in range(signalEmbedding+run*singleRunDataSize,signalEmbedding+run*singleRunDataSize+trainLength)]).reshape(-1,1)
            
    u_test = np.array([inputSignal[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize+trainLength,signalEmbedding+(run+1)*singleRunDataSize)])
    d_test = np.array([targetSignal[i] for i in range(signalEmbedding+run*singleRunDataSize+trainLength,signalEmbedding+(run+1)*singleRunDataSize)]).reshape(-1,1)
    return u_train, u_test, d_train, d_test

def customEmbeddingForKAFs(inputSignal, signalEmbedding, run, singleRunDataSize, trainLength):
    import numpy as np
    u_train = np.array([inputSignal[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize,signalEmbedding+run*singleRunDataSize+trainLength)])
    d_train = np.array([inputSignal[i] for i in range(signalEmbedding+run*singleRunDataSize,signalEmbedding+run*singleRunDataSize+trainLength)]).reshape(-1,1)
            
    u_test = np.array([inputSignal[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize+trainLength,signalEmbedding+(run+1)*singleRunDataSize)])
    d_test = np.array([inputSignal[i] for i in range(signalEmbedding+run*singleRunDataSize+trainLength,signalEmbedding+(run+1)*singleRunDataSize)]).reshape(-1,1)
    return u_train, u_test, d_train, d_test

def customExogenousEmbeddingForKAFs(inputSignal, ExY, ExZ,signalEmbedding, run, singleRunDataSize, trainLength):
    import numpy as np
    input_train = np.array([inputSignal[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize,signalEmbedding+run*singleRunDataSize+trainLength)])        
    input_test = np.array([inputSignal[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize+trainLength,signalEmbedding+(run+1)*singleRunDataSize)])
    
    ExY_train = np.array([ExY[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize,signalEmbedding+run*singleRunDataSize+trainLength)])        
    ExY_test = np.array([ExY[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize+trainLength,signalEmbedding+(run+1)*singleRunDataSize)])
    
    ExZ_train = np.array([ExZ[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize,signalEmbedding+run*singleRunDataSize+trainLength)])        
    ExZ_test = np.array([ExZ[i-signalEmbedding:i] for i in range(signalEmbedding+run*singleRunDataSize+trainLength,signalEmbedding+(run+1)*singleRunDataSize)])
    
    u_train = np.concatenate((input_train,ExY_train,ExZ_train),axis=1)
    u_test= np.concatenate((input_test,ExY_test,ExZ_test),axis=1)
    
    d_train = np.array([inputSignal[i] for i in range(signalEmbedding+run*singleRunDataSize,signalEmbedding+run*singleRunDataSize+trainLength)]).reshape(-1,1)
    d_test = np.array([inputSignal[i] for i in range(signalEmbedding+run*singleRunDataSize+trainLength,signalEmbedding+(run+1)*singleRunDataSize)]).reshape(-1,1)
    return u_train, u_test, d_train, d_test    

def tradeOffDistance(MSE,sizeCB):
    from scipy.spatial.distance import cdist
    import numpy as np
    reference = np.array([0,0]).reshape(1,-1)
    result = np.array([MSE,sizeCB]).reshape(1,-1)
    return cdist(reference,result).item()

def selectBestResultFromKafSearch(resultsDataFrame):
    import pandas as pd
    results = pd.read_csv(resultsDataFrame).dropna(axis=0) 
    return results[results.tradeOff_dist == results.tradeOff_dist.min()].iloc[0]

def codebook4KAF(inputSignal, signalEmbedding, N, trainSplit, gridSearchPath, filterType,ExTest=False,y=None,z=None):
    parameters = selectBestResultFromKafSearch(gridSearchPath)
    
    if ExTest:
        u_train, u_test, d_train, d_test = customExogenousEmbeddingForKAFs(inputSignal, y, z, signalEmbedding, 0, len(inputSignal), int(len(inputSignal)*trainSplit))
    else:
        u_train, u_test, d_train, d_test = customEmbeddingForKAFs(inputSignal, signalEmbedding, 0, N, int(N*trainSplit))
    
    import KAF
    
    if filterType == 'QKLMS':
        kafFilter = KAF.QKLMS(sigma=parameters['sigma'],epsilon=parameters['epsilon'],eta=parameters['eta'])
    elif filterType == 'QKLMS_AKB':
        kafFilter = KAF.QKLMS_AKB(sigma_init=parameters['sigma_init'],epsilon=parameters['epsilon'],eta=parameters['eta'],mu=parameters['mu'], K=int(parameters['K']))
    elif filterType == 'QKLMS_AMK':         
        kafFilter = KAF.QKLMS_AMK(epsilon=parameters['epsilon'],eta=parameters['eta'],mu=parameters['mu'], Ka=int(parameters['K']),A_init="pca")
        kafFilter.evaluate(u_train[:100],d_train[:100])
    
    kafFilter.evaluate(u_train,d_train)
    return [filterType, len(kafFilter.CB)]
    