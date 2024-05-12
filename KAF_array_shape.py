import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import KAF


import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from datasets.tools import z_scorer,mc_sampler
from datasets.TestingSystems import GenerateSystem
from datasets.TestingSystems import generateRandomVectorModel
from datasets.ChaoticTimeSeries import GenerateAttractor

#%%
# Tunnig process
from tqdm import tqdm_notebook as tqn
def params_grid_organizer(model, params):
    try:
        if isinstance(model, KAF.QKLMS):
            grid = [{'eta':et,'epsilon':ep, 'sigma':s } for et in params['eta'] for ep in params['epsilon'] for s in params['sigma']]
        elif isinstance(model, KAF.QKLMS_AKB):
            grid = [{'eta':et,'epsilon':ep, 'sigma_init':s, 'mu':m, 'K':int(k)} for et in params['eta'] for ep in params['epsilon'] for s in params['sigma_init'] for m in params['mu'] for k in params['K']]
        elif isinstance(model, KAF.QKLMS_varIP_FC):
            grid = [{'eta':et,'epsilon':ep, 'sigma':s, 'FC':fc, 'bounds':bnds} for et in params['eta'] for ep in params['epsilon'] for s in params['sigma'] for fc in params['FC'] for bnds in params['bounds']]
        elif isinstance(model, KAF.QKLMS_AMK):
           grid = [{'eta':et,'epsilon':ep, 'sigma':s, 'mu':m, 'K':int(k)} for et in params['eta'] for ep in params['epsilon'] for s in params['sigma'] for m in params['mu'] for k in params['K']]
        return grid
    except:
        raise ValueError("Parameter assignation for {} failed".format(filt))

def model_set_params(model, params):
  if isinstance(model, KAF.QKLMS):
    model.sigma = params['sigma']
    model.epsilon = params['epsilon']
    model.eta = params['eta']
  elif isinstance(model, KAF.QKLMS_AKB):
    model.sigma_init = params['sigma_init']
    model.epsilon = params['epsilon']
    model.eta = params['eta']
    model.mu = params['mu']
    model.K = params['K']
    model.sigma_n = [params['sigma_init']]
    model.sigma = params['sigma_init']
  elif isinstance(model,KAF.QKLMS_varIP_FC):
    model.sigma_init = params['sigma']
    model.epsilon = params['epsilon']
    model.eta = params['eta']
    model.FC = params['FC']
    model.bounds = params['bounds']
  elif isinstance(model, KAF.QKLMS_AMK):
    model.sigma = params['sigma']
    model.epsilon = params['epsilon']
    model.eta = params['eta']
    model.mu = params['mu']
    model.Ka = params['K']
  else:
    raise ValueError('Model not supported')
  return model

def best_params_finder(model, X, y, params):
  class filt(type(model)): # model safe copy
    pass
  results = []
  param_grid = params_grid_organizer(model, params)
  for p in tqn(param_grid):
    run_model = model_set_params(filt(), p)
    y_pred = run_model.evaluate(X,y)
    p['MSE'] = MSE(y,y_pred)
    p['MAE'] = MAE(y,y_pred)
    p['MAPE'] = MAPE(y,y_pred)
    p['final_CB'] = len(run_model.CB)
    results.append(p)
  return results

def best_params_finder_split(model, Xtrain, ytrain, Xtest, ytest,params):
  class filt(type(model)): # model safe copy
    pass
  results = []
  param_grid = params_grid_organizer(model, params)

  for p in tqn(param_grid):
    run_model = model_set_params(filt(), p)
    if  isinstance(model, KAF.QKLMS_AMK):
      run_model.evaluate(Xtrain[:100], ytrain[:100])
    y_pred = run_model.evaluate(Xtrain,ytrain)
    y_pred_train = run_model.predict(Xtest)
    # print(y_pred_train.shape, ytest.shape)
    p['MSE'] = MSE(ytest,y_pred_train.reshape(-1,1))
    p['MAE'] = MAE(ytest,y_pred_train.reshape(-1,1))
    p['MAPE'] = MAPE(ytest,y_pred_train.reshape(-1,1))
    p['R2'] = R2(ytest,y_pred_train.reshape(-1,1))
    p['final_CB'] = len(run_model.CB)
    cb_size_std = len(run_model.CB)/len(Xtrain)
    p['toff_MSE'] = dist_toff(p['MSE'], cb_size_std)
    p['toff_MAE'] = dist_toff(p['MAE'], cb_size_std)
    p['toff_MAPE'] = dist_toff(p['MAPE'], cb_size_std)
    p['toff_R2'] = dist_toff_r2(p['R2'], cb_size_std)
    results.append(p)
  return results

# Metrics
def MSE(y_true, y_pred):
    err = y_true-y_pred.reshape(-1,1)
    return np.mean(err**2)

def MAE(y_true, y_pred):
    err = y_true-y_pred.reshape(-1,1)
    return np.mean(abs(err))

def MASE(y_true, y_pred):
    err = y_true-y_pred.reshape(-1,1)
    n = len(y_true)
    num = abs(err)
    den = abs(np.diff(y_true, axis=0)).sum()/(n-1)
    return np.mean(num)/den

def MAPE(y_true,y_pred):
    err = y_true-y_pred.reshape(-1,1)
    ape = (abs(err)/abs(y_true)).sum()
    return np.mean(ape)

def APE(y_true,y_pred):
    err = y_true-y_pred.reshape(-1,1)
    return (abs(err)/abs(y_true))

def dist_toff(error, CB_size_std):
  from scipy.spatial.distance import cdist
  return cdist(np.array([error,CB_size_std]).reshape(1,-1), np.array([0, 0]).reshape(1,-1)).item()


def dist_toff_r2(error, CB_size_std):
  from scipy.spatial.distance import cdist
  return cdist(np.array([error,CB_size_std]).reshape(1,-1), np.array([1, 0]).reshape(1,-1)).item()

def R2(y_true,y_pred):
  from sklearn.metrics import r2_score
  return r2_score(y_true,y_pred)
#%%

samples = 2200
embedding=2

system = GenerateSystem(samples=samples+embedding, systemType='4.2_AKB')
system = z_scorer(system)

system_emb = mc_sampler(system, samples , 1, embedding=embedding)
X,y = system_emb[0,:,:-1],system_emb[0,:,-1]

train_portion=2000/2200
train_size = int(samples*train_portion)
Xtrain,ytrain = system_emb[0,:train_size,:-1],system_emb[0,:train_size,-1].reshape(-1,1)
Xtest,ytest = system_emb[0,train_size:,:-1],system_emb[0,train_size:,-1].reshape(-1,1)
print("train ", Xtrain.shape,ytrain.shape)
print("test ", Xtest.shape,ytest.shape)

#%%

#tunning
filt = KAF.QKLMS()
params = {'eta':[0.1, 0.5, 0.9], 'epsilon': [0.1, 0.2, 0.3, 0.6], 'sigma': [0.05, 0.1, 0.15, 0.2, 0.35]}
search_results = best_params_finder_split(filt, Xtrain, ytrain, Xtest, ytest, params)
results = pd.DataFrame(data=search_results)
results.sort_values(by=['MAPE']).head()