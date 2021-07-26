import numpy as np
"""
Make the embedding for KAF processing
"""
def Embedder(X, embedding = 2):     
    u = np.array([X[i-embedding:i] for i in range(embedding,len(X))])
    d = np.array([X[i] for i in range(embedding,len(X))]).reshape(-1,1)
    return u,d

def TrainTestSplit(u, d, train_portion=0.8):
    train_slice = slice(0, int(len(u)*train_portion))
    test_slice = slice(int(len(u)*train_portion),-1)
    return [u[train_slice], u[test_slice], d[train_slice], d[test_slice]]

def z_scorer(system):
    system -= system.mean()
    system /= system.std()
    return system

def parameter_picker(filt, grid):
    try:
        params = {
            "QKLMS" : [{'eta':et,'epsilon':ep, 'sigma':s } for et in grid['eta'] for ep in grid['epsilon'] for s in grid['sigma']],
            "QKLMS_AKB": [{'eta':et,'epsilon':ep, 'sigma_init':s, 'mu':m, 'K':int(k)} for et in grid['eta'] for ep in grid['epsilon'] for s in grid['sigma'] for m in grid['mu'] for k in grid['K']],
            "QKLMS_AMK": [{'eta':et,'epsilon':ep, 'mu':m, 'K':int(k)} for et in grid['eta'] for ep in grid['epsilon'] for m in grid['mu'] for k in grid['K']]
        }
    except:
        raise ValueError("Parameter asignation for {} failed".format(filt))
    return params[filt]
    
def KAF_picker(filt, params):
    try:
        import KAF
        filers = {
            "QKLMS":KAF.QKLMS(eta=params['eta'],epsilon=params['epsilon'],sigma=params['sigma']),
            "QKLMS_AKB":KAF.QKLMS_AKB(eta=params['eta'],epsilon=params['epsilon'],sigma_init=params['sigma_init'], mu=params['mu'], K=params['K']),
            "QKLMS_AMK":KAF.QKLMS_AMK(eta=params['eta'],epsilon=params['epsilon'], mu=params['mu'], Ka=params['K'], A_init="pca")
        }
    except: 
        raise ValueError("Filter definition for {} failed".format(filt))

def tradeOff(TMSE,CB):
    from scipy.spatial.distance import cdist
    import numpy as np
    reference = np.array([0,0]).reshape(1,-1)
    result = np.array([TMSE,CB]).reshape(1,-1)
    return cdist(reference,result).item()