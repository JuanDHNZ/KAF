import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
import matplotlib as mpl
import tikzplotlib
mpl.rcParams['figure.dpi'] = 300
import KAF
# from KAF.datasets.ChaoticTimeSeries import GenerateAttractor

"""
Make the embedding for KAF processing
"""
def Embedder(X, embedding = 2):     
    u = np.array([X[i-embedding:i] for i in range(embedding,len(X))])
    d = np.array([X[i] for i in range(embedding,len(X))]).reshape(-1,1)
    return u,d

def mc_sampler(X, n_samples, mc_runs, embedding = 2):
    
    embed = np.arange(embedding+1).reshape(1,1,-1)
    samples = np.arange(n_samples).reshape(1,-1,1)    
    mc = np.arange(0,len(X),n_samples+embedding+1)[:mc_runs].reshape(-1,1,1)
    
    indices = mc+samples+embed   
        
    return X[indices]
    
    
    # X = X.tolist()
    # run_samples = n_samples + embedding + 1
    # X_mc = []
    # for run in range(mc_runs):
    #     run_slice = slice(0, run_samples)
    #     X_mc.append(X[run_slice])
    #     X = X[run_samples:]
    # return np.array(X_mc)
    

def TrainTestSplit(u, d, train_portion=0.8):
    train_slice = slice(0, int(len(u)*train_portion))
    test_slice = slice(int(len(u)*train_portion),-1)
    return [u[train_slice], u[test_slice], d[train_slice], d[test_slice]]

def z_scorer(x):
    system = x.copy()
    system -= system.mean()
    system /= system.std()
    return system

def grid_picker(kaf):
    try:
        grids = {
            "QKLMS" : {
                "eta":[0.05, 0.1, 0.2],
                "epsilon":[0.35, 0.5, 1],
                "sigma": [0.2, 0.4, 0.6]
            },
            "QKLMS_AKB": {
                "eta":[0.05, 0.1, 0.2],
                "epsilon":[0.35, 0.5, 1],
                "sigma_init": [0.2, 0.4, 0.6],
                "mu":[0.2 , 0.4, 0.6],
                "K":[1,2,4,6,8]
            },
            "QKLMS_AMK": {
                "eta":[0.05, 0.1, 0.2],
                "eta":[0.3, 0.5, 0.9],
                "epsilon":[0.35, 0.5, 1],
                "mu":[0.05, 0.1, 0.2 , 0.3, 0.4],
                "K":[1,2,4,6,8]
                # "eta":[0.5],
                # "epsilon":[0.05, 0.1, 0.2],
                # "mu":[0.0001, 0.0005, 0.001],
                # "K":[20, 22, 26, 28, 32]
            }
        }
        return grids[kaf]
    except:
        raise ValueError("Grid definition for {} failed".format(kaf))

def parameter_picker(filt, grid):
    try:
        if filt == "QKLMS":
            params = [{'eta':et,'epsilon':ep, 'sigma':s } for et in grid['eta'] for ep in grid['epsilon'] for s in grid['sigma']]
        elif filt == "QKLMS_AKB": 
            params = [{'eta':et,'epsilon':ep, 'sigma_init':s, 'mu':m, 'K':int(k)} for et in grid['eta'] for ep in grid['epsilon'] for s in grid['sigma_init'] for m in grid['mu'] for k in grid['K']]
        elif filt == "QKLMS_AMK": 
            params = [{'eta':et,'epsilon':ep, 'mu':m, 'K':int(k)} for et in grid['eta'] for ep in grid['epsilon'] for m in grid['mu'] for k in grid['K']]
        return params
    except:
        raise ValueError("Parameter asignation for {} failed".format(filt))
    
    
def KAF_picker(filt, params):
    try:
        if filt == "QKLMS":
            kaf_filt = KAF.QKLMS(eta=params['eta'],epsilon=params['epsilon'],sigma=params['sigma'])
        elif filt == "QKLMS_AKB":
            kaf_filt = KAF.QKLMS_AKB(eta=params['eta'],epsilon=params['epsilon'],sigma_init=params['sigma_init'], mu=params['mu'], K=params['K'])
        elif filt == "QKLMS_AMK":
            kaf_filt = KAF.QKLMS_AMK(eta=params['eta'],epsilon=params['epsilon'], mu=params['mu'], Ka=params['K'], A_init="pca")
        elif filt == "QKLMS_AKS":
            kaf_filt = KAF.QKLMS_AKS(eta=params['eta'],epsilon=params['epsilon'], mu=params['mu'], sigma=params['sigma'])
        elif filt == "QKLMS_MIPV":
            kaf_filt = KAF.QKLMS_MIPV(eta=params['eta'],epsilon=params['epsilon'],sigma=params['sigma'])
        return kaf_filt
    except: 
        raise ValueError("Filter definition for {} failed".format(filt))
        
def best_params_picker(filt, params_df, criteria='CB'): # CHANGE CRITERIA FOR TMSE CURVE
    best_params = params_df[params_df[criteria] == params_df[criteria].min()]
    
    if len(best_params.index) > 1 and criteria=='CB_median':
        best_params = best_params[best_params['TMSE']==best_params['TMSE'].min()]
    
    if filt == "QKLMS":
        bps = {'eta':best_params.eta.values[0],
                'epsilon':best_params.epsilon.values[0],
                'sigma':best_params.sigma.values[0]}
        # bps = {'eta':0.2,
        #        'epsilon':0.1,
        #        'sigma':0.35}
        
    elif filt == "QKLMS_AKB":
        bps = {'eta':best_params.eta.values[0],
               'epsilon':best_params.epsilon.values[0],
               'sigma_init':best_params.sigma_init.values[0], 
               'mu':best_params.mu.values[0], 
               'K':best_params.K.values[0]}
    elif filt == "QKLMS_AMK":
        bps = {'eta':best_params.eta.values[0],
                'epsilon':best_params.epsilon.values[0],
                'mu':best_params.mu.values[0],
                'K':best_params.K.values[0], 'A_init':"pca"}
                # 'K':6, 'A_init':"pca"}
    elif filt == "QKLMS_AKS":
        bps = {'eta':0.05,
               'epsilon':0.35,
               'sigma':0.4,
               'mu':0.01}
        

    return bps

def tradeOff(TMSE,CB):
    from scipy.spatial.distance import cdist
    import numpy as np
    reference = np.array([0,0]).reshape(1,-1)
    result = np.array([TMSE,CB]).reshape(1,-1)
    return cdist(reference,result).item()

def plotCB(model,X,savename="test"): 
    means = np.array(model.CB)
    covs = [np.linalg.inv(np.dot(A.T,A)) for A in model.At]

    fig, ax = plt.subplots()
    ax.scatter(means[:, 0], means[:, 1], s=20, color='red', marker="X", label="CB_centroid")
    ax.scatter(X[:, 0], X[:, 1], s=10, color='blue', marker="x", label="Samples")
    plt.ylim([-6,6])
    plt.xlim([-6,6])
    plt.title("CB = {}".format(len(model.CB)))
    # # plot_gmm(gmm, u)
    for mean, cov in zip(means, covs):
        confidence_ellipse(cov=cov, mean=mean, ax=ax, n_std=1, edgecolor='red')
    plt.legend()
    folder = "Graficos/4.2v2/CB/TMSE/"
    plt.savefig(folder + '{}.png'.format(savename), dpi=300)
    tikzplotlib.save(folder + 'tex/{}.tex'.format(savename))
    plt.show()
    return

def plotCB_AKB(model,X,savename="test"): 
    means = np.array(model.CB)
    covs = [(model.sigma**2)*np.eye(2) for n in range(len(model.CB))]
    # covs = [np.linalg.inv(np.dot(A.T,A)) for A in model.At]

    fig, ax = plt.subplots()
    ax.scatter(means[:, 0], means[:, 1], s=20, color='red', marker="X", label="CB_centroid")
    ax.scatter(X[:, 0], X[:, 1], s=10, color='blue', marker="x", label="Samples")
    plt.ylim([-6,6])
    plt.xlim([-6,6])
    plt.title("CB = {}  |  $\sigma$ = {}".format(len(model.CB),model.sigma))
    # # plot_gmm(gmm, u)
    for mean, cov in zip(means, covs):
        confidence_ellipse(cov=cov, mean=mean, ax=ax, n_std=1, edgecolor='red')
    plt.legend()
    plt.axis("equal")
    plt.ylim([-6,6])
    plt.xlim([-6,6])
    folder = "Graficos/4.2v2/CB/TMSE/"
    plt.savefig(folder + '{}.png'.format(savename), dpi=300)
    tikzplotlib.save(folder + 'tex/{}.tex'.format(savename))
    plt.show()
    return

def confidence_ellipse(cov, mean, ax, n_std=3.0, facecolor='none', edgecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    
def noisy_chua_generator(n_samples, seed=None, alpha = None, beta = None, noise=True):
    #parameter generation
    import sys
    import random
    var = 1
    if seed == None:
        seed_train = random.randint(0, 2000)
        seed_test = random.randint(0, 2000)
    
    if alpha == None:
        alpha_train = random.uniform(13.6,17.6)
        alpha_test = random.uniform(13.6,17.6)
    else:
        alpha_train = alpha_test = alpha
    if beta == None:
        beta_train= random.uniform(26,30)
        beta_test = random.uniform(26,30)
    else:
        beta_train = beta_test = beta
    
    dic1 = {'noise':noise, 'seed':seed_train, 'alpha':alpha_train, 'beta':beta_train, 'noise_var':var}
    train,_,_ = GenerateAttractor(samples=n_samples, attractor='chua', **dic1)
    
    seed = random.randint(0, 2000)
    # print(seed)
    alpha = random.uniform(13.6,17.6)
    beta = random.uniform(26,30)
    dic2 = {'noise': noise, 'seed':seed_test, 'alpha':alpha_test, 'beta':beta_test, 'noise_var':var}
    test,_,_ = GenerateAttractor(samples=n_samples, attractor='chua', **dic2)

    return train, test, dic1, dic2

def noisy_chua_splited(n_samples, seed=None, alpha = None, beta = None, noise=True):
    import random
    var = 1
    if seed == None:
        seed = random.randint(0, 2000)
    params = {'noise':noise, 'seed':seed, 'alpha':alpha, 'beta':beta, 'noise_var':var}
    return GenerateAttractor(samples=n_samples, attractor='chua', **params)

    
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
    