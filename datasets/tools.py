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