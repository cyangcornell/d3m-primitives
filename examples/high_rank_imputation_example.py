# demo of recovering the missing entries of incomplete matrix of multiple-subspace data

import sys
sys.path.append('..')
import numpy as np
from d3m.container import DataFrame
import high_rank_imputer

# define m,n,r for synthetic data matrix
K=5 # number of subspaces 
m = 50 # row dimension
n = 50 # column dimension
r = 5  # matrix rank (latent dimension), r<<min(m,n)
missing_rate = 0.5 # proportion of unknown entries, within [0.1,0.9]

# generate random matrix
A = np.random.randn(m,r)
B = np.random.randn(r,n)
X_org = np.dot(A,B)
if K>1:
    for k in range(K-1):
        A = np.random.randn(m,r)
        B = np.random.randn(r,n)
        X_org = np.append(X_org,np.dot(A,B),axis=1)    

# mask a fraction of entries randomly
X_incomplete=X_org.copy()
m,n=X_org.shape
for i in range(n):
    idx=np.random.choice(m,int(np.round(m*missing_rate)), replace=False)
    X_incomplete[idx,i]=np.nan  

# recover the missing entries
# hp= hrmc_sf.Hyperparams.defaults()
hp=high_rank_imputer.Hyperparams({'d':0,'alpha':1,'beta':1,'tol':1e-4,'maxiter':500})
# if d=0, d will be automatically estimated; otherwise (d>=1), the value of d will be applied 
sf=high_rank_imputer.HighRankImputer(hyperparams=hp)
df_incomplete = DataFrame(X_incomplete.T)
# the missing entries in the matrix must be noted by NaN
df_recovered=sf.produce(inputs=df_incomplete).value

X_recovered = df_recovered.values.T

# compute the recovery error (relative mean squared error, within [0,1], the smaller the better)
RMSE=np.square(X_recovered-X_org).sum()/np.square(X_org).sum()
print("RMSE:", RMSE)


