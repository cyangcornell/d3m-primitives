import os.path
from typing import Union,Dict

import numpy as np
import pandas as pd
from d3m import container
from d3m.container import ndarray
from d3m import utils

from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult,DockerContainer

Inputs = container.DataFrame
Outputs = container.DataFrame


__author__ = 'Cornell'
__version__ = 'v0.1.1'

class Params(params.Params):
    X: ndarray 
class Hyperparams(hyperparams.Hyperparams):
    d = hyperparams.Hyperparameter(default=0,
                                   description='The reduced dimension of matrix factorization. It is usually smaller than the sizes of the matrix. When setting to 0, d will be automatically roughly estimated.',
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    tol = hyperparams.Hyperparameter(default=1e-4,
                                     description='The tolerance of the relative changes of the variables in optimization. It will be utilized to check the convergence.',
                                     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    maxiter = hyperparams.Hyperparameter(default=200,
                                         description='The maximum number of iterations.',
                                         semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    alpha = hyperparams.Hyperparameter(default=1.0,
                                       description='The regularization parameter for the factorized dense matrix A.',
                                       semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    beta = hyperparams.Hyperparameter(default=1.0,
                                      description='The regularization parameter for the factorized sparse matrix Z.',
                                      semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

class HighRankImputer(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    """
    This primitive imputes a dataset in which data points are drawn from multiple subspaces, which in pratice means the data have mutiple groups/classes. In such cases, the data matrices are often of high-rank. In such cases, Sparse Factorization based Matrix Completion (SFMC) can outperform classical low-rank matrix completion methods.
    The optimization is solved via accelerated proximal alternating minimization (APALM). The NaNs in the input matrix will be regarded as missing entries. The algorithm will recover the missing entries and return the recovered matrix as output.
    In this primitive, the method is used for collaborative filtering (recommendation system).
    """

    metadata = metadata_base.PrimitiveMetadata({
        'id': 'e6ee30fa-af68-4bfe-9234-5ca7e7ac8e93',
        'version': __version__,
        'name': "Matrix Completion via Sparse Factorization",
        'keywords': ['Matrix completion', 'low-rank matrix', 'high-rank matrix', 'sparse factorization',],
        'source': {
            'name': __author__,
            'contact': 'mailto:cy438@cornell.edu',
           'uris': [
                    'https://github.com/cyangcornell/d3m-primitives.git',
                    ],
        },
       'installation': [{
                        'type': 'PIP',
                        'package_uri': 'git+https://github.com/cyangcornell/d3m-primitives.git@{git_commit}#egg=pyglrm-d3m'.format(git_commit=utils.current_git_commit(os.path.dirname(__file__)))
                        
                        }],
        'python_path': 'd3m.primitives.collaborative_filtering.high_rank_imputer.Cornell',

        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.LOW_RANK_MATRIX_APPROXIMATIONS,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.COLLABORATIVE_FILTERING,
    })

    def __init__(self, *, hyperparams: Hyperparams, docker_containers: Dict[str,DockerContainer] = None, _versbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, docker_containers = docker_containers)
        
        self.d: int = hyperparams['d']
        self.tol: float = hyperparams['tol']
        self.maxiter: int = hyperparams['maxiter']
        self.alpha: float = hyperparams['alpha']
        self.beta: float = hyperparams['beta']
            

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._fitted = False
        self._keys = list(outputs)
        self._fitted = False
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted: 
            return

        x_rating=self._training_inputs.copy()
        x_rating[self._training_outputs.columns[0]]=self._training_outputs.values
        X_incomplete=x_rating.pivot(index=x_rating.columns[0], columns=x_rating.columns[1], values=x_rating.columns[2])
        
        tol=self.tol
        maxiter=self.maxiter
        
        X=X_incomplete.values.copy()
       # X=X_incomplete.values
        #print(X)
       # print(X_incomplete)
        m0,n0 = X.shape
        if m0>n0:
            X = X.T
        
        m,n = X.shape  
          
        
        M=np.ones([m,n])
        M[np.isnan(X)]=0
        sr=M.sum()/m/n
        X[np.isnan(X)]=0 
        
        
        if self.d==0:
            if sr>0.5:
                d=np.int(np.round(0.25*min(m,n)))
            else:
                d=np.int(4*np.round(sr*min(m,n)))
            #print('The auto-estimated latent dimension is %d.' % d)
        else:
            d=self.d
            #print('The user-defined latent dimension is %d.' % d)
         
        alpha=self.alpha*np.sqrt(max(m,n)/d)*2
        beta=self.beta*np.sqrt(max(m,n)/d)
        
        A=np.random.randn(m,d)
        Z=np.zeros((d,n))
        rho=max(1.5*np.sqrt(M.mean()),0.5)

        iter=0
        cc=0.5

        while iter<self.maxiter:
            
            iter=iter+1
            
            # Z_new
            if iter==1:
                Z=Z
            else:
                Z=Z_new+cc*(Z_new-Z_old)

            tau=rho*np.linalg.norm(np.dot(A.T,A),2)
            G=Z-(-np.dot(A.T,np.multiply(M,X-np.dot(A,Z))))/tau
            Z_new=np.maximum(0,G-beta/tau)+np.minimum(0,G+beta/tau)
            
            # A_new
            if iter==1:
                A=A
            else:
                A=A_new+cc*(A_new-A_old)

            kai=rho*np.linalg.norm(np.dot(Z_new,Z_new.T),2)
            H=A+np.dot(np.multiply(M,X-np.dot(A,Z_new)),Z_new.T)/kai;
            A_new=1/(alpha+kai)*H*kai;

            # check convergence
            stopC=max(np.linalg.norm(Z_new-Z,'fro')/np.linalg.norm(Z_new,'fro'),np.linalg.norm(A_new-A,'fro')/np.linalg.norm(A_new,'fro'))
            isstopC=stopC<tol

            """ 
            if np.mod(iter,10)==0 or isstopC or iter==1:
                obj_func=0.5*np.linalg.norm(np.multiply(M,X-np.dot(A_new,Z_new)),'fro')**2+0.5*alpha*np.linalg.norm(A_new,'fro')**2+beta*abs(Z_new).sum()
                print('iteration: %d/%d, obj_func=%f, var_change=%f.' % (iter,maxiter,obj_func,stopC))
            """

            if isstopC:
                Z=Z_new;
                A=A_new;
                #print('converged!')
                break

            Z_old=Z
            A_old=A
            Z=Z_new
            A=A_new
            
    #    D=abs(Z)>1e-5
   #     D=D+0
        
        X_temp=np.multiply(X,M)+np.multiply(np.dot(A,Z),1-M)
        if m0>n0:
            X_temp = X_temp.T
            
        #print('The proportion of nonzero entries in the sparse coefficient matrix is %f.' % (D.sum()/D.shape[0]/D.shape[1]))
        self._X=pd.DataFrame(X_temp,X_incomplete.index,X_incomplete.columns)               
        
        self._fitted = True
        return CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        
        Xp = self._X
        testData = inputs
        y_pred=np.zeros(testData.shape[0])#+Xp.values.mean()
        idr=testData[testData.columns[0]].isin(Xp.index)
        idc=testData[testData.columns[1]].isin(Xp.columns)
        dd=np.where(idr&idc==True)
        #print(type(X.columns[1]))
        #print(X.columns.get_loc(np.str(5)))
       # ee=np.where(testData[testData.columns[0]].isin(X.index)&testData[testData.columns[1]].isin(X.columns))
        #print(X)
        
        
        for i in dd[0]:
            tpc=testData.values[i,1]
            #tp=np.where(X.columns==tpc)
            id_col=Xp.columns.get_loc(np.str(tpc))

            tpr=testData.values[i,0]
            #tp=np.where(X.index==tpr)
            id_row=Xp.index.get_loc(np.str(tpr))
            
            y_pred[i] = Xp.values[id_row,id_col]
          #  y_pred[i] = 0
           # if np.mod(i,10000)==0:
               # print(i)
        

        self._index = inputs.index
        outputs = pd.DataFrame(y_pred, self._index, self._keys)
        outputs.metadata = inputs.metadata

        return CallResult(outputs)


    def get_params(self) -> Params:
        return Params(X=self._X)

    def set_params(self, *, params: Params) -> None:
        self._X = params.X
