  
import os
from typing import Union,Dict
import numpy as np
import pandas as pd

import pandas as pd
from d3m import container
from d3m.container import ndarray
from d3m import utils

from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.primitive_interfaces import base, unsupervised_learning
from d3m.primitive_interfaces.base import CallResult,DockerContainer


Inputs = container.DataFrame
Outputs = container.DataFrame

__author__ = 'Cornell'
__version__ = 'v0.1.1'

class Params(params.Params):
    A: ndarray

#Hyperparameter class.
class Hyperparams(hyperparams.Hyperparams):
    d = hyperparams.Hyperparameter(default=0,
                                   description='The reduced dimension of matrix factorization. It is usually smaller than the sizes of the matrix. When setting to 0, d will be automatically roughly estimated.',
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    tol = hyperparams.Hyperparameter(default=1e-5,
                                     description='The tolerance of the relative changes of the variables in optimization. It will be utilized to check the convergence.',
                                     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    maxiter = hyperparams.Hyperparameter(default=200,
                                         description='The maximum number of iterations.',
                                         semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    alpha = hyperparams.Hyperparameter(default=0.1,
                                       description='The regularization parameter for the factorized dense matrix A.',
                                       semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

#The LowRankImputer primitive.
class LowRankImputer(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    """
        This primitive performs low rank imputation: rather than just imputing missing entries with, for example, means or medians of each feature, it recover missing entries based on low rank structure of the dataset. 
    """

    # Primitive metadata.
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'c959da5a-aa2e-44a6-86f2-a52fe2ab9db7',
        'version': __version__,
        'name': "Low Rank Imputer",
        'keywords': ['imputation', 'preprocessing', 'low rank approximation',],
        'source': {
            'name': __author__,
                                               'contact':'mailto:cy438@cornell.edu',
            'uris': [
                'https://github.com/cyangcornell/d3m-primitives.git',
            ],
        },

        'installation': [{
            'type': 'PIP',
                         'package_uri': 'git+https://github.com/cyangcornell/d3m-primitives.git@{git_commit}#egg=pyglrm-d3m'.format(git_commit=utils.current_git_commit(os.path.dirname(__file__)))

        }],

        'python_path': 'd3m.primitives.data_preprocessing.low_rank_imputer.Cornell',

        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.LOW_RANK_MATRIX_APPROXIMATIONS,
            "IMPUTATION",
        ],
        'primitive_family': "DATA_PREPROCESSING",
        'preconditions': ["NO_CATEGORICAL_VALUES"],
        'effects': ["NO_MISSING_VALUES"],
    })

    def __init__(self, *, hyperparams: Hyperparams, docker_containers: Dict[str,DockerContainer] = None, _versbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, docker_containers = docker_containers)
        
        self.d: int = hyperparams['d']
        self.tol: float = hyperparams['tol']
        self.maxiter: int = hyperparams['maxiter']
        self.alpha: float = hyperparams['alpha']
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._fitted = False

        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted: 
            return
        
        X_incomplete=self._training_inputs.copy()
        X=self._training_inputs.values.copy()
        X=X.T 
        m0=1
        n0=0
                
        tol=self.tol
        maxiter=self.maxiter
        m,n = X.shape
        M=np.ones([m,n])
        M[np.isnan(X)]=0
        sr=M.sum()/m/n
        X[np.isnan(X)]=0

        if self.d==0:
            d=np.int(np.round(0.25*sr*min(m,n)))
        else:
            d=self.d
        
        d=min(d,min(m,n))
        alpha=self.alpha*(m+n)/d
        self._alpha=alpha
        self._d=d
        U, S, Vt = np.linalg.svd(X/sr, full_matrices=False)
        A=np.dot(U[:,0:d],np.diag(S[0:d]**0.5))
        Z=np.dot(np.diag(S[0:d]**0.5),Vt[0:d,:])
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
            G=Z+np.dot(A.T,np.multiply(M,X-np.dot(A,Z)))/tau
            Z_new=1/(alpha+tau)*G*tau

            # A_new
            if iter==1:
                A=A
            else:
                A=A_new+cc*(A_new-A_old)

            kai=rho*np.linalg.norm(np.dot(Z_new,Z_new.T),2)
            H=A+np.dot(np.multiply(M,X-np.dot(A,Z_new)),Z_new.T)/kai
            A_new=1/(alpha+kai)*H*kai

            # check convergence
            stopC=max(np.linalg.norm(Z_new-Z,'fro')/np.linalg.norm(Z_new,'fro'),np.linalg.norm(A_new-A,'fro')/np.linalg.norm(A_new,'fro'))
            isstopC=stopC<tol

            if isstopC:
                Z=Z_new;
                A=A_new;
                break
            Z_old=Z
            A_old=A
            Z=Z_new
            A=A_new

        #X_temp=np.multiply(X,M)+np.multiply(np.dot(A,Z),1-M)
        X_temp=np.dot(A,Z)

        if m0>n0:
            X_temp = X_temp.T
            
        self._A=A
        #self._X=pd.DataFrame(X_temp,X_incomplete.index,X_incomplete.columns) 
        self._X=container.DataFrame(X_temp, index=X_incomplete.index, columns=X_incomplete.columns)        
        self._fitted = True

        return CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        testData = inputs
        X=inputs.values.copy()
        tol=self.tol
        maxiter=self.maxiter
        X = X.T
        m,n = X.shape
        M=np.ones([m,n])
        M[np.isnan(X)]=0
        sr=M.sum()/m/n
        X[np.isnan(X)]=0
        alpha=self._alpha
        d=self._d
        A=self._A
        Z=np.zeros((d,n))
        rho=max(1.5*np.sqrt(M.mean()),0.5)
        tau=rho*np.linalg.norm(np.dot(A.T,A),2)
        iter=0
        cc=0.5
        while iter<self.maxiter:

            iter=iter+1
            # Z_new
            if iter==1:
                Z=Z
            else:
                Z=Z_new+cc*(Z_new-Z_old)
            
            G=Z+np.dot(A.T,np.multiply(M,X-np.dot(A,Z)))/tau
            Z_new=1/(alpha+tau)*G*tau

            # check convergence
            stopC=np.linalg.norm(Z_new-Z,'fro')/np.linalg.norm(Z_new,'fro')
            isstopC=stopC<tol

            if isstopC:
                break

            Z=Z_new                
            Z_old=Z
            Z=Z_new

            #X_temp=np.multiply(X,M)+np.multiply(np.dot(A,Z),1-M)
            X_temp=np.dot(A,Z)
            X_temp = X_temp.T

            outputs = container.DataFrame(X_temp, index=testData.index, columns=testData.columns)
            
        outputs.metadata = inputs.metadata

        return CallResult(outputs)
        
    
    
    def get_params(self) -> Params:
        return Params(A=self._A)

    def set_params(self, *, params: Params) -> None:
        self._A = params.A

