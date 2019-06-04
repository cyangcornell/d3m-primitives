import os.path
from typing import Union,Dict

import numpy as np
import pandas as pd
from d3m import container

from d3m import utils

from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.primitive_interfaces import base, transformer

from d3m.primitive_interfaces.base import CallResult,DockerContainer

Inputs = container.DataFrame
Outputs = container.DataFrame


__author__ = 'Cornell'
__version__ = 'v0.1.1'


class Hyperparams(hyperparams.Hyperparams):
    d = hyperparams.Hyperparameter(default=0,
                                   description='The reduced dimension of matrix factorization. It is usually smaller than the sizes of the matrix. When setting to 0, d will be automatically roughly estimated.',
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    tol = hyperparams.Hyperparameter(default=1e-4,
                                     description='The tolerance of the relative changes of the variables in optimization. It will be utilized to check the convergence.',
                                     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    maxiter = hyperparams.Hyperparameter(default=1000,
                                         description='The maximum number of iterations.',
                                         semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    alpha = hyperparams.Hyperparameter(default=1.0,
                                       description='The regularization parameter for the factorized dense matrix A.',
                                       semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    beta = hyperparams.Hyperparameter(default=1.0,
                                      description='The regularization parameter for the factorized sparse matrix Z.',
                                      semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

class HighRankImputer(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    This primitive imputes a dataset in which data points are drawn from multiple subspaces, which in pratice means the data have mutiple groups/classes. In such cases, the data matrices are often of high-rank. In such cases, Sparse Factorization based Matrix Completion (SFMC) can outperform classical low-rank matrix completion methods.
    The optimization is solved via accelerated proximal alternating minimization (APALM).The NaNs in the input matrix will be regarded as missing entries. The algorithm will recover the missing entries and return the recovered matrix as output.
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

        'python_path': 'd3m.primitives.cornell.high_rank_imputer',

        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.LOW_RANK_MATRIX_APPROXIMATIONS,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
    })

    def __init__(self, *, hyperparams: Hyperparams, docker_containers: Dict[str,DockerContainer] = None, _versbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, docker_containers = docker_containers)
        
        self.d: int = hyperparams['d']
        self.tol: float = hyperparams['tol']
        self.maxiter: int = hyperparams['maxiter']
        self.alpha: float = hyperparams['alpha']
        self.beta: float = hyperparams['beta']
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:

        alpha=self.alpha
        beta=self.beta
        tol=self.tol
        maxiter=self.maxiter
        
        X=inputs.values.copy()
        X = X.T
        self._keys = list(inputs)
        self._index = inputs.index
        m,n = X.shape
        
        M=np.ones([m,n])
        M[np.isnan(X)]=0

        sr=M.sum()/m/n
        
        X[np.isnan(X)]=0 

        if self.d==0:
            if sr>0.8:
                d=np.int(np.round(0.25*min(m,n)))
            else:
                d=np.int(np.round(sr*min(m,n)))
            if m*5<n:
                d=np.int(np.round(2*sr*min(m,n)))
            if min(m,n)>1000:
                d=min(d,500)
            print('The auto-estimated latent dimension is %d.' % d)
        else:
            d=self.d
            print('The user-defined latent dimension is %d.' % d)
  
        A=np.random.randn(m,d)
        Z=np.zeros((d,n))
        rho=max(1.5*np.sqrt(M.mean()),0.25)

        iter=0
        cc=0.99

        while iter<self.maxiter:
            
            iter=iter+1
            
            # Z_new
            if iter==1:
                Z=Z
            else:
                Z=Z_new+cc*(Z_new-Z_old)

            tau=rho*np.linalg.norm(A,2)**2
            
            G=Z-(-np.dot(A.T,np.multiply(M,X-np.dot(A,Z))))/tau
            Z_new=np.maximum(0,G-beta/tau)+np.minimum(0,G+beta/tau)

            
            # A_new
            if iter==1:
                A=A
            else:
                A=A_new+cc*(A_new-A_old)

            kai=rho*np.linalg.norm(Z_new,2)**2
            H=A+np.dot(np.multiply(M,X-np.dot(A,Z_new)),Z_new.T)/kai;
            A_new=1/(alpha+kai)*H*kai;

            # check convergence
            stopC=max(np.linalg.norm(Z_new-Z,'fro')/np.linalg.norm(Z_new,'fro'),np.linalg.norm(A_new-A,'fro')/np.linalg.norm(A_new,'fro'))
            isstopC=stopC<tol

            if np.mod(iter,100)==0 or isstopC or iter==1:
                obj_func=0.5*np.linalg.norm(np.multiply(M,X-np.dot(A_new,Z_new)),'fro')**2+0.5*alpha*np.linalg.norm(A_new,'fro')**2+beta*abs(Z_new).sum()
                print('iteration: %d/%d, obj_func=%f, var_change=%f.' % (iter,maxiter,obj_func,stopC))

            if isstopC:
                Z=Z_new;
                A=A_new;
                print('converged!')
                break

            Z_old=Z
            A_old=A
            Z=Z_new
            A=A_new
            
        D=abs(Z)>1e-5
        D=D+0

        print('The proportion of nonzero entries in the sparse coefficient matrix is %f.' % (D.sum()/D.shape[0]/D.shape[1]))

        Xr = np.multiply(X,M)+np.multiply(np.dot(A,Z),1-M)
        Xr = Xr.T
        
        outputs = pd.DataFrame(Xr, self._index, self._keys)
        outputs.metadata = inputs.metadata

        return CallResult(outputs)


    
