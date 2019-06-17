import os.path
from typing import Union,Dict

import numpy as np
import pandas as pd
from d3m import container
from d3m.container import ndarray
from d3m import utils

from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.base import CallResult,DockerContainer
from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.primitive_interfaces import base, unsupervised_learning
Inputs = container.DataFrame
Outputs = container.DataFrame


__author__ = 'Cornell'
__version__ = 'v0.1.1'

class Params(params.Params):
    A: ndarray
class Hyperparams(hyperparams.Hyperparams):
    d = hyperparams.Hyperparameter(default=10,
                                   description='The reduced dimension of matrix factorization. It is usually smaller than the sizes of the matrix. When setting to 0, d will be automatically roughly estimated.',
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    epsilon = hyperparams.Hyperparameter(default=1.0,
                                   description='The parameter of huber function.',
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    t = hyperparams.Hyperparameter(default=0.001,
                                   description='Step size.',
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    tol = hyperparams.Hyperparameter(default=1e-4,
                                     description='The tolerance of the relative changes of the variables in optimization. It will be utilized to check the convergence.',
                                     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    maxiter = hyperparams.Hyperparameter(default=500,
                                         description='The maximum number of iterations.',
                                         semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    alpha = hyperparams.Hyperparameter(default=1.0,
                                       description='The regularization parameter for the factorized dense matrix A.',
                                       semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
   
class HuberPCA(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        By Huber PCA, this primitive gets low rank representation of the original dataset via Huber loss (rather than the L2 loss in standard PCA), and is thus more robust to outliers. 
    """

    # Primitive metadata.
    metadata = metadata_base.PrimitiveMetadata({
        'id': '7c357e6e-7124-4f2a-8371-8021c8c95cc9',
        'version': __version__,
        'name': "Huber PCA",
        'keywords': ['dimensionality reduction', 'low rank approximation',],
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

        'python_path': 'd3m.primitives.feature_extraction.huber_pca.Cornell',

        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.LOW_RANK_MATRIX_APPROXIMATIONS,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,

        'preconditions': [
            metadata_base.PrimitivePrecondition.NO_MISSING_VALUES,
            metadata_base.PrimitivePrecondition.NO_CATEGORICAL_VALUES,
        ]
    })


    def __init__(self, *, hyperparams: Hyperparams, docker_containers: Dict[str,DockerContainer] = None, _versbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, docker_containers = docker_containers)
        
        self.d: int = hyperparams['d']
        self.tol: float = hyperparams['tol']
        self.maxiter: int = hyperparams['maxiter']
        self.alpha: float = hyperparams['alpha']
        self.t: float = hyperparams['t']
        self.epsilon: float = hyperparams['epsilon']
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._fitted = False
        
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._fitted: 
            return

        X=self._training_inputs.values.copy()
        X=X.T 
                
        tol=self.tol
        maxiter=self.maxiter
        m,n = X.shape
        d=self.d
        alpha=self.alpha
        self._alpha=alpha
        A=np.random.randn(m,d)*0.0001
        Z=np.random.randn(d,n)*0.0001
        epsilon=self.epsilon
        iter=0
        # 0.5*||X-AZ||+0.5alpha(||X||+||Z||)
        while iter<self.maxiter:
            iter=iter+1
            gL_1=X-np.dot(A,Z)
            gL_2=epsilon*np.sign(gL_1)
            M=np.ones([m,n])
            gL_1[np.abs(gL_1)>epsilon]=0
            gL_2[np.abs(gL_1)<=epsilon]=0
            gL=gL_1+gL_2  
            gA=np.dot(gL,-Z.T)+alpha*A
            gZ=np.dot(-A.T,gL)+alpha*Z
            A_new=A-self.t*gA
            Z_new=Z-self.t*gZ
            
            stopC=max(np.linalg.norm(Z_new-Z,'fro')/np.linalg.norm(Z_new,'fro'),np.linalg.norm(A_new-A,'fro')/np.linalg.norm(A_new,'fro'))
            isstopC=stopC<tol
            if isstopC:
                Z=Z_new;
                A=A_new;
                break

            Z=Z_new
            A=A_new
            
        self._A=A 
        self._fitted = True

        return base.CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        testData = inputs
        X=inputs.values.copy()
        tol=self.tol
        alpha=self._alpha
        X = X.T
        m,n = X.shape
        epsilon=self.epsilon
        A=self._A
        Z=np.random.randn(self.d,n)*0.0001
        iter=0
        while iter<self.maxiter:

            iter=iter+1
            gL_1=X-np.dot(A,Z)
            gL_2=epsilon*np.sign(gL_1)
            M=np.ones([m,n])
            gL_1[np.abs(gL_1)>epsilon]=0
            gL_2[np.abs(gL_1)<=epsilon]=0
            gL=gL_1+gL_2  
            gZ=np.dot(-A.T,gL)+alpha*Z
            Z_new=Z-self.t*gZ
            
            stopC=np.linalg.norm(Z_new-Z,'fro')/np.linalg.norm(Z_new,'fro')
            isstopC=stopC<tol
            if isstopC:
                Z=Z_new;
                break

            Z=Z_new

        Z=Z.T

        outputs = pd.DataFrame(Z, index=testData.index)
            
        outputs.metadata = inputs.metadata.clear()

        return base.CallResult(outputs)
        
    
    
    def get_params(self) -> Params:
        return Params(A=self._A)

    def set_params(self, *, params: Params) -> None:
        self._A = params.A



