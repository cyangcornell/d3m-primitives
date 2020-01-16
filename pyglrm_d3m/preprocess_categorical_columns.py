import os.path
from typing import Union, Dict

import numpy as np
import pandas as pd
from d3m import container
from d3m.container import ndarray, List
from d3m import utils

from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult,DockerContainer

Inputs = container.DataFrame
Outputs = container.DataFrame

__author__ = 'Cornell'
__version__ = 'v0.1.1'

class Params(params.Params):
    is_categorical: List

class Hyperparams(hyperparams.Hyperparams):
    convert = hyperparams.Hyperparameter(default=True,
                                   description='Whether to convert non-numerical features to numerical.',
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    to_type = hyperparams.Hyperparameter(default=float,
                                    description='The data type we convert features to.',
                                    semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])
#     one_hot_encode = hyperparams.Hyperparameter(default=True,
#                                      description='Whether to one-hot-encode the categorical features we identify.',
#                                      semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])

class PreprocessCategoricalColumns(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Whether we preprocess categorical columns, which are not determined by column semantic type, but determined by number of unique values.    
    """

    metadata = metadata_base.PrimitiveMetadata({
        'id': '196152a7-a873-4676-bbde-95627f4b5306',
        'version': __version__,
        'name': "Preprocessing for categorical columns",
        'keywords': ['preprocessing', 'semantic_types', ],
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
        'python_path': 'd3m.primitives.column_parser.preprocess_categorical_columns.Cornell',

        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
    })

    def __init__(self, *, hyperparams: Hyperparams, docker_containers: Dict[str,DockerContainer] = None, _versbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, docker_containers = docker_containers)
        
        self.convert: bool = hyperparams['convert']
        self.to_type: type = hyperparams['to_type']
#         self.one_hot_encode: bool = hyperparams['one_hot_encode']
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._fitted = False
        
    def fit(self, *, timeout: float=None, iterations: int=None) -> CallResult[None]:
        if self._fitted:
            return
        data = self._training_inputs.copy()
        X = self._training_inputs.values.copy()
        self.is_categorical = []
        self._u = {}
        n, m = X.shape
        
        if self.convert:
            for i in range(0,m):
                u, indices = np.unique(X[:, i], return_index=True)
                if len(u) < np.sqrt(n):
                    is_categorical_single_feature = True
            

                else:
                    try:
                        float(sorted(list(np.unique(X[:, i])))[-1])
                        is_categorical_single_feature = False
                    except:
                        is_categorical_single_feature = True
                        

 #              if is_categorical_single_feature:
 #                   k = 0
 #                   for j in range(0, len(u)):
 #                       temp = X[:, i]
 #                       if not np.isnan(u[j]):
 #                           X[temp==u[j], i] = k
 #                           k = k + 1

                self.is_categorical.append(is_categorical_single_feature)
                if is_categorical_single_feature:
                    self._u[i] = u

  #          data.values = X
 #           self._u = u
        self._fitted = True
        return CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        data = inputs.copy()
        X = inputs.values.copy()
        n, m = X.shape
 #       u=self._u
        if self.convert:
            for i in range(0, m):                
                if self.is_categorical[i]:
                    u = self._u[i]
                    temp = X[:, i]
                    k = 0                    
                    for j in range(0, len(u)):             
                        if (type(u[j]) is not str and not np.isnan(u[j])) or (type(u[j]) is str and u[j] != ''):
                            X[temp==u[j], i] = k
                            k = k + 1
#        data.values = X
#        print(X)
        if self.to_type == float:
            f = lambda x: float(x) if x != '' else np.nan
        elif self.to_type == int:
            f = lambda x: int(x) if x != '' else np.nan
            
        f_vec = np.vectorize(f)
        X = np.array(list(map(f_vec, X)))
        
        outputs = container.DataFrame(X, index=data.index, columns=data.columns)
        outputs.metadata = inputs.metadata
        return base.CallResult(outputs)
    
    def get_params(self) -> Params:
        return Params(is_categorical=self.is_categorical)

    def set_params(self, *, params: Params) -> None:
        self.is_categorical = params.is_categorical
