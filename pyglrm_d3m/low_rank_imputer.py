import os
import ctypes
ctypes.CDLL("/opt/julia_files/julia-903644385b/lib/julia/libstdc++.so.6", os.RTLD_DEEPBIND)

from typing import Union,Dict
import numpy as np
import pandas as pd

## Python-wrapped Julia and GLRM components
#global QuadLoss, L1Loss, HuberLoss
#global ZeroReg, NonNegOneReg, QuadReg, NonNegConstraint
#global j
#from . import loss_reg
#
#QuadLoss = loss_reg.QuadLoss
#L1Loss = loss_reg.L1Loss
#HuberLoss = loss_reg.HuberLoss
#ZeroReg = loss_reg.ZeroReg
#NonNegOneReg = loss_reg.NonNegOneReg
#QuadReg = loss_reg.QuadReg
#NonNegConstraint = loss_reg.NonNegConstraint
#j = loss_reg.j

# d3m modules
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.primitive_interfaces import base, transformer
from common_primitives.utils import remove_columns_metadata
from d3m.primitive_interfaces.base import CallResult,DockerContainer

Inputs = container.DataFrame
Outputs = container.DataFrame

__author__ = 'Cornell'
__version__ = 'v0.1.1'

#__all__ = ('LowRankImputer',)

# This function allows us to delay the import process for Julia components to
# avoid up front overhead.
julia_components_loaded = False
def julia_components_delayed_import():
  # Only perform  the import once.
  global julia_components_loaded
  if julia_components_loaded:
    return
  else:
    julia_components_loaded = True

  # Now import and define the symbols we want:
  global QuadLoss, L1Loss, HuberLoss
  global ZeroReg, NonNegOneReg, QuadReg, NonNegConstraint
  global j
  from . import loss_reg

  QuadLoss = loss_reg.QuadLoss
  L1Loss = loss_reg.L1Loss
  HuberLoss = loss_reg.HuberLoss
  ZeroReg = loss_reg.ZeroReg
  NonNegOneReg = loss_reg.NonNegOneReg
  QuadReg = loss_reg.QuadReg
  NonNegConstraint = loss_reg.NonNegConstraint
  j = loss_reg.j

#Hyperparameter class.
class Hyperparams(hyperparams.Hyperparams):
    k = hyperparams.Hyperparameter(default=2,
                                   description='Maximum rank of the decomposed matrices. For example, if the matrix A to be decomposed is m-by-n, then after decomposition A≈XY, X is m-by-k, Y is k-by-n. ',
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])


#The LowRankImputer primitive.
class LowRankImputer(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

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
                'https://gitlab.datadrivendiscovery.org/cyang2/pyglrm_d3m.git',
            ],
        },

        'installation': [{
            'type': 'PIP',
                         'package_uri': 'git+https://gitlab.datadrivendiscovery.org/cyang2/pyglrm_d3m.git@{git_commit}#egg=pyglrm-d3m'.format(git_commit=utils.current_git_commit(os.path.dirname(__file__)))

        }],

        'python_path': 'd3m.primitives.cornell.pyglrm_d3m.low_rank_imputer',

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

        self._k: float = hyperparams['k']

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        julia_components_delayed_import()

        columns_to_drop = list(np.where(np.array(np.sum(np.invert(np.isnan(inputs))))==0)[0])

        inputs.drop(columns=inputs.columns[columns_to_drop], inplace=True)
        obs = list(zip(*np.where(np.invert(np.isnan(inputs.values)))))
        obs = [(item[0]+1, item[1]+1) for item in obs]
        self._keys = list(inputs)
        self._index = inputs.index

        glrm_j = j.GLRM(j.DataFrame(inputs.values), QuadLoss(), ZeroReg(), ZeroReg(), self._k, obs=obs)
        X, Y, ch = j.fit_b(glrm_j)

        self._X = X
        self._Y = Y
        A = np.dot(self._X.T, self._Y)
        outputs = container.DataFrame(A, index=self._index, columns=self._keys)
        outputs.metadata = remove_columns_metadata(inputs.metadata, columns_to_drop)
        
        return CallResult(outputs)
