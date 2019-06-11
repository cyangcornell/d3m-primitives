import os
import ctypes
ctypes.CDLL("/opt/julia_files/julia-903644385b/lib/julia/libstdc++.so.6", os.RTLD_DEEPBIND)

import typing
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
from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.primitive_interfaces import base, unsupervised_learning

__author__ = 'Cornell'
__version__ = 'v0.1.1'

#__all__ = ('HuberPCA',)

Inputs = container.DataFrame
Outputs = container.DataFrame


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


#Parameter class.
class Params(params.Params):
    Y: typing.Union[container.DataFrame, pd.DataFrame]

#Hyperparameter class.
class Hyperparams(hyperparams.Hyperparams):
    #The number of dimensions of the latent representation.
    k = hyperparams.Hyperparameter(default=2,
                                   description='Maximum rank of the decomposed matrices. For example, if the matrix A to be decomposed is m-by-n, then after decomposition A≈XY, X is m-by-k, Y is k-by-n. ',
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])


#The HuberPCA primitive.
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


#    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, str] = None) -> None:

#        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def __init__(self, *, hyperparams: Hyperparams) -> None:

        super().__init__(hyperparams=hyperparams)
        self._k: float = hyperparams['k']
        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None
        self._index = None
        self._header = None
        self._fitted: bool = False

    def get_params(self) -> Params:
        try:
            Y_output = pd.DataFrame(self._Y, columns=self._header)
            return Params(Y=Y_output)
        except:
            raise Exception('Initial GLRM fitting not executed!')

    def set_params(self, *, params: Params) -> None:
        self._Y = params['Y'].values
        self._header = list(params['Y'])

    def get_hyperparams(self) -> Hyperparams:
        return Hyperparams(k=self._k)

    def set_hyperparams(self, *, hyperparams: Hyperparams) -> None:
        self._k = hyperparams['k']

    def set_training_data(self, *, inputs: Inputs) -> None:
        julia_components_delayed_import()
        self._losses = HuberLoss()
        self._rx = ZeroReg()
        self._ry = ZeroReg()
        self._training_inputs = inputs.values
        self._index = inputs.index
        self._header = list(inputs)
        self._fitted = False


    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        julia_components_delayed_import()
        if self._fitted:
            return
        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        glrm_j = j.GLRM(self._training_inputs, self._losses, self._rx, self._ry, self._k)
        X, Y, ch = j.fit_b(glrm_j)
        self._X = X
        self._Y = Y
        self._fitted = True
        return base.CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        julia_components_delayed_import()
        try:
            self._Y
        except NameError:
            raise Exception('Initial GLRM fitting not executed!')
        else:
            try:
                if self._Y.shape[1] != inputs.values.shape[1]:
                    raise ValueError
            except ValueError:
                raise Exception('Dimension of input vector does not match Y!')
            else:
                inputs_values = inputs.values
                inputs_index = inputs.index
                self._Y = self._Y.astype(float) #make sure column vectors finally have the datatype Array{float64,1} in Julia
                num_cols = self._Y.shape[1]
                _ry = [j.FixedLatentFeaturesConstraint(self._Y[:, i]) for i in range(num_cols)]
                glrm_j_new = j.GLRM(inputs_values, self._losses, self._rx, _ry, self._k)
                x, yp, ch = j.fit_b(glrm_j_new)
                x = x.reshape(inputs_values.shape[0], -1)
                outputs = pd.DataFrame(x, index=inputs_index)
                outputs.metadata = inputs.metadata.clear()

        return base.CallResult(outputs)

