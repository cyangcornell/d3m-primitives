from pyglrm_d3m.low_rank_imputer import LowRankImputer
#import sys
#sys.path.append('../pyglrm_d3m')
#from low_rank_imputer import LowRankImputer
from d3m.container import DataFrame
import numpy as np

data = np.random.rand(4, 5)
norm1 = np.linalg.norm(data)
data[1, 1] = data[2, 2] = data[3, 3] = np.nan
data = DataFrame(data, generate_metadata=True)

model = LowRankImputer(hyperparams={'k':2})
data_clean = model.produce(inputs=data).value
norm2 = np.linalg.norm(data_clean)

print("\nInput Data with missing values\n")
print(data)

print("\nImputed Data\n")
print(data_clean)

print("\nImputation Error Norm : ",abs(norm2-norm1))
