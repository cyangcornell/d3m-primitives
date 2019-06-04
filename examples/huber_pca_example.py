from pyglrm_d3m.huber_pca import HuberPCA
from d3m.container import DataFrame

A = DataFrame([[1, 2, 3, 4], [2, 4, 6, 8], [4, 5, 6, 7]], generate_metadata=True)

model = HuberPCA(hyperparams={'k':2}) #create a class for Huber PCA
model.set_training_data(inputs=A)
model.fit()
a_new = DataFrame([[6, 7, 8, 9]]) #initialize a new row to be tested
x = model.produce(inputs=a_new).value.values #get the latent representation of a_new

print(x)
