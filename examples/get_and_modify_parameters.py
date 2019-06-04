from d3m.container import DataFrame
from pyglrm_d3m.huber_pca import HuberPCA

A = DataFrame([[1, 2, 3, 4], [2, 4, 6, 8], [4, 5, 6, 7]])

model = HuberPCA(hyperparams={'k':2}) #create a class for Huber PCA
model.set_training_data(inputs=A)
model.fit()

#get parameter
parameter = model.get_params()
print("Initial parameter (Y): {}".format(parameter['Y'].values))

#modify parameter
print("Now we change the (0,0) entry of Y to 0, and set the modified Y as parameter of the Huber PCA class.")
parameter['Y'].values[0, 0] = 0
model.set_params(params={'Y': parameter['Y']})

#check if parameter has been modified
parameter = model.get_params()
print("Modified parameter (Y): {}".format(parameter['Y'].values))


