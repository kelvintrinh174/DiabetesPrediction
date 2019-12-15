#%%
#exemples
#https://www.kdnuggets.com/2019/01/build-api-machine-learning-model-using-flask.html
#https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

import numpy as np
import pandas as pd
import os as os
import pickle
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

#---------------Importing the data file-------------------
dir = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir + '\diabetes.csv')  # load data set
print(data.describe())
#---------------------------------------------------------

#---------------Build the model --------------------------
X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

#---------------------------------------------------------

#---------------Ploting graph-----------------------------
img = plt.gcf()
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
plt.draw()
#---------------------------------------------------------

#---------------Pickle model and plot---------------------
model_filename = dir + '\diab_liearReg.pkl'
img_filename = dir + '\scatterPlot.png'

tuple_objects =  (linear_regressor, X, Y)
pickle.dump(tuple_objects, open(model_filename, 'wb'))
#---------------------------------------------------------


# %%
