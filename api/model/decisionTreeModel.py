# %%
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:25 2019

@author: kelvi
"""
####---------Implementing a decision tree with scikit-learn-------------
##D:\Project\Python\solution\api\model
import pandas as pd
import os as os
import pickle 
data=pd.read_csv('diabetes.csv')
data.head()

data['Outcome'].unique()

colnames=data.columns.values.tolist()
print(colnames)
##column: species
predictors=colnames[:8]
print(predictors)

target=colnames[8]
print(target)

import numpy as np
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75 
print(data.head(15))
train, test = data[data['is_train']==True], data[data['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# %%
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt.fit(train[predictors], train[target])

preds=dt.predict(test[predictors])
pd.crosstab(test['Outcome'],preds,rownames=['Actual'],colnames=['Predictions'])
# %%
###Visualizing the tree

from sklearn.tree import export_graphviz
with open('dtree2.dot', 'w') as dotfile:
    export_graphviz(dt, out_file = dotfile, feature_names = predictors)
dotfile.close()


###generate the tree image using pydot conda
import pydot
(graph,) = pydot.graph_from_dot_file('dtree2.dot')
graph.write_png('./img/dtree.png')

# %%
X=data[predictors]
Y=data[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)

# %%
dt1 = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20, random_state=99)
dt1.fit(trainX,trainY)
#---------------Pickle model ---------------------
dir = os.path.dirname(os.path.realpath(__file__))
model_filename = dir + '\diab_decisionTree.pkl'
tuple_objects =  (dt1, X, Y)
pickle.dump(tuple_objects, open(model_filename, 'wb'))



# 10 fold cross validation using sklearn and all the data i.e validate the data 
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))

import seaborn as sns
import matplotlib.pyplot as plt     
cm = confusion_matrix(testY, testY_predict, labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); 
ax.yaxis.set_ticklabels(['0', '1']);

plt.savefig(dir+'/img/decisionTreeConfusionMatrix.png')

plt.show()
