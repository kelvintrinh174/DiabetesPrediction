# %%
import pandas as pd
import numpy as np
import matplotlib
import os as os
import pickle 
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix

####---------Logistic Regression Model-------------
####Data Exploration
# import dataset
data_diabetes=pd.read_csv('diabetes.csv')
dir = os.path.dirname(os.path.realpath(__file__))
#----describe data elements (columns)------------ 
print('Sample of 5 records:\n')
print(data_diabetes.head()) #print first 5 records
print('Shape of dataset: (records,column):')
print(data_diabetes.shape)
print('Data type of columns\n')
print(data_diabetes.dtypes)
# provide descriptions & types with values of each element
print(data_diabetes.describe())

#-----------------Data Cleaning and Wrangling---------------------
# copy dataset and replace 0 values in appropriate columns w/ NaN
nan_data_diabetes = data_diabetes.copy(deep = True)
nan_data_diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = nan_data_diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
## describe datasets and show how many values are missing 
#print(data_diabetes.describe().T)
#print(nan_data_diabetes.describe().T)
#print(nan_data_diabetes.isnull().sum())
# impute missing values w/ mean/median of the column, try later with mean/median for values with same outcome 
nan_data_diabetes['Glucose'].fillna(nan_data_diabetes['Glucose'].mean(), inplace = True)
nan_data_diabetes['BloodPressure'].fillna(nan_data_diabetes['BloodPressure'].mean(), inplace = True)
nan_data_diabetes['SkinThickness'].fillna(nan_data_diabetes['SkinThickness'].median(), inplace = True)
nan_data_diabetes['Insulin'].fillna(nan_data_diabetes['Insulin'].median(), inplace = True)
nan_data_diabetes['BMI'].fillna(nan_data_diabetes['BMI'].median(), inplace = True)

#---------Graphs and visualizations-------------------------------------
#%%
#create a scatterplot
print('Scatter Plot with Age and Insulin')
fig_diabetesAgeInsulin = nan_data_diabetes.plot(kind='scatter',x='Insulin',y='Age')
print(dir)
# Save the scatter plot with Age and Insulin
fig_diabetesAgeInsulin.figure.savefig(dir+'/img/ScatterPlotAgeInsulin.png')

# Plot a histogram about the frequency of Age

#not use this histogram
#nan_data_diabetes.Age.hist()
# plt.title('Histogram of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency') 
# plt.figure(dpi=400).savefig('./img/HistogramAgeInsulin.png')

print(nan_data_diabetes.Age.dtypes)

pd.crosstab(nan_data_diabetes.Age,nan_data_diabetes.Outcome)
pd.crosstab(nan_data_diabetes.Age,nan_data_diabetes.Outcome).plot(kind='bar')
plt.title('Frequency of Age')
plt.xlabel('Age')
plt.ylabel('Frequency of Outcome')
plt.xscale('linear')
plt.savefig(dir+'/img/HistogramAgeOutCome.png')


# Outcome is the class, all other columns are variables
X=nan_data_diabetes[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y=nan_data_diabetes['Outcome']


# 70/30 training/test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3,random_state=0)

# Create and fit the model with test data
diabetesClassifier = linear_model.LogisticRegression(solver='lbfgs', max_iter=250)
diabetesClassifier.fit(X_train,Y_train)

#---------------Pickle model ---------------------

model_filename = dir + '\diab_liearReg.pkl'
tuple_objects =  (diabetesClassifier, X, Y)
pickle.dump(tuple_objects, open(model_filename, 'wb'))
#---------------------------------------------------------

# make predictions for class of remaining test data 
probs = diabetesClassifier.predict_proba(X_test)
print("\nProbabilities:")
print(probs)
predicted = diabetesClassifier.predict(X_test)
print("\nPredicted Outcomes:")
print(predicted)

# compare predictions to actual outcome to get model accuracy
print("\nAccuracy:")
print(metrics.accuracy_score(Y_test,predicted))

# run cross validation 10 times 

scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs', max_iter=250),X,Y, scoring='accuracy',cv=10)
print("\nCross validation scores from 10 runs:")
print(scores)
print("\nAverage of the above 10 runs:")
print(scores.mean())

# generate a confusion matrix for the model
prob=probs[:,1]
prob_df=pd.DataFrame(prob)
prob_df['predict']=np.where(prob_df[0]>=0.5,1,0)
Y_A=Y_test.values
Y_P=np.array(prob_df['predict'])


confusion_matr=confusion_matrix(Y_A,Y_P)
print("\nConfusion matrix:")
print(confusion_matr)

fig, ax = plot_confusion_matrix(conf_mat= confusion_matr,colorbar=True)
plt.savefig(dir+'/img/diabConfusionMatrix.png')
plt.show()

# %%


