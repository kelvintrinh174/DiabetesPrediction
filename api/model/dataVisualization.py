# -*- coding: utf-8 -*-
"""
Spyder Editor

Credits to Vincent Lugat of Kaggle for 99% of the code and motivation.
"""

# %%
import pandas as pd
import numpy as np
import os as os
import plotly.offline as plotter
import plotly.graph_objs as plotGraph
import itertools
import matplotlib.pyplot as plt

####---------Logistic Regression Model-------------
####Data Exploration
# import dataset
data_diabetes=pd.read_csv('C:/Dev/python/ML/solution/api/model/diabetes.csv')
dir = os.path.dirname(os.path.realpath(__file__))
#----describe data elements (columns)------------ 
#------------COUNT BY BAR-----------------------
def getOutcomeCountByBar(dataset):
    barData = plotGraph.Bar( 
                    x = dataset['Outcome'].value_counts().values.tolist(), 
                    y = ['healthy','diabetic'], 
                    orientation = 'h', 
                    text=dataset['Outcome'].value_counts().values.tolist(), 
                    textfont=dict(size=16),
                    textposition = 'auto',
                    opacity = 0.7,
                    marker=dict(
                    color=['green', 'red'],
                    line=dict(color='#000000',width=1.5)))

    layout = dict(title =  'Count of Outcome variable')

    fig = plotGraph.Figure(data=barData, layout=layout)
    #plotter.iplot(fig)
    plotter.plot(fig)
    

#------------COUNT BY PIE-------------------
def getOutcomeCountByPie(dataset):
    pieData = plotGraph.Pie(
                   labels = ['healthy','diabetic'], 
                   values = dataset['Outcome'].value_counts(), 
                   textfont=dict(size=16),
                   opacity = 0.7,
                   marker=dict(
                           colors=['green', 'red'], 
                           line=dict(color='#000000', width=1.5)))

    layout = dict(title =  'Distribution of Outcome variable')

    fig = plotGraph.Figure(data = pieData, layout=layout)
    #plotter.iplot(fig)
    plotter.plot(fig)
    
#------------MISSING VALUES-----------------------
# Define missing plot to detect all missing values in dataset
def getDataMissingVal(dataset, key):
    nan_data_diabetes = dataset.copy(deep = True)
    nan_data_diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = nan_data_diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
    null_feat = pd.DataFrame(len(nan_data_diabetes[key]) - nan_data_diabetes.isna().sum(), columns = ['Count'])
    percentage_null = pd.DataFrame((len(nan_data_diabetes[key]) - (len(nan_data_diabetes[key]) - nan_data_diabetes.isnull().sum()))/len(nan_data_diabetes[key])*100, columns = ['Count'])
    percentage_null = percentage_null.round(2)

    barData = plotGraph.Bar(
                x = null_feat.index, 
                y = null_feat['Count'],
                opacity = 0.7,
                text = percentage_null['Count'], 
                textposition = 'auto',
                marker=dict(
                        color = '#ff6600',
                        line=dict(color='#000000',width=1.5)))

    layout = dict(title =  "Missing Values by %")

    fig = plotGraph.Figure(data = barData, layout=layout)
    #py.iplot(fig)
    plotter.plot(fig)
    print(nan_data_diabetes.isnull().sum())

#---------------CORRELATION PLOT-----------------------
def correlation_plot(dataset):
    #correlation
    correlation = dataset.corr()
    #tick labels
    matrix_cols = correlation.columns.tolist()
    #convert to array
    corr_array  = np.array(correlation)
    heatMap = plotGraph.Heatmap(
                       z = corr_array,
                       x = matrix_cols,
                       y = matrix_cols,
                       colorscale='Picnic')
    layout = dict(title = 'Correlation Matrix for variables')

    fig = plotGraph.Figure(data = heatMap, layout = layout)
    #py.iplot(fig)
    plotter.plot(fig)
    
#--------------DATA PLOTS-----------------------------
def showDataplot(dataset):
    columns=dataset.columns[:8]
    plt.subplots(figsize=(18,15))
    length=len(columns)
    for i,j in itertools.zip_longest(columns,range(length)):
        plt.subplot((length/2),3,j+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.5)
        dataset[i].hist(bins=20,edgecolor='black')
        plt.title(i)
    plt.savefig(dir+'barplot-dataplot.png')
    plt.show()
        
def featureCorrelationPlot(dataset, feature1, feature2):
    D = dataset[(dataset['Outcome'] != 0)]
    H = dataset[(dataset['Outcome'] == 0)]
    scatterPlotDiabetic = plotGraph.Scatter(
        x = D[feature1],
        y = D[feature2],
        name = 'diabetic',
        mode = 'markers', 
        marker = dict(color = 'red', line = dict(width = 1)))

    scatterPlotHealthy = plotGraph.Scatter(
        x = H[feature1],
        y = H[feature2],
        name = 'healthy',
        mode = 'markers',
        marker = dict(color = 'green', line = dict(width = 1)))

    layout = dict(title = feature1 +" "+"vs"+" "+ feature2,
                  yaxis = dict(title = feature2,zeroline = False),
                  xaxis = dict(title = feature1, zeroline = False),
                  plot_bgcolor = "darkgrey")

    plots = [scatterPlotDiabetic, scatterPlotHealthy]

    fig = dict(data = plots, layout=layout)
    #py.iplot(fig)
    plotter.plot(fig)

def cleanData(dataset):
    nan_data= dataset.copy(deep = True)
    nan_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = nan_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
    nan_data['Glucose'].fillna(nan_data['Glucose'].mean(), inplace = True)
    nan_data['BloodPressure'].fillna(nan_data['BloodPressure'].mean(), inplace = True)
    nan_data['SkinThickness'].fillna(nan_data['SkinThickness'].median(), inplace = True)
    nan_data['Insulin'].fillna(nan_data['Insulin'].median(), inplace = True)
    nan_data['BMI'].fillna(nan_data['BMI'].median(), inplace = True)
    return nan_data


#-----------UTILIZING FUNCTIONS------------------------
#getOutcomeCountByBar(data_diabetes)
#getOutcomeCountByPie(data_diabetes)
#getDataMissingVal(data_diabetes, 'Outcome')
#correlation_plot(data_diabetes)
#showDataplot(data_diabetes)    
#featureCorrelationPlot(data_diabetes, "Age", "Glucose")
#featureCorrelationPlot(cleanData(data_diabetes), "Insulin", "Glucose")
#featureCorrelationPlot(cleanData(data_diabetes), "BMI", "Age")
featureCorrelationPlot(cleanData(data_diabetes), "BloodPressure", "SkinThickness")