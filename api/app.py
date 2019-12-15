#python -m flask run
from flask import Flask, request, redirect, url_for, flash, jsonify, make_response, send_file,send_from_directory
import pickle as p
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO, open
from base64 import b64encode
from flask_cors import CORS
import pandas as pd


APP_DIR = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)
CORS(app)

@app.route('/api/welcome',methods=['GET'])
def welcome():
    response_body = {
            "msg": "OK"
        }
        
    return make_response(jsonify(response_body), 200)
    
@app.route('/api/getplot/<img_name>', methods=['GET'])
def getPlotFile(img_name):
    # Load figure from disk and display
    return send_from_directory(APP_DIR+'/model',img_name, mimetype='image/png')
     

@app.route('/api/getplotdata/confusion', methods=['GET'])
def getPlotConfBase64():
    with open(APP_DIR+'/model/img/diabConfusionMatrix.png', 'rb') as img:
        img_encoded = b64encode(img.read())

    response_body = {
            "ext": "png",
            "img": img_encoded.decode('utf-8')
        }
    return make_response(jsonify(response_body), 200)

@app.route('/api/getplotdata/histogram', methods=['GET'])
def getPlotHistBase64():
    with open(APP_DIR+'/model/img/HistogramAgeOutCome.png', 'rb') as img:
        img_encoded = b64encode(img.read())

    response_body = {
            "ext": "png",
            "img": img_encoded.decode('utf-8')
        }
    return make_response(jsonify(response_body), 200)

@app.route('/api/getplotdata/scatter', methods=['GET'])
def getPlotScatBase64():
    with open( APP_DIR+'\model\img\ScatterPlotAgeInsulin.png', 'rb') as img:
        img_encoded = b64encode(img.read())

    response_body = {
            "ext": "png",
            "img": img_encoded.decode('utf-8')
        }
    return make_response(jsonify(response_body), 200)

@app.route('/api/predict', methods=['POST'])
def predict():
    modelfile = APP_DIR+'/model/diab_liearReg.pkl'
    linear_regressor, X, Y = p.load(open(modelfile, 'rb'))
    
    req_data = request.get_json()
    print(req_data)
    
    model_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    query = pd.DataFrame([req_data],columns=model_columns)

    ### return list of predictions based on request data and return
    prediction = list(linear_regressor.predict(query))
    print(prediction)
    message = "You are likely to have diabetes" if prediction[0] == 1 else "You are not likely to have diabetes"
    return jsonify({'prediction': str(prediction),'message':message})

@app.route('/api/decisionTree', methods=['POST'])
def decisionTree():
    modelfile = APP_DIR+'/model/diab_decisionTree.pkl'
    linear_regressor, X, Y = p.load(open(modelfile, 'rb'))
    
    req_data = request.get_json()
    print(req_data)
    
    model_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    query = pd.DataFrame([req_data],columns=model_columns)

    ### return list of predictions based on request data and return
    prediction = list(linear_regressor.predict(query))
    print(prediction)
    message = "You are likely to have diabetes" if prediction[0] == 1 else "You are not likely to have diabetes"
    return jsonify({'prediction': str(prediction),'message':message})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5001)