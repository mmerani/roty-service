# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:43:44 2020

@author: Michael
"""

from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
import json


flask_app = Flask(__name__)
app = Api(app = flask_app, 
          version = "1.0", 
          title = "2020 NBA Rookie of the Predictions", 
          description = "Predict the rookie of the year")

name_space = app.namespace('predictions', description='Prediction API')

model = app.model('Prediction params', 
                 {'model': fields.Integer(required=True)})

def load_json():
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)
    return predictions
    
predictions = load_json()

@name_space.route("/")
class MainClass(Resource):

    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

    @app.expect(model)     
    def post(self):
        try: 
            #with open('predictions.json', 'r') as f:
               #predictions = json.load(f)
            formData = request.json 
            data = int(formData['model'])
            models = {0: 'Linear Regression', 1:'Gradient Descent', 2:'Ridge Regression', 3:'Lasso Regression',4:'Elastic Net'}
            response = jsonify({
                "statusCode": 200,
                "status": "Predictions added",
                "result": predictions[models[data]]
                })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as error:
            return jsonify({
                "statusCode": 500,
                "status": "Could not make prediction",
                "error": str(error)
            })
    
    
flask_app.run()