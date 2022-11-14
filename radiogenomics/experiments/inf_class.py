"""
Author: Ivo Gollini Navarrete
Date: 14/nov/2022
Institution: MBZUAI
"""

#Imports
import os
import numpy as np
import pickle

MODELS =[
    "qda",
    "dt",
    "rf",
    "svc"
]

"""Class to perform inference on a trained model"""
class Inf_class:
    def __init__(self, data_path, classifiers_path, model_class):
        self.data_path = data_path
        self.classifiers_path = classifiers_path
        self.model_class= model_class

        # Initialize model
        model_path = os.path.join(self.classifiers_path, self.model_class + '_model.sav')
        try:
            self.loaded_model = pickle.load(open(model_path, 'rb'))
            print("Model {} loaded".format(model_class))
            # print(self.loaded_model.get_params())
        except:
            print("Model {} not found".format(model_class))
            print(MODELS)
            exit()

        # Initialize data
        X = []
        
        for case in os.listdir(data_path):
            feat_path = os.path.join(data_path, case, case + "_lda_feat.npy")
            X.append(np.load(feat_path))

        self.X = np.array(X)
        print("Data loaded with {} cases".format(len(self.X)))

    def run(self):
        # Predict
        result = self.loaded_model.predict(self.X)
        print(result)

