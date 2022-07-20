import re
import uuid
import nltk
import pickle
import requests
import subprocess
import numpy as np
import pandas as pd
from flask import Flask,url_for,request,jsonify
from nltk.tokenize import sent_tokenize, word_tokenize
#from preprocessing_pipeline import text_cleaner
#from config import icliniq_stopwords
from flashtext import KeywordProcessor
#import spam_identifier
#from predict_symptom import predict_disease
from nltk.stem import PorterStemmer,SnowballStemmer,WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, redirect, url_for,send_from_directory
from werkzeug.utils import secure_filename
#from responsetime import MSKCC_Analyzer,MSKCC_Analyzer_v2
#from dataloader import summary_time_calculator
#from dataloader import column_checker
#from dataloader import dataloader_from_S3
#from datapipeline import my_dic_ad
from flashtext import KeywordProcessor
gbm = pickle.load(open("E:/Pred_sym/gbm.pkl", "rb"))

#---------- Create Flask Instance-------------#
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def hello():
    return 'iCliniq Advertisement API and iCliniq smartAPI !!'

'''@app.route('/predict',methods=['GET','POST'])
def predict():
    main_data = request.get_json()
    message = main_data['message']
    data = str(message)
    pred = model.predict([data])
    print(pred)
    return pred
    
    #output = ""
    #print(output)
    #return float(output) '''
@app.route('/word',methods=['GET','POST'])
def ketword():
    symptoms = [i if regex.search(i) == None else i.replace('_', ' ') for i in symptoms ]
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(symptoms)
    text = 'I have ITCHING, joint pain and fatigue'
    keyword_processor.extract_keywords(text)
'''def predict_disease(query):
    #query = []
    matched_keyword = keyword_processor.extract_keywords(query)
   # model = GradientBoostingClassifier()
    gbm = pickle.load(open("/home/icliniq-n10/Videos/Predicting-Diseases-From-Symptoms/gbm.pkl", "rb"))
    if len(matched_keyword) == 0:
        print("No Matches")
    else:
        regex = re.compile(' ')
        processed_keywords = [i if regex.search(i) == None else i.replace(' ', '_') for i in matched_keyword]
        print(processed_keywords)
        coded_features = []
        for keyword in processed_keywords:
            coded_features.append(feature_dict[keyword])
        #print(coded_features)
        sample_x = []
        for i in range(len(features)):
            try:
                sample_x.append(i/coded_features[coded_features.index(i)])
            except:
                sample_x.append(i*0)
        sample_x = np.array(sample_x).reshape(1,len(sample_x))
        #sample_x = model.predict([query])
        print('Predicted Disease: ',gbm.predict(sample_x)[0])
        json_obj = json.dumps(gbm.predict(sample_x)[0], indent=4)
        return json_obj'''
@app.route('/predict',methods=['GET','POST'])
def predict():
    main_data = request.get_json()
    message = main_data['query']
    #predict_disease(query)
    matched_keyword = keyword_processor.extract_keywords(message)
    #data = str(matched_keyword)
    if len(matched_keyword) == 0:
        print("No Matches")
    else:
        regex = re.compile(' ')
        #print(data)
        processed_keywords = [i if regex.search(i) == None else i.replace(' ', '_') for i in matched_keyword]
        #print(processed_keywords)
        coded_features = []
        for keyword in processed_keywords:
            coded_features.append(feature_dict[keyword])
            print(coded_features)
        sample_x = []
        for i in range(len(features)):
            try:
                sample_x.append(i/coded_features[coded_features.index(i)])
            except:
                sample_x.append(i*0)
        sample_x = np.array(sample_x).reshape(1,len(sample_x))
        print('Predicted Disease: ',gbm.predict(sample_x)[0])
        predict_disease = gbm.predict([sample_x][0])
        #output = gbm.predict(sample_x[0])

            #pred0 = icliniqML.predict_proba([data])
        #for label in pred:
        #predict_disease = np.array(predict_disease).reshape(1,len(predict_disease))
        #output = ""
        #print(output)
        #print(sample_x)
        return predict_disease
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False, port = 4675)
