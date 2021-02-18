#Importing the Libraries
import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import os
import sklearn.externals
import joblib
import flask
import os
import newspaper
from newspaper import Article
import urllib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langdetect import detect

#Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

# with open('model.pickle', 'rb') as handle:
# 	model = pickle.load(handle)

Eng_Model = tf.keras.models.load_model('Eng_Model.h5')
Ar_Model = tf.keras.models.load_model('Ar_Model.h5')

max_len = 400
max_words = 2000
token = Tokenizer(num_words=max_words, lower=True, split=' ')

@app.route('/')
def main():
    return render_template('main.html')

#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict',methods=['GET','POST'])
def predict():
    text = request.get_data(as_text=True)[5:]
    Lang = detect(text)
    # print("PREDICTED  LANGUAGE IS =========>>>  " , Lang)
    # print("TYPEEE --------- >>>>>> ",type(text))

    seq = token.texts_to_sequences([text]) 
    padded = pad_sequences(seq, maxlen=max_len)

    print("SHAPE ----->>> ",)
    if Lang == 'en':
    	pred = Eng_Model.predict(padded)
    else:
    	pred = Ar_Model.predict(padded)

    if pred[0]>0.6:
    	message = "FAKE"
    elif pred[0]<=0.6 and pred[0]>=0.4:
    	message = "Undetermined"
    elif pred[0]<0.4:
    	message = "REAL"
    	
    return render_template('main.html', prediction_text='The news is {}'.format(message))
    
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)