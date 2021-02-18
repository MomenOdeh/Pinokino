# Introduction
Do you trust all the news you hear from social media? All news are not real, right? So how will you detect the fake news? We will be using Multi-layer Deep Learnign Model to classify the news as fake or real.

# Project Structre
This project has four major parts :

* ModelAR.ipynb - This contains code fot our Machine Learning model to classify Arabic News.
* ModelAR.ipynb - This contains code fot our Machine Learning model to classify English News.
* datasets - Link to download the dataset from googledrive!!
* app.py - This contains Flask APIs that receives news url through GUI or API calls, extracts the article from the url, feeds it to the model and returns the prediction.
* Ar_Model.h5 & Eng_Model.h5 - Pre-trained Models to work with the Web App.
* templates - This folder contains the HTML template to allow user to enter url and displays whether the news is fake or real.
* static - This folder contains the CSS file.
* requirements.txt - It contains the list of libraries required to run the heroku app

# Running the project on local machine

Ensure that you are in the project home directory. Create the machine learning model by running below command -

Run app.py using below command to start Flask API
python app.py
By default, flask will run on port 5000.

Navigate to URL http://127.0.0.1:5000 
