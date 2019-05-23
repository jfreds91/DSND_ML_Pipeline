# Disaster Response Pipeline Project
Jesse Fredrickson

5/22/19

## Motivation:
The purpose of this project is to create a machine learning pipeline which implements ETL steps to clean and store data in a SQL server, and then trains a model based on that data with the intention of classifying future text based on a number of parameters in the training set. Specifically, this dataset contains the text from tweets in a disaster scenario and attempts to learn to classify if they pertain to helpful information concerning categories such as water, lone children, or military presence.

## Files:
**.ipynb:** .ipynb files were included for debugging purposes and experimental code segments
**app/run.py:** This file contains all of the backend python for the webapp, including data processing and flask routing
**app/templates/.html:** These are the html files for the main webapp page (master) and the results of a text classification (go)
**data/.csv:** These are the training datasets passed into the ETL pipeline
**data/.db:** Database file(s) for the cleaned dataset
**models/.pkl:** Pickle file(s) for the trained and optimized model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
