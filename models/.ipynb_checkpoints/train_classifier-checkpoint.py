import sys

# import libraries
from sqlalchemy import create_engine

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import pickle


def load_data(database_filepath):
    '''
    INPUTS
    database_filepath: SQL database filepath
    
    RETURNS
    X: Series 
    Y: Series 
    category_names: list of category names
    '''
    
    # load data from database
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('InsertTableName', con = engine)

    predict_cols = ['request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']
    X = df['message']
    y = df[predict_cols]

    return X, y, predict_cols

def tokenize(text):
    tokens = word_tokenize(text)
    lemm = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemm.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    # Build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        #('clf', MultiOutputClassifier(RandomForestClassifier()))
        #('clf', MultiOutputClassifier(SVC()))
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_preds = pipeline.predict(X_test)
    print(classification_report(Y_test, y_preds, labels=category_names))


def save_model(model, model_filepath):
    pickle.dump(cv.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()