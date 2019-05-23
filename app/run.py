import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Box
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    INPUTS:
        text (string): text for tokenization
    RETURNS:
        clean_tokens (list): lemmatized and lower case tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    cat_names = df.columns.tolist()
    for x in ['id', 'message', 'original', 'genre']:
        cat_names.remove(x)
    cat_counts = df[cat_names].sum()
    
    df['length'] = df['message'].str.len()
    name_len_val = []
    for col in cat_names:
        name_len_val.append(df[df[col]==1]['length'].values)
        
    data3 = []
    for i, name in enumerate(cat_names):
        data3.append(Box(y=name_len_val[i],
                            name = name,
                            boxpoints = 'outliers'))



    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        ####################### PLOT 1 ##########################
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        ####################### PLOT 2 ##########################
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Tagged Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Cat"
                }
            }
        },
        ####################### PLOT 3 ##########################
        {
            'data': data3,

            'layout': {
                'title': "Box Plot Styling Outliers",
                'yaxis': {
                    'type':'log',
                    'autorange':'True'
                }
    
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph1JSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph1JSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()