import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
        '''
    INPUTS:
        messages_filepath (string): filepath of disaster_messages.csv
        categories_filepath (string): filepath of disaster_categories.csv
    RETURNS:
        df (DataFrame): merged dataframe, with columns transformed
        categories (list): list of category names
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge dfs
    df = messages.merge(categories, on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)
    
    # rename cols
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # convert cols to binary
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # drop the original categories column from df, add in new catcols
    df.drop(columns = ['categories'], inplace = True)
    df = df.join(categories, sort = False)
    
    return df, categories.columns.tolist()

def clean_data(df, categories):
    '''
    INPUTS:
        df (DataFrame): dataframe for cleaning
        categories (list): list of categories for classification
    RETURNS:
        df (DataFrame): cleaned dataframe
    '''
    # drop dupes
    df.drop_duplicates(subset='id', keep='first', inplace=True)
    
    # Remove NA
    df.dropna(subset = [cat for cat in categories], inplace = True)    
    
    return df
    
def save_data(df, database_filename):
    # save df
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories_list = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories_list)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
