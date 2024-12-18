import pandas as pd
from sklearn.model_selection import train_test_split


# Function that returns df_train and df_test
def get_dataset():
    x_train = pd.read_csv('./data/Xtrain.csv')
    y_train = pd.read_csv('./data/Ytrain.csv')

    y_train = y_train.rename(columns={'Unnamed: 0': 'ID'})

    # Verify the shape of the datasets they should have same number of rows
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('Train datasets do not have same number of rows')

    df_all = x_train.join(y_train, how='inner')

    # Drop the ID column
    df_all = df_all.drop(columns='ID')

    # Renaming columns
    column_renaming = {
        'p0q0': 't0s0',
        'p1q0': 't1s0',
        'p2q0': 't2s0',
        'p3q0': 't3s0',
        'p0q1': 't0s1',
        'p0q2': 't0s2',
        'p0q3': 't0s3'
    }

    df_all.rename(columns=column_renaming, inplace=True)
    
    # Drop constant columns
    df_all = df_all.drop(columns=['way', 'composition'])

    # Split the data into train and test
    df_train, df_test = train_test_split(df_all, test_size=0.3, random_state=42)

    return df_train, df_test
