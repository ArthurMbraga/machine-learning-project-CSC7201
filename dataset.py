import pandas as pd


# Function that returns df_train and df_test
def get_dataset():
    x_train = pd.read_csv('./data/Xtrain.csv')
    y_train = pd.read_csv('./data/Ytrain.csv')
    x_test = pd.read_csv('./data/Xtest.csv')
    y_sample = pd.read_csv('./data/Ysample.csv')

    y_train = y_train.rename(columns={'Unnamed: 0': 'ID'})
    y_sample = y_sample.rename(columns={'Unnamed: 0': 'ID'})

    # Verify the shape of the datasets they should have same number of rows
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('Train datasets do not have same number of rows')

    if x_test.shape[0] != y_sample.shape[0]:
        raise ValueError('Test datasets do not have same number of rows')

    df_train = x_train.join(y_train, how='inner')
    df_test = x_test.join(y_sample, how='inner')

    # Drop the ID column
    df_train = df_train.drop(columns='ID')
    df_test = df_test.drop(columns='ID')

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

    df_train.rename(columns=column_renaming, inplace=True)
    df_test.rename(columns=column_renaming, inplace=True)
    
    # Drop constant columns
    df_train = df_train.drop(columns=['way', 'composition'])
    df_test = df_test.drop(columns=['way', 'composition'])


    return df_train, df_test
