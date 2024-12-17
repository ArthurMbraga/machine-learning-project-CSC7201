import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def preprocessData(df_train, df_test):

    def addAvgLagColumns(X):
        cols = ['t1s0', 't2s0', 't3s0']
        stations_lag_train = X.groupby('station')[cols].mean()
        stations_avg_lag_train = stations_lag_train.mean(axis=1)

        cols = ['t0s1', 't0s2', 't0s3']
        stations_lag_station = X.groupby('station')[cols].mean()
        stations_avg_lag_station = stations_lag_station.mean(axis=1)

        X['station_avg_lag_train'] = X['station'].map(stations_avg_lag_train)
        X['station_avg_lag_station'] = X['station'].map(
            stations_avg_lag_station)
        return X

    def extractHour(X):
        X['hour'] = pd.to_datetime(X['hour'], format='%H:%M:%S').dt.hour
        return X

    def removeUnusedColumns(X):
        return X.drop(columns=['date', 'train', 'station'])

    new_df_test = addAvgLagColumns(df_test.copy())
    new_df_train = addAvgLagColumns(df_train.copy())

    new_df_test = extractHour(new_df_test)
    new_df_train = extractHour(new_df_train)

    new_df_test = removeUnusedColumns(new_df_test)
    new_df_train = removeUnusedColumns(new_df_train)

    # Normalize
    numeric_features = new_df_train.select_dtypes(include=[np.number]).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    new_df_train = preprocessor.fit_transform(new_df_train)
    new_df_test = preprocessor.transform(new_df_test)

    # Keep data in DataFrame format
    new_df_train = pd.DataFrame(new_df_train, columns=numeric_features)
    new_df_test = pd.DataFrame(new_df_test, columns=numeric_features)

    return new_df_train, new_df_test
