from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder


class Object(object):
    pass

def load_pima(data):
    df = pd.read_csv('data/diabetes.csv')
    data.data = np.asarray(df.loc[:, df.columns!='Outcome'])
    data.target = np.asarray(df['Outcome'])
    return data


def load_heart_cleveland(data):
    df = pd.read_csv('data/heart_cleveland.csv')
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(df.iloc[:, -1])
    return data

def load_dermatalogy(data):
    df = pd.read_csv('data/dermatalogy.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(y_train)
    return data

def  load_thyroid(data):
    df = pd.read_csv('data/thyroid.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(y_train)
    return data


def  load_hepatitis(data):
    df = pd.read_csv('data/hepatitis.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(y_train)
    return data

def load_data(dataset_name, test_size=0.1, random_state=1):
    sc = StandardScaler()
    data = Object()
    if dataset_name == 'pima':
        data = load_pima(data)
    elif dataset_name == 'heart_cleveland':
        data = load_heart_cleveland(data)
    elif dataset_name == 'dermatalogy':
        data = load_dermatalogy(data)
    elif dataset_name == 'thyroid':
        data = load_thyroid(data)
    elif dataset_name == 'hepatitis':
        data = load_hepatitis(data)  
    X, y = data.data, data.target
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=test_size, 
                                                            random_state=random_state
                                                            )
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print('Train: ', X_train.shape, ' | Test: ', X_test.shape)
        print('Train labels: ', np.unique(y_train, return_counts=True))
        print('Test labels: ', np.unique(y_test, return_counts=True))
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        return X_train, X_test, y_train, y_test
    else:
        X_train = sc.fit_transform(X)
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y)
        return X_train, None, y_train, None