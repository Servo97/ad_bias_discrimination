
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from math import cos, pi
import pickle
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.datasets import make_regression

def check_type(dir_path):
    for i in os.listdir(dir_path):
        if i[-3:]=='csv':
            data_type = "csv"
            break
        elif i[-3:]=="zip":
            data_type = "zip"
            break
    return data_type

def generate_splits(df, target, is_train=True, split=0.25):
    if is_train:
        y = df[target]
        X = df.loc[:, df.columns != target]
        x_train, x_val, y_train, y_val = train_test_split(X,y,test_size=split)
        print(f'Training Dataset: x: {x_train.shape}, y: {y_train.shape}\
                \n Validation Dataset: x: {x_val.shape}, y: {y_val.shape}')
        return x_train, x_val, y_train, y_val
    else:
        try:
            y = df[target]
            X = df.loc[:, df.columns != target]
            print(f'Testing Dataset: x: {X.shape}, y: {y.shape}')
            return X, y
        except:
            X = df.loc[:, df.columns != target]
            return X, None

def preprocess_csv(file_path, target):
    df = pd.read_csv(file_path)
    tgt_type = None
    real_valued_cols = []
    ###### Drop NaNs
    bad_na, good_na, useless = [],[],[]
    for column in df:
        if column.lower() in ["id", 'index', 'date', 'timestamp', 'time', 'cell_id', 'userid','user_id']:
            useless.append(column)
            continue
        
        if df[column].isna().sum()/df.shape[0] > 0.3:
            bad_na.append(column)
        elif df[column].isna().sum()/df.shape[0] <= 0.3 and df[column].isna().any()==True:
            good_na.append(column)
    df = df.drop(useless, axis=1)
    df = df.drop(bad_na,axis=1)
    for col in good_na:
        df[col] = df[col].fillna(stats.stats.mode(df[col])[0][0])
    
    ###### Scale dataset
    for column in df:
        if column==target:
            if df[column].nunique()<=2: 
                tgt_type="binary classification"
            elif df[column].nunique()<=15: 
                tgt_type="softmax classification"
            else: 
                tgt_type="regression"
                scaler_t = StandardScaler()
                df[[target]] = scaler_t.fit_transform(df[[target]])
                pickle.dump(scaler_t, open('./model_info/scaler_t.pkl','wb'))
        else:
            if df[column].nunique()<=2: 
                pass
            elif df[column].nunique()<=15:
                if isinstance(df[column][0], str):
                    df[column] = df[column].astype('category')
                    df[[column]] = df[[column]].apply(lambda x:x.cat.codes)
            else: 
                real_valued_cols.append(column)
    scaler = StandardScaler()
    df[real_valued_cols] = scaler.fit_transform(df[real_valued_cols])
    pickle.dump(scaler, open('./model_info/scaler.pkl','wb'))
    pickle.dump(real_valued_cols, open('./model_info/scaled_cols.pkl','wb'))
    return df, tgt_type

def preprocess_csv_test(file_path, target, tgt_type):
    df = pd.read_csv(file_path)
    real_valued_cols = pickle.load(open('./model_info/scaled_cols.pkl','rb'))

    ###### Drop NaNs
    bad_na, good_na, useless = [],[],[]
    for column in df:
        if column.lower() in ["id", 'index', 'date', 'timestamp', 'time', 'cell_id', 'userid','user_id']:
            useless.append(column)
            continue
        
        if df[column].isna().sum()/df.shape[0] > 0.3:
            bad_na.append(column)
        elif df[column].isna().sum()/df.shape[0] <= 0.3 and df[column].isna().any()==True:
            good_na.append(column)
    df = df.drop(useless, axis=1)
    df = df.drop(bad_na,axis=1)
    for col in good_na:
        df[col] = df[col].fillna(stats.stats.mode(df[col])[0][0])

    ##### Scale dataset
    for column in df:
        if column==target:
            continue
        else:
            if df[column].nunique()<=2:
                continue
            elif df[column].nunique()<=15:
                if isinstance(df[column][0], str):
                    df[column] = df[column].astype('category')
                    df[[column]] = df[[column]].apply(lambda x:x.cat.codes)
            else:
                continue
    scaler = pickle.load(open('./model_info/scaler.pkl','rb'))
    df[real_valued_cols] = scaler.inverse_transform(df[real_valued_cols])
    try:
        scaler_t = pickle.load(open('./model_info/scaler_t.pkl','rb'))
        df[[target]] = scaler_t.inverse_transform(df[[target]])
    except:
        pass
    return df

def read_and_preprocess_test(file_path,target, tgt_type):
    df = preprocess_csv_test(file_path, target, tgt_type)
    X,y = generate_splits(df, target, is_train=False)
    return X, y

def read_and_preprocess(file_path,target,split):
    df, tgt_type = preprocess_csv(file_path, target)
    x_train, x_val, y_train, y_val = generate_splits(df, target, True, split=split)
    return x_train, x_val, y_train, y_val, tgt_type

if __name__=="__main__":
    if check_type('./heart')=="csv":
        file_path = './upvotes/train.csv'
        target = 'Upvotes'
        x_train, x_val, y_train, y_val = read_and_preprocess(file_path,target,0.2)
    else:
        print('This functionality will come in future.')
