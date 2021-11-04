#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:24:16 2021

@author: chuckzumbaugh
"""
import numpy as np
import sklearn

def get_dataframe_info(df):
    equals = np.repeat("=", 80)
    print("".join(equals))
    print("DATA DESCRIPTION:")
    print("Data description: {} rows and {} columns".format(df.shape[0], df.shape[1]))
    print("Data types: {}".format(df.dtypes.unique()))
    print("Number of missing values: {}".format(df.isnull().sum().sum()))
    print("Number of duplicated rows: {}".format(df.duplicated().sum()))
    print("".join(equals))
    
    
def get_zero_var_cols(df):
    zeros = {}
    for col in df.columns:
        vals = df[col].unique()
        if (len(vals) == 1):
            zeros[col] = vals
    if not zeros:
        print("No columns have 0 variation")
    else:
        print(zeros)

def get_model_metrics(actual, predicted):
    model_accuracy = sklearn.metrics.accuracy_score(actual, predicted)
    f1_score = sklearn.metrics.f1_score(actual, predicted)
    precision = sklearn.metrics.precision_score(actual, predicted)
    recall = sklearn.metrics.recall_score(actual, predicted)
    
    print("Accuracy: {}%".format(round(model_accuracy * 100, 2)))
    print("F1 score: ", round(f1_score, 4))
    print("Precision: ", round(precision, 4))
    print("Recall: ", round(recall, 4))