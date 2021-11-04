#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:00:25 2021

@author: chuckzumbaugh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import sklearn.metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import helpers
from xgboost import XGBClassifier

sns.set_style('darkgrid')

df = pd.read_csv('swarm.csv')

helpers.get_dataframe_info(df)
helpers.get_zero_var_cols(df)

# Drop duplicated rows
df = df.drop_duplicates(keep = 'first')

X = df.drop(['Swarm_Behaviour'], axis = 1)
y = df['Swarm_Behaviour']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_scaled = sc.fit_transform(X_train)

pca_comps = np.arange(50, 1050, 50)
pca_variance = []
for n in pca_comps:
    pca = PCA(n_components = n,
              random_state=42)
    pca.fit(X_scaled)
    pca_variance.append(pca.explained_variance_ratio_.sum())

ax = sns.scatterplot(x=pca_comps,
                y=pca_variance,
                color="#966FD6")
ax.set_xlabel("# of Components",
              fontsize=18)
ax.set_ylabel("Explained Variance",
              size=18)
ax.tick_params(labelsize=16)
ax.axvline(x=450,
           color="steelblue",
           linestyle="dashed")

pca = PCA(n_components=450,
          random_state=42)


# params = {'C': np.arange(0.01, 1.1, 0.1),
#           'gamma': np.arange(0.01, 1.1, 0.1)}
# 
# grid = GridSearchCV(SVC(), param_grid=params, cv=5)
# grid.fit(X_scaled, y_train)
# =============================================================================

"""
Fit a SVC model. 
"""
params = {'C': np.arange(0.01, 1.1, 0.1),
          'gamma': np.arange(0.01, 1.1, 0.1)}
grid = GridSearchCV(SVC(random_state=42), param_grid=params, cv=5)
grid.fit(pca.fit_transform(X_scaled), y_train)

pipeln = make_pipeline(StandardScaler(), 
                       PCA(n_components=450,
                           random_state=42), 
                       SVC(C = 0.01, gamma = 0.01, kernel = "linear"))

pipeln.fit(X_train, y_train)


predicted = pipeln.predict(X_test)
# Test set
helpers.get_model_metrics(y_test, predicted)
# Train set
helpers.get_model_metrics(y_train, pipeln.predict(X_train))

"""
Fit a logistic regression model to compare.
"""
pipe_logistic = make_pipeline(StandardScaler(),
                              PCA(n_components=450,
                                  random_state=42),
                              LogisticRegression(penalty='l1',
                                                 solver="saga",
                                                 random_state=42))
pipe_logistic.fit(X_train, y_train)
# Training metrics
helpers.get_model_metrics(y_train, pipe_logistic.predict(X_train))
predicted_logistic = pipe_logistic.predict(X_test)
helpers.get_model_metrics(y_test, predicted_logistic)

xg_pipe = make_pipeline(StandardScaler(),
                        PCA(n_components=450,
                            random_state=42),
                        XGBClassifier(n_estimators=10,
                                      max_depth=4))
xg_pipe.fit(X_train, y_train)
helpers.get_model_metrics(y_train, xg_pipe.predict(X_train))
xg_predicted = xg_pipe.predict(X_test)
helpers.get_model_metrics(y_test, xg_predicted)
