# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:35:35 2018

@author: spm9r
"""

import os
import numpy as np
import pandas as pd
import sklearn as slr
import scipy as sp
import nltk
import time

import csv
from sklearn.decomposition import TruncatedSVD

os.chdir("C://Users//spm9r//eclipse-workspace-spm9r//sys6018-competition-blogger-characteristics")

#tfidf = sp.sparse.load_npz("data//Training_TF_IDF_02.npz")

y = train.age
x = tfidf

# Subset data for quick training
inds = np.random.choice(tfidf.shape[0], 1000, replace=False)

train_small = tfidf[inds, :]
#train_small_test = tfidf[np.random.choice(tfidf.shape[0], 1000, replace=False), :]


# Try SVD
svd = TruncatedSVD(n_components=500)
x_train_svd = svd.fit_transform(train_small)
explained_variance = svd.explained_variance_ratio_.sum()

train_EN_model(x_train_svd, y[inds], x_train_svd)


# SVD on full matrix
svd = TruncatedSVD(n_components=500)
x_train_svd = svd.fit_transform(tfidf)
x_test_svd = svd.transform(tfidf_test)
explained_variance = svd.explained_variance_ratio_.sum()
explained_variance


x_train_all = pd.DataFrame(x_train_svd)
x_train_all = x_train_all.join(pd.get_dummies(train['topic']))
x_train_all = x_train_all.join(pd.get_dummies(train['gender']))

x_test_all = pd.DataFrame(x_test_svd)
x_test_all = x_test_all.join(pd.get_dummies(test['topic']))
x_test_all = x_test_all.join(pd.get_dummies(test['gender']))

train_EN_model(x_train_all, y, x_train_all)

enmodel = slr.linear_model.ElasticNet(l1_ratio = 0.97, alpha = 0.00134, precompute = True)
enmodel.fit(x_train_all, y)

print(enmodel.coef_)
preds = enmodel.predict(x_train_all)

# Do postprocessing of predictions FOR TRAINING SET

outputs = train[['post.id', 'user.id', 'age']]
outputs['age_pred'] = preds
outputs_agg = outputs[['user.id', 'age']]
outputs_agg.drop_duplicates(subset=['user.id', 'age'], keep='last', inplace=True)

outputs = outputs.sort_values(by=['user.id'])
outputs_agg = outputs_agg.sort_values(by=['user.id'])

outputs_agg['age_pred_mean'] = outputs.groupby(['user.id'])['age_pred'].mean().values
outputs_agg['age_pred_sd'] = outputs.groupby(['user.id'])['age_pred'].std().values

outputs_agg['age_pred_m-1'] = outputs_agg['age_pred_mean'] - outputs_agg['age_pred_sd']
outputs_agg['age_pred_m-1'] = np.where(outputs_agg['age_pred_m-1'].isnull(), outputs_agg['age_pred_mean'], outputs_agg['age_pred_m-1'])
#outputs_agg['age_pred_m-2'] = outputs_agg['age_pred_mean'] - 2*outputs_agg['age_pred_sd']

outputs_agg['age_pred_m+1'] = outputs_agg['age_pred_mean'] + outputs_agg['age_pred_sd']
outputs_agg['age_pred_m+1'] = np.where(outputs_agg['age_pred_m+1'].isnull(), outputs_agg['age_pred_mean'], outputs_agg['age_pred_m+1'])
#outputs_agg['age_pred_m+2'] = outputs_agg['age_pred_mean'] + 2*outputs_agg['age_pred_sd']

np.sum(np.square(outputs_agg['age_pred_mean'] - outputs_agg['age']))/12800
np.sum(np.square(outputs_agg['age_pred_m-1'] - outputs_agg['age']))/12800
np.sum(np.square(outputs_agg['age_pred_m+1'] - outputs_agg['age']))/12800



# Do predictions and postprocessing FOR TESTING SET
preds = enmodel.predict(x_test_all)
outputs = test[['post.id', 'user.id']]
outputs['age_pred'] = preds


outputs_agg = outputs[['user.id']]
outputs_agg.drop_duplicates(subset=['user.id'], keep='last', inplace=True)

outputs = outputs.sort_values(by=['user.id'])
outputs_agg = outputs_agg.sort_values(by=['user.id'])
outputs_agg['age_pred_mean'] = outputs.groupby(['user.id'])['age_pred'].mean().values

outputs_agg.to_csv("py_predictions_01.csv")

# Cribbed from https://github.com/wjlei1990/EarlyWarning/blob/master/ml/regressor.py
def train_EN_model(train_x, train_y, predict_x):
    print("ElasticNet")
    #train_x, predict_x = standarize_feature(_train_x, _predict_x)

    #l1_ratios = [1e-4, 1e-3, 1e-2, 1e-1]
    #l1_ratios = [1e-5, 1e-4, 1e-3]
    l1_ratios = [0.9, 0.92, 0.95, 0.97, 0.99]
    best_l1_ratio = -1
    best_alpha = -1
    #l1_ratios = [0.5]
    min_mse = 1
    for r in l1_ratios:
        t1 = time.time()
        reg_en = slr.linear_model.ElasticNetCV(
            l1_ratio=r, cv=5, n_jobs=6, verbose=1, precompute=True)
        reg_en.fit(train_x, train_y)
        n_nonzeros = (reg_en.coef_ != 0).sum()
        _mse = np.mean(reg_en.mse_path_, axis=1)[
            np.where(reg_en.alphas_ == reg_en.alpha_)[0][0]]
        if _mse < min_mse:
            min_mse = _mse
            best_l1_ratio = r
            best_alpha = reg_en.alpha_
        t2 = time.time()
        print("ratio(%e) -- n: %d -- alpha: %f -- mse: %f -- "
              "time: %.2f sec" %
              (r, n_nonzeros, reg_en.alpha_, _mse, t2 - t1))

    print("Best l1_ratio and alpha: %f, %f" % (best_l1_ratio, best_alpha))
    # predict_model
    #reg = slr.linear_model.ElasticNet(l1_ratio=best_l1_ratio, alpha=best_alpha)
    #reg.fit(train_x, train_y)
    #predict_y = reg.predict(predict_x)
    #train_y_pred = reg.predict(train_x)
    return None #{"y": predict_y, "train_y": train_y_pred, "coef": reg.coef_} 