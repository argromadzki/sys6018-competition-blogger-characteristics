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

os.chdir("C://Users//spm9r//eclipse-workspace-spm9r//sys6018-competition-blogger-characteristics")

tfidf = sp.sparse.load_npz("data//Training_TF_IDF_02.npz")

y = train.age
x = tfidf

# Subset data for quick training

train_small = tfidf[np.random.choice(tfidf.shape[0], 1000, replace=False), :]
train_small_test = tfidf[np.random.choice(tfidf.shape[0], 1000, replace=False), :]

results = train_EN_model(tfidf[train_small.indices,:], y[train_small.indices], tfidf[train_small_test.indices,:])


# Cribbed from https://github.com/wjlei1990/EarlyWarning/blob/master/ml/regressor.py
def train_EN_model(train_x, train_y, predict_x):
    print("ElasticNet")
    #train_x, predict_x = standarize_feature(_train_x, _predict_x)

    #l1_ratios = [1e-4, 1e-3, 1e-2, 1e-1]
    #l1_ratios = [1e-5, 1e-4, 1e-3]
    l1_ratios = [0.9, 0.92, 0.95, 0.97, 0.99]
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
    reg = slr.linear_model.ElasticNet(l1_ratio=best_l1_ratio, alpha=best_alpha)
    reg.fit(train_x, train_y)
    predict_y = reg.predict(predict_x)
    train_y_pred = reg.predict(train_x)
    return {"y": predict_y, "train_y": train_y_pred, "coef": reg.coef_} 