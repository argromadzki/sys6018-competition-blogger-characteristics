# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:17:08 2018

@author: spm9r
"""


import os
import numpy as np
import pandas as pd
import sklearn as slr
import scipy as sp
import nltk


# Set wd
#os.chdir("C://Users/alxgr//Documents//UVA//DSI//Fall 2018//SYS//Kaggle Competition 3//sys6018-competition-blogger-characteristics")
#os.chdir("/Users/SM/DSI/classes/fall2018/SYS6018/kaggle/sys6018-competition-blogger-characteristics/")
os.chdir("C://Users//spm9r//eclipse-workspace-spm9r//sys6018-competition-blogger-characteristics")

# read in data
train = pd.read_csv(os.path.join("data", "input", "train.csv"))
test = pd.read_csv(os.path.join("data", "input", "test.csv"))

# convert to datetime, leaving non-valid/non-English dates null
train['date_date'] = pd.to_datetime(train.date, errors='coerce')
test['date_date'] = pd.to_datetime(test.date, errors='coerce')

# Remove rows where date cannot be parsed - these are non-English and so unless translated
# into English, cannot be used in a text model with this corpus
train = train[train['date_date'].notnull()]
train.date = train.date_date
train = train.drop(['date_date'], axis=1)

# Optional: concatenate posts to single 
train_text = train[['post.id','user.id','gender','topic','sign','date','text', 'age']].groupby(['user.id'])['text'].transform(lambda x: ','.join(x))
train['text'] = train_text
train.drop_duplicates(subset=['user.id'], keep='last', inplace=True)

test_text = test[['post.id','user.id','gender','topic','sign','date','text']].groupby(['user.id'])['text'].transform(lambda x: ','.join(x))
test['text'] = test_text
test.drop_duplicates(subset=['user.id'], keep='last', inplace=True)

porter = nltk.stem.porter.PorterStemmer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# lowercase, tokenize & stem words in text
train['text'] = train['text'].apply(str.lower)
train['tokens']=train['text'].apply(nltk.word_tokenize)
train['tokens']=[[porter.stem(x) for x in tokens] for tokens in train['tokens']]
train['tokens']=[[w for w in tokens if not w in stop_words] for tokens in train['tokens']]
train['tokens']=[[w for w in tokens if w.isalpha()] for tokens in train['tokens']]
train['newtext'] = train['tokens'].apply(lambda x: " ".join(x))

# lowercase, tokenize & stem words in text
test['text'] = test['text'].apply(str.lower)
test['tokens']=test['text'].apply(nltk.word_tokenize)
test['tokens']=[[porter.stem(x) for x in tokens] for tokens in test['tokens']] # triggers a bug in nltk 3.2
test['tokens']=[[w for w in tokens if not w in stop_words] for tokens in test['tokens']]
test['tokens']=[[w for w in tokens if w.isalpha()] for tokens in test['tokens']]
test['newtext'] = test['tokens'].apply(lambda x: " ".join(x))

#pd.write_csv('data//Corpus.csv')

# Read in product of the above
train = pd.read_csv('data//Corpus.csv')
test = pd.read_csv('data//Testing_Corpus.csv') # FIX change this

# Create Term-Document Matrix
#vectorizer = slr.feature_extraction.text.CountVectorizer()
#tdm = vectorizer.fit_transform(train['newtext'])

# Create reduced-dimension TDM
vectorizer_min = slr.feature_extraction.text.CountVectorizer(min_df=.0005)
tdm_reduced = vectorizer_min.fit_transform(train['newtext'].astype('U'))
tdm_test_reduced = vectorizer_min.transform(test['newtext'].astype('U'))

# TFIDF transform
transformer = slr.feature_extraction.text.TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(tdm_reduced)
tfidf_test = transformer.transform(tdm_test_reduced)

sp.sparse.save_npz("data//Training_TF_IDF_02.npz", tfidf)
sp.sparse.save_npz("data//Testing_TF_IDF_02.npz", tfidf_test)


