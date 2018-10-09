
# coding: utf-8

# In[23]:


'''
Tasks:

-- Date preprocessing -- 
    converting dates between languages, and then structure
    --> alternatively, we can duplicate the set, change weird dates to none
    
--feature creation--
    date by year
    looking at if they are posting weekday/weekend might suggest that they are working age
    language (foreign in general or specifics)
    number of posts per user.id
    
-- text --
    remove the stopwords
        NLTK
    remove the punctuation
    stemming the words
    SVD (singular value decomposition) --> will show what is most important by 
        taking the matrix of term counts (needs to be created), but returns importance values ~ variability with each word
    TF-IDF: could filter down to top 500-1000 words (we can probably just implement using a package) 
        if not...
        calculate the frequency of word in every document
        inverse log 
        
-- train model: multiple linear regression --
    glm net --> just make sure not to do log, but can still use this for normal parametric
        (generalization of ridge and lasso regression)
    
    CV procedure to judge accuracy of predictor based on words we keep or parameters of the linear model

    
general notes: what is a document?

    post?
        -this would let us use the distribution of predictions for a user to do a second layer to the model
        -potentially average/take median of different predictions for individual documents
        
        -might want to consider the number of unique pairings between user.id and post.id
        
    single user?
    
    bigrams or trigrams instead of words (two or three words in order can be treated as words)
        n words in sentence --> n-1 bigrams --> blows up dimensionality, but could give more information because context
        comes into play (may or may not be feasible depending on our corpus)
        
        
Note from Sean: We may want to use the "gensim" package, which is Google's pre-trained word2vec model, which I think could be really useful.

'''
("")


# In[24]:


import os
import numpy as np
import pandas as pd
import sklearn as slr
import nltk


# In[25]:


# Set wd
os.chdir("C://Users/alxgr//Documents//UVA//DSI//Fall 2018//SYS//Kaggle Competition 3//sys6018-competition-blogger-characteristics//data")
# os.chdir("/Users/SM/DSI/classes/fall2018/SYS6018/kaggle/sys6018-competition-blogger-characteristics/")


# In[26]:


# read in data
# train = pd.read_csv(os.path.join("data", "input", "train.csv"))
# test = pd.read_csv(os.path.join("data", "input", "test.csv"))

# (os.path did not work for alex.  below for wd on alex's pc)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[27]:


# start data exploration
train.head()


# In[28]:


pd.set_option('display.max_rows', 5000)
train.groupby(['user.id', 'post.id']).size()


# In[29]:


train.shape


# In[30]:


train.describe()


# In[31]:


train.isnull().sum()
test.isnull().sum()


# In[32]:


train.topic.value_counts()


# In[33]:


train.sign.value_counts()


# In[34]:


# join data sets
test['age'] = None
train['set'] = "train"
test['set'] = "test"


# In[35]:


test.head()


# In[36]:


alldata = pd.concat([train, test])


# In[66]:


alldata.isnull().sum()


# In[37]:


# convert to datetime, leaving non-valid/non-English dates null
alldata['date_date'] = pd.to_datetime(alldata.date, errors='coerce')


# In[47]:


# I would recommend removing NA's this way -- it is much easier.
alldata.dropna(inplace = True) 
print(train.shape,"\n",alldata.shape)


# In[79]:


# Remove rows where date cannot be parsed - these are non-English and so unless translated
# into English, cannot be used in a text model with this corpus

alldata = alldata[alldata['date_date'].notnull()]
alldata.date = alldata.date_date
alldata = alldata.drop(['date_date'], axis=1)


# In[20]:


# adjusting age to age at date of posting by subtracting post date from max(post dates)
# This may be a bad idea if ages are already age at time of posting. I think that is actually the case upon inspection.

'''
alldata['adj_age'] = alldata.age - (2006 - alldata['date'].dt.year)
alldata.adj_age = alldata.adj_age.astype(float)
'''
("")


# In[108]:





# In[109]:


alldata.head()


# In[110]:


alldata.describe()

