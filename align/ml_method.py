from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from numpy import array
from numpy import asarray
from numpy import zeros
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

df1 = pd.read_excel('banglish total.xlsx').iloc[:,:].values

sentiment = []
labels = []

for i in range(df1.shape[0]):
    try:
        if not np.isnan(df1[i,2]):
            if df1[i,2] != 0:
                sentiment.append(df1[i,0])
                labels.append(df1[i,2])
        else:
            sentiment.append(df1[i,0])
            labels.append(df1[i,1])
    except Exception:
        print('Problem encountered in index ', i)

sentiment = np.array(sentiment).reshape((-1,1))
labels = np.array(labels).reshape((-1,1))

df = np.concatenate((sentiment, labels), axis=1)

df = pd.DataFrame({'text': df[:, 0], 'category': df[:, 1]})


text = df[['text']]
y = df[['category']]

y = y.iloc[:].values.reshape((-1,)).astype('int')

X_train, X_test, y_train, y_test = train_test_split(df['text'].tolist(), y, random_state=5, test_size=0.2, stratify=y)

def data_processing(data):
    data_list = []
    for i in range(len(data)):
        line = re.sub('[^a-zA-Z]', ' ', str(data[i]))
        #lower_words = line.lower()
        tokens = nltk.word_tokenize(line)
        words = [w.lower() for w in tokens]
        data_list.append(words)
    return data_list

X_train = data_processing(X_train)
X_test = data_processing(X_test)

from gensim.models import FastText
#model_ted = FastText.load('saved_model_fastex_mono_supervised')
model_ted = FastText.load('saved_model_fastex_mask_banglish_supervised')
Embedding_Dim = 100

def data_processing(data):
    data_list = []
    for i in range(len(data)):
        line = re.sub('[^a-zA-Z]', ' ', str(data[i]))
        #lower_words = line.lower()
        tokens = nltk.word_tokenize(line)
        words = [w.lower() for w in tokens]
        data_list.append(words)
    return data_list

X_train = data_processing(X_train)
X_test = data_processing(X_test)

def embedding(data, max_len, Embedding_Dim):
    data_lstm = np.zeros((len(data),max_len, Embedding_Dim))
    for i in range(len(data_lstm)):
        news = data[i]
        count = len(news)
        for w in range(len(news)):
            if w >= max_len:
                continue
            else:
                try:
                    first_vec = model_ted[news[w]]
                    vec = first_vec
                    data_lstm[i,w] = vec
                except Exception:
                    print("We can not convert word ", news[w], "to a vector")
                    continue
    data_lstm = np.reshape(data_lstm,(data_lstm.shape[0],data_lstm.shape[1]*data_lstm.shape[2]))
    return data_lstm
            
max_len = 50
X_train_vec = embedding(X_train, max_len, Embedding_Dim)
X_test_vec = embedding(X_test, max_len, Embedding_Dim)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
class_weights = dict(zip(np.unique(y_train), class_weights))

classifier = DecisionTreeClassifier(max_depth=700, min_samples_split=5, class_weight=class_weights)
classifier.fit(X_train_vec, y_train)

#class_weight=class_weights

from sklearn.naive_bayes import GaussianNB, MultinomialNB
classifier = GaussianNB()
classifier.fit(X_train_vec, y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
classifier.fit(X_train_vec, y_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class='auto', solver='newton-cg')
classifier.fit(X_train_vec, y_train)

classifier = RandomForestClassifier(max_depth=800, min_samples_split=5, class_weight=class_weights)
classifier.fit(X_train_vec, y_train)

from xgboost import XGBClassifier
y_train = y_train.reshape((-1,1))
classifier = XGBClassifier()
classifier.fit(X_train_vec, y_train)

classifier.predict(X_train_vec)

accuracy = accuracy_score(y_train, classifier.predict(X_train_vec))
print("Training Accuracy:", accuracy)       
test_predictions = classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:", )
print(confusion_matrix(y_test, test_predictions))
print(classification_report([i for i in y_test],
                            [i for i in test_predictions]))
