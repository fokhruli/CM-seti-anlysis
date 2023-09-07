import numpy as np 
import pandas as pd
import re
import gc
import os
import fileinput
import string
#import tensorflow as tf
import zipfile
import datetime
import sys
from tqdm  import tqdm
tqdm.pandas()
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer

# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import copy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class_num = 5
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
convert_dict = {'category': float}
y = y.astype(convert_dict)

X_train, X_test, y_train, y_test = train_test_split(df['text'].tolist(), y, random_state=5, test_size=0.2, stratify=y)


from gensim.models import FastText
#model_ted = FastText.load('saved_model_fastex_mask_banglish_supervised')
model_ted = FastText.load('saved_model_fastex_mono_supervised')

Embedding_Dim = 100

def data_processing(data):
    data_list = []
    for i in range(len(data)):
        line = re.sub('[^a-zA-Z]', ' ', str(data[i]))
        tokens = nltk.word_tokenize(line)
        words = [w.lower() for w in tokens]
        data_list.append(words)
    return data_list

X_train = data_processing(X_train)
X_test = data_processing(X_test)

X_train_sent = X_train
X_test_sent = X_test
y_trainv0 = y_train.iloc[:].values.reshape((-1,)).astype('int')-1
y_testv0 = y_test.iloc[:].values.reshape((-1,)).astype('int')-1

max_len = max([len(x) for x in X_train_sent])

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
    return data_lstm
            
max_len = 50
X_train_vec = embedding(X_train, max_len, Embedding_Dim)
X_test_vec = embedding(X_test, max_len, Embedding_Dim)
 
def label_generator(data, class_num):
    one_hot = np.zeros((data.shape[0],class_num))
    for i in range(data.shape[0]):
        val = int(data[i,0])-1
        one_hot[i,val] = 1
    return one_hot
y_train = label_generator(y_train.iloc[:].values, class_num)
y_test = label_generator(y_test.iloc[:].values, class_num)

vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(sequences, maxlen=max_len)
sequences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(sequences, maxlen=max_len)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_trainv0),
                                        y = y_trainv0                                                    
                                    )
class_weights = dict(zip(np.unique(y_trainv0), class_weights))

seq_input = Input(shape=(X_train_vec.shape[1], X_train_vec.shape[2]), batch_size=None)
seq_one_hot = Input(shape=(X_train.shape[1]), batch_size=None)

embd = tf.keras.layers.Embedding(20000, 100, input_length=50)(seq_one_hot)
embd = seq_input + embd
embd = tf.keras.layers.Dropout(0.3)(embd)
conv1 = tf.keras.layers.Conv1D(64, 5, activation='relu')(embd)
conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
lstm = tf.keras.layers.LSTM(100)(conv1)
out = tf.keras.layers.Dense(class_num, activation='softmax')(lstm)

model = Model(inputs=[seq_one_hot, seq_input], outputs=out)

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.01), metrics=['accuracy'])

checkpoint = ModelCheckpoint("best model/best_model.hdf5", monitor='val_accuracy', save_best_only=True, mode='auto', period=1)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
model.fit([X_train, X_train_vec], y_train,
                    batch_size=128,
                    epochs=25,
                    verbose=1,
                    validation_data=([X_test, X_test_vec], y_test), callbacks=[checkpoint, callback])

preds = model.predict([X_test, X_test_vec])

print(classification_report(np.argmax(y_test,axis=1),np.argmax(preds,axis=1)))


