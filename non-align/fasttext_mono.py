import scipy.linalg
import pandas as pd
import numpy as np
from gensim.models import FastText
import re
import nltk
#nltk.download('stopwords')
#nltk.download('words')
#nltk.download('wordnet')
from nltk.corpus import stopwords
#import enchant 
#from banglish_detector import Banglish_English_Detector
from collections import Counter

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
corpus = df[:,0]

def tokenize(data):
    data_list = []
    for i in range(len(data)):
        line = re.sub('[^a-zA-Z]', ' ', str(data[i]))
        #lower_words = line.lower()
        tokens = nltk.word_tokenize(line)
        words = [w.lower() for w in tokens]
        data_list.append(words)
    return data_list

tokens = tokenize(corpus)

from gensim.models import FastText
Embedding_Dim = 250

model_ted = FastText(tokens, size=Embedding_Dim, window=5, min_count=1, workers=4,sg=1)

#Test similar words
model_ted.wv.most_similar('good',topn=15)

model_ted.save('saved_model_fastex_mono_supervised_250')

