import scipy.linalg
import pandas as pd
import numpy as np
from gensim.models import FastText
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import random

dic = pd.read_csv('Book1.csv').iloc[:,:].values
df = pd.read_csv('sentiment_corpus.csv').iloc[0:5000,:].values
corpus = df[:,0]

def tokenize(data, sampling_rate):
    data_list = []
    for i in range(len(data)):
        line = re.sub('[^a-zA-Z]', ' ', str(data[i]))
        #lower_words = line.lower()
        tokens = nltk.word_tokenize(line)
        words = [w.lower() for w in tokens]
        data_list.append(words)
        for rate in sampling_rate:
            idxA = 0
            idxB = 0
            new_tokens = []
            for word in words:
                if word in dic and idxA == 0:
                    context_index = np.where(dic == word)
                    alpha = random.randint(0,len(context_index[0])-1)
                    row = context_index[0][alpha]
                    if context_index[1][0] == 0:
                        col = 1
                    else:
                        col = 0
                    context_word = str(dic[row,col])
                    new_tokens.append(context_word)
                    idxA = 1
                else:
                    new_tokens.append(word)
                    idxB += 1
                    #idxA = 0
                if idxB >= rate:
                    idxA = 0
                    idxB = 0
            if len(new_tokens) > 1:
                data_list.append(new_tokens)
            else:
                continue
    return data_list

tokens = tokenize(corpus, sampling_rate = [1,2,3])

from gensim.models import FastText
import gensim
Embedding_Dim = 100
#model_ted = FastText(tokens, size=Embedding_Dim, window=5, min_count=1, workers=4,sg=1)
model_ted = gensim.models.Word2Vec(tokens, size=Embedding_Dim, window=5, min_count=1, workers=4,sg=1)

#Test similar words
model_ted.wv.most_similar('good',topn=15)
model_ted.save('saved_model_w2v_mask_banglish_zero_shot')
