from __future__ import print_function
import json
import os
import numpy as np
import sys
import h5py
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.preprocessing import sequence
from intersect_embeddings import Embeddings
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import word_tokenize
import random
from itertools import groupby

# ## Instantiate Embeddings 
embeddings = Embeddings(300, 4, 1, 4)

# ### Getting data from preprocessing
word2vec_model = embeddings.get_intersected_model()
word2index, index2word = embeddings.get_vocabulary()
word2vec_weights = word2vec_model.wv.syn0
tokenized_indexed_sentences = embeddings.get_indexed_sentences()

word2index = {word:index+1 for word, index in word2index.items()}
index2word = {index:word for word, index in word2index.items()}

new_weights = np.zeros((1, word2vec_weights.shape[1]))
new_weights = np.append(new_weights, word2vec_weights, axis = 0)


# ## Defining model
# Changes to the model to be done here
model = Sequential()
model.add(Embedding(input_dim = new_weights.shape[0], output_dim = new_weights.shape[1], weights = [new_weights], mask_zero = True))
model.add(LSTM(1024, return_sequences = True))
model.add(LSTM(1024, return_sequences = True))
model.add(LSTM(300, return_sequences = True))
model.load_weights("../weights/lstm-3-1024-1024-batchsize-512-epochs-30-Sequence/weights.29.hdf5")
model.compile(loss='cosine_proximity', optimizer='adam',metrics=['accuracy'])
model.summary()


# ### model predict
def predict_next(input_sent, n):
    indexed_sentences = embeddings.get_indexed_query(input_sent)
    sent = np.array(indexed_sentences) + 1
    pattern = list(sent)
    return_sent = ' '.join(index2word[index] for index in pattern if index!=0)
    for i in range(n):
        prediction = model.predict(np.array([pattern]))
        pred_word = word2vec_model.similar_by_vector(prediction[0][prediction.shape[1] - 1])[0][0]
        return_sent += " "
        return_sent += pred_word
        pattern.append(word2index[pred_word])
        pattern = pattern[:len(pattern)]
    return_sent = [word for word in return_sent.split(' ') if word not in ['squadstart','squadend']]
    return ' '.join(return_sent)
