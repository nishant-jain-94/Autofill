
# coding: utf-8

# ### Importing require packages

# In[1]:

from __future__ import print_function

import json
import os
import numpy as np
import sys
import h5py

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras.engine import Input
from keras.layers import Embedding, merge
from keras.models import Model
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


# ## Setting Parameters

# In[34]:

model_name = "lstm-2-1024-300-batchsize-512-epochs-25-Sequence"
word_embedding_dimension = 300
word_embedding_window_size = 4
batch_size = 512 # 32, 64, 128
epochs = 25 # 10, 15, 30
window_size = "None" # 3, 4, 5
accuracy_threshold = 1
activation = 'softmax' # sigmoid, relu, softmax
custom_accuracy = 0
loss_function = 'mse' # mse


# ## Instantiate Embeddings 

# In[3]:

embeddings = Embeddings(word_embedding_dimension, word_embedding_window_size, 1, 4)


# ### Getting data from preprocessing

# In[4]:

word2vec_model = embeddings.get_intersected_model()
word2index, index2word = embeddings.get_vocabulary()
word2vec_weights = word2vec_model.wv.syn0
tokenized_indexed_sentences = embeddings.get_indexed_sentences()


# In[5]:

word2index = {word:index+1 for word, index in word2index.items()}
index2word = {index:word for word, index in word2index.items()}


# In[6]:

word2index


# In[7]:

tokenized_indexed_sentences[0]


# In[8]:

tokenized_indexed_sentences = [np.array(sentence) + 1 for sentence in tokenized_indexed_sentences if len(sentence) > 0]


# In[9]:

tokenized_indexed_sentences[0]


# In[10]:

new_weights = np.zeros((1, word2vec_weights.shape[1]))


# In[11]:

new_weights = np.append(new_weights, word2vec_weights, axis=0)


# ### generating training data

# In[12]:

# window_size = 5
vocab_size = len(word2index)
print(vocab_size)


# In[13]:

maxlen = max([len(sentence) for sentence in tokenized_indexed_sentences])


# In[14]:

tokenized_indexed_sentences = sequence.pad_sequences(tokenized_indexed_sentences)


# In[19]:

seq_in = np.zeros_like(tokenized_indexed_sentences)
seq_out = np.zeros((tokenized_indexed_sentences.shape[0], tokenized_indexed_sentences.shape[1], 300))

# Generating Dataset
for index, sentence in enumerate(tokenized_indexed_sentences):
    y = np.append(sentence[1:], np.array(sentence[len(sentence)-1]))
    seq_in[index] += sentence
    seq_out[index] += [new_weights[i] for i in y]

n_samples = len(seq_in)
print ("Number of samples : ", n_samples)


# ## Defining model

# In[35]:

# Changes to the model to be done here
model = Sequential()
model.add(Embedding(input_dim=new_weights.shape[0], output_dim=new_weights.shape[1], weights=[new_weights], mask_zero=True))
model.add(LSTM(1024,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300, return_sequences=True, activation=activation))
# model.load_weights("../weights/lstm-2-1024-300-batchsize-512-epochs-25-Sequence/weights.24.hdf5")
model.compile(loss=loss_function, optimizer='adam',metrics=['accuracy'])
model.summary()


# ## Creating Weights Directory

# In[36]:

model_weights_path = "../weights/" + model_name
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/weights.{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=False, mode='max')


# ## Train Model

# In[37]:

model_fit_summary = model.fit(seq_in, seq_out, epochs=epochs, verbose=1, batch_size=batch_size, callbacks=[checkpoint])


# ## Predictions

# In[38]:

start = 0
sentence_test = "In which regions in particular did"
indexed_sentences = embeddings.get_indexed_query(sentence_test)
print("indexed_sentences ",indexed_sentences)
sent = np.array(indexed_sentences)
#pattern = list(seq_in[start])
pattern = list(sent)
print("\"",' '.join(index2word[index] for index in pattern))
for i in range(10):
    prediction = model.predict(np.array([pattern]))
    pred_word = word2vec_model.similar_by_vector(prediction[0][prediction.shape[1] - 1])[0][0]
    sys.stdout.write(pred_word+" ")
    pattern.append(word2index[pred_word])
    pattern = pattern[:len(pattern)]


# ## Model Summary

# In[33]:

model_results = model_fit_summary.history
model_results.update(model_fit_summary.params)
model_results["word_embedding_dimension"] = word_embedding_dimension
model_results["word_embedding_window_size"] = word_embedding_window_size
model_results["window_size"] = window_size
model_results["batch_size"] = batch_size
model_results["epochs"] = epochs
model_results["model_name"] = model_name
model_results["accuracy_threshold"] = accuracy_threshold
model_results["activation"] = activation 
model_results["custom_accuracy"] = custom_accuracy
model_results["loss_function"] = loss_function
model_results["layers"] = []
model_results["dropouts"] = []
for layer in model.layers:
    if hasattr(layer, "units"):
        layer_summary = {}
        layer_summary["units"] = layer.get_config()["units"]
        layer_summary["name"] = layer.name
        model_results["layers"].append(layer_summary)
    if hasattr(layer, "rate"):
        dropout_summary = {}
        dropout_summary["rate"] = layer.get_config()["rate"]
        model_results["dropouts"].append(dropout_summary)
text_file_path = "../weights/{0}/model_results.json".format(model_name)
with open(text_file_path, "w") as f:
        json.dump(model_results, f)

