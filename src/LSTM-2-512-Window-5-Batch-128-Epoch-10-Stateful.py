
# coding: utf-8

# ### importing require packages

# In[19]:

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
from embeddings import Embeddings
from keras.callbacks import ModelCheckpoint

from nltk.tokenize import word_tokenize
import random


# In[20]:

np.mean([1, 2, 3])


# ## Instantiate Embeddings 

# In[21]:

embeddings = Embeddings(100, 4, 1, 4)


# ### getting data from preprocessing

# In[22]:

word2vec_weights = embeddings.get_weights()
word2index, index2word = embeddings.get_vocabulary()
word2vec_model = embeddings.get_model()
tokenized_indexed_sentences = embeddings.get_tokenized_indexed_sentences()


# ### generating training data

# In[23]:

window_size = 5
vocab_size = len(word2index)
print(vocab_size)
#sorted(window_size,reverse=True)
#sentence_max_length = max([len(sentence) for sentence in tokenized_indexed_sentence ])


# ## Defining model

# In[24]:

model_weights_path = "../weights/LSTM-2-512-Window-5-Batch-128-Epoch-10-Stateful"
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)


# In[25]:

seq_in = []
seq_out = []

# generating dataset
for sentence in tokenized_indexed_sentences:
    sentence_seq_in = []
    sentence_seq_out = []
    for i in range(len(sentence)-window_size-1):
        x = sentence[i:i + window_size]
        y = sentence[i + window_size]
        sentence_seq_in.append(x)#[]
        sentence_seq_out.append(word2vec_weights[y])
    seq_in.append(sentence_seq_in)
    seq_out.append(sentence_seq_out)

# converting seq_in and seq_out into numpy array
seq_in = np.array(seq_in)
seq_out = np.array(seq_out)
n_samples = len(seq_in)
print ("Number of samples : ", n_samples)


# In[26]:

subsamples = np.array([len(seq) for seq in seq_in])
print(np.sum(subsamples))


# In[27]:

subsamples_in = np.array([s for seq in seq_in for s in seq])
subsamples_out = np.array([s for seq in seq_out for s in seq])


# ## Train Model

# In[28]:

np.expand_dims(seq_in[0][0], axis=1)


# In[29]:

total_batches = int(subsamples_in.shape[0] / 256)


# In[30]:

batch_len = []
for i in range(total_batches):
    batch_len.append(len(subsamples_in[i::total_batches]))
min_batch_len = min(batch_len)


# In[31]:

# Changes to the model to be done here
model = Sequential()
model.add(Embedding(input_dim=word2vec_weights.shape[0], output_dim=word2vec_weights.shape[1], weights=[word2vec_weights], batch_input_shape=(min_batch_len, 5)))
model.add(LSTM(512, return_sequences=True, stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(512, stateful=True))
model.add(Dropout(0.1))
model.add(Dense(word2vec_weights.shape[1], activation='sigmoid'))
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[33]:

print("Train")
for epoch in range(15):
    print("Epoch {0}/{1}".format(epoch+1, 15))
    mean_tr_accuracy = []
    mean_tr_loss = []
    for i in range(total_batches):
        # print("Done with {0}/{1} batches".format(i, total_batches))
        train_accuracy, train_loss = model.train_on_batch(subsamples_in[i::total_batches][:min_batch_len], subsamples_out[i::total_batches][:min_batch_len])
        mean_tr_accuracy.append(train_accuracy)
        mean_tr_loss.append(train_loss)
        model.reset_states()
    mean_accuracy = np.mean(mean_tr_accuracy)
    mean_loss = np.mean(mean_tr_loss)
    print("Mean Accuracy", mean_accuracy)
    print("Mean Loss", mean_loss)
    filepath = "../weights/LSTM-2-512-Window-5-Batch-128-Epoch-10-Stateful/weights-{0}-{1}".format(epoch+1, mean_accuracy, mean_loss)
    model.save_weights(filepath)


# ### model predict

# In[73]:

start = 20
samples = subsamples_in[start::total_batches][:min_batch_len]
predictions = model.predict_on_batch(samples)
for index, prediction in enumerate(predictions):
    print(' '.join(index2word[index] for index in samples[index]))
    pred_word = word2vec_model.similar_by_vector(prediction)[0][0]
    sys.stdout.write("*"+pred_word+" \n")


# ## Accuracy

# In[74]:

def accuracy():
    start = 27
    count = 0
    correct = 0
    predictions = model.predict_on_batch(subsamples_in[start::total_batches][:min_batch_len])
    ytrue = subsamples_out[start::total_batches][:min_batch_len]
    for index, prediction in enumerate(predictions):
        pred_word = word2vec_model.similar_by_vector(prediction)[0][0]
        true_word = word2vec_model.similar_by_vector(ytrue[index])[0][0]
        sim = word2vec_model.similarity(pred_word, true_word)
        if (sim >= 0.85):
            correct +=1
        count += 1
    accur = float(correct/(count))
    print('accuracy = ', float(accur))
    


# In[75]:

# n = no. of predictions
print(accuracy())


# In[ ]:



