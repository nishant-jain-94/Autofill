
# coding: utf-8

# ### importing require packages

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
from embeddings import Embeddings
from keras.callbacks import ModelCheckpoint

from nltk.tokenize import word_tokenize
import random


# ## Instantiate Embeddings 

# In[2]:

embeddings = Embeddings(100, 4, 1, 4)


# ### getting data from preprocessing

# In[3]:

word2vec_weights = embeddings.get_weights()
word2index, index2word = embeddings.get_vocabulary()
word2vec_model = embeddings.get_model()
tokenized_indexed_sentences = embeddings.get_tokenized_indexed_sentences()


# ### generating training data

# In[4]:

window_size = 5
vocab_size = len(word2index)
print(vocab_size)


# In[5]:

seq_in = []
seq_out = []
# generating dataset
for sentence in tokenized_indexed_sentences:
    for i in range(len(sentence)-window_size-1):
        x = sentence[i:i + window_size]
        y = sentence[i + window_size]
        seq_in.append(x)#[]
        seq_out.append(word2vec_weights[y])

# converting seq_in and seq_out into numpy array
seq_in = np.array(seq_in)
seq_out = np.array(seq_out)
n_samples = len(seq_in)
print ("Number of samples : ", n_samples)


# ## Defining model

# In[6]:

# Changes to the model to be done here
model = Sequential()
model.add(Embedding(input_dim=word2vec_weights.shape[0], output_dim=word2vec_weights.shape[1], weights=[word2vec_weights]))
model.add(LSTM(1024,return_sequences =True))
model.add(Dropout(0.2))
model.add(LSTM(512))
#model.add(Dropout(0.2))
model.add(Dense(word2vec_weights.shape[1], activation='relu'))
#model.load_weights("../weights/lstm-2-1024-512-batchsize-128-epochs-25/weights.24-0.22.hdf5")

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[7]:

model_weights_path = "../weights/lstm-2-1024-512-batchsize-128-epochs-25"
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')


# ## Train Model

# In[8]:

model_fit_summary = model.fit(seq_in, seq_out, epochs=25, verbose=1, validation_split=0.2, batch_size=128, callbacks=[checkpoint])


# ### model predict

# In[21]:

start = 100
pattern = list(seq_in[start])
print("\"",' '.join(index2word[index] for index in pattern))
for i in range(10):
    prediction = model.predict(np.array([pattern]))
    pred_word = word2vec_model.similar_by_vector(prediction[0])[0][0]
    sys.stdout.write(pred_word+" ")
    pattern.append(word2index[pred_word])
    pattern = pattern[1:len(pattern)]


# ## Accuracy

# In[32]:

def accuracy():
    count = 0
    correct = 0
    for sub_sample_in, sub_sample_out in zip(seq_in, seq_out):
        ypred = model.predict_on_batch(np.expand_dims(sub_sample_in, axis=0))[0]
        ytrue = sub_sample_out
        pred_word = word2vec_model.similar_by_vector(ypred)[0][0]
        true_word = word2vec_model.similar_by_vector(ytrue)[0][0]
        similarity = word2vec_model.similarity(pred_word, true_word)
        if similarity >= .85:
            correct += 1
        count += 1
    print("Accuracy {0}".format(correct/count))


# In[33]:

#seq_out[0]


# In[38]:

model_results = model_fit_summary.history


# In[35]:

model_results.update(model_fit_summary.params)


# In[36]:

model_results["train_accuracy"] = accuracy()


# In[37]:

accuracy = accuracy()


# In[26]:

text_file_path = "../weights/lstm-2-1024-512-batchsize-128-epochs-25/model_results.json"


# In[27]:

with open(text_file_path, "w") as f:
       json.dump(model_results, f)
        


# In[ ]:



