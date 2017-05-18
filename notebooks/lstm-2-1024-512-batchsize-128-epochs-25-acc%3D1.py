
# coding: utf-8

# ### importing require packages

# In[14]:

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

# In[15]:

embeddings = Embeddings(100, 4, 1, 4)


# ### getting data from preprocessing

# In[16]:

word2vec_weights = embeddings.get_weights()
word2index, index2word = embeddings.get_vocabulary()
word2vec_model = embeddings.get_model()
tokenized_indexed_sentences = embeddings.get_tokenized_indexed_sentences()


# ### generating training data

# In[17]:

window_size = 5
vocab_size = len(word2index)
print(vocab_size)


# In[18]:

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


# In[19]:

seq_in.shape


# ## Defining model

# In[20]:

# Changes to the model to be done here
model = Sequential()
model.add(Embedding(input_dim=word2vec_weights.shape[0], output_dim=word2vec_weights.shape[1], weights=[word2vec_weights]))
model.add(LSTM(1024,return_sequences =True))
model.add(Dropout(0.2))
model.add(LSTM(512))
#model.add(Dropout(0.2))
model.add(Dense(word2vec_weights.shape[1], activation='relu'))
model.load_weights("../weights/lstm-2-1024-512-batchsize-128-epochs-25/weights.24-0.22.hdf5")
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[21]:

model_weights_path = "../weights/lstm-2-1024-512-batchsize-128-epochs-25"
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')


# ## Train Model

# In[22]:

#model_fit_summary = model.fit(seq_in, seq_out, epochs=25, verbose=1, validation_split=0.2, batch_size=128, callbacks=[checkpoint])


# ### model predict

# In[36]:

np.array(pattern[0])


# In[43]:

list(sent[0])


# In[42]:

list(seq_in)


# In[64]:

start = 0
sentence_test = "In which regions in particular did"
indexed_sentences = embeddings.tokenize_index_sentence(sentence_test)
print("indexed_sentences ",indexed_sentences)
sent = np.array(indexed_sentences)
#pattern = list(seq_in[start])
pattern = list(sent[start])
print("\"",' '.join(index2word[index] for index in pattern))
for i in range(5):
    prediction = model.predict(np.array([pattern]))
    pred_word = word2vec_model.similar_by_vector(prediction[0])[0][0]
    sys.stdout.write(pred_word+" ")
    pattern.append(word2index[pred_word])
    pattern = pattern[:len(pattern)]


# In[54]:

e_model = embeddings.get_model()


# In[63]:

e_model.similar_by_word("profitabl")


# ## Accuracy

# In[74]:

def accuracy():
    count = 0
    correct = 0
    for sub_sample_in, sub_sample_out in zip(seq_in, seq_out):
        ypred = model.predict_on_batch(np.expand_dims(sub_sample_in, axis=0))[0]
        ytrue = sub_sample_out
        pred_word = word2vec_model.similar_by_vector(ypred)[0][0]
        true_word = word2vec_model.similar_by_vector(ytrue)[0][0]
        similarity = word2vec_model.similarity(pred_word, true_word)
        if similarity == 1:
            correct += 1
        count += 1
    print("Accuracy {0}".format(correct/count))


# In[75]:

#seq_out[0]


# In[ ]:

accuracy()


# In[68]:

model_results = model_fit_summary.history


# In[23]:

model_results.update(model_fit_summary.params)


# In[67]:

model_results["train_accuracy"] = accuracy()


# In[28]:

# n = no. of predictions
# accuracy = accuracy(400)
#print(model_results)


# In[26]:

text_file_path = "../weights/lstm-2-1024-512-batchsize-128-epochs-25/model_results.json"


# In[27]:

with open(text_file_path, "w") as f:
        json.dump(model_results, f)
        


# In[ ]:



