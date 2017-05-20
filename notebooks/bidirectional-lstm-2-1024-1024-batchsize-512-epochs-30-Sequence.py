
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
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.preprocessing import sequence
from intersect_embeddings import Embeddings
from keras.callbacks import ModelCheckpoint

from nltk.tokenize import word_tokenize
import random
from itertools import groupby


# ## Instantiate Embeddings 

# In[2]:

embeddings = Embeddings(300, 4, 1, 4)


# ### Getting data from preprocessing

# In[3]:

word2vec_model = embeddings.get_intersected_model()
word2index, index2word = embeddings.get_vocabulary()
word2vec_weights = word2vec_model.wv.syn0
tokenized_indexed_sentences = embeddings.get_indexed_sentences()


# In[4]:

word2index = {word:index+1 for word, index in word2index.items()}
index2word = {index:word for word, index in word2index.items()}


# In[5]:

word2index


# In[6]:

tokenized_indexed_sentences[0]


# In[7]:

tokenized_indexed_sentences = [np.array(sentence) + 1 for sentence in tokenized_indexed_sentences if len(sentence) > 0]


# In[8]:

tokenized_indexed_sentences[0]


# In[9]:

new_weights = np.zeros((1, word2vec_weights.shape[1]))


# In[10]:

new_weights = np.append(new_weights, word2vec_weights, axis=0)


# In[11]:

new_weights.shape


# In[12]:

new_weights[52730]


# ### generating training data

# In[13]:

window_size = 5
vocab_size = len(word2index)
print(vocab_size)


# In[14]:

maxlen = max([len(sentence) for sentence in tokenized_indexed_sentences])


# In[15]:

tokenized_indexed_sentences = sequence.pad_sequences(tokenized_indexed_sentences)


# In[16]:

seq_in = []
seq_out = []
# generating dataset
tokenized_indexed_sentences = [sentence for sentence in tokenized_indexed_sentences if len(sentence) > 0]
for sentence in tokenized_indexed_sentences:
    x = sentence
    y = np.append(sentence[1:], np.array(sentence[len(sentence)-1]))
    seq_in.append(x)
    seq_out.append([new_weights[index] for index in y])

# converting seq_in and seq_out into numpy array
seq_in = np.array(seq_in)
seq_out = np.array(seq_out)
n_samples = len(seq_in)
print ("Number of samples : ", n_samples)


# ## Defining model

# In[17]:

# Changes to the model to be done here
model = Sequential()
model.add(Embedding(input_dim=new_weights.shape[0], output_dim=new_weights.shape[1], weights=[new_weights], mask_zero=True))
model.add(Bidirectional(LSTM(1024, return_sequences=True), merge_mode="average"))
model.add(Bidirectional(LSTM(300, return_sequences=True), merge_mode="average"))
model.compile(loss='cosine_proximity', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[18]:

model_weights_path = "../weights/lstm-3-1024-1024-batchsize-512-epochs-30-Sequence"
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/weights.{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=False, mode='max')


# ## Train Model

# In[ ]:

model.fit(seq_in, seq_out, epochs=1, verbose=1, batch_size=512, callbacks=[checkpoint])


# ### model predict

# In[32]:

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


# In[ ]:

#e_model = embeddings.get_model()


# In[ ]:

#e_model.similar_by_word("profitabl")


# ## Accuracy

# In[ ]:

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


# In[ ]:

#seq_out[0]


# In[ ]:

# accuracy()


# In[ ]:

#model_results = model_fit_summary.history


# In[ ]:

#model_results.update(model_fit_summary.params)


# In[ ]:

#model_results["train_accuracy"] = accuracy()


# In[ ]:

# n = no. of predictions
# accuracy = accuracy(400)
#print(model_results)


# In[ ]:

#text_file_path = "../weights/lstm-2-1024-512-batchsize-128-epochs-25/model_results.json"


# In[ ]:

#with open(text_file_path, "w") as f:
        #json.dump(model_results, f)
        


# In[ ]:



