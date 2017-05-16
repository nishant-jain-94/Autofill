
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

window_size = 3
vocab_size = len(word2index)
print(vocab_size)
#sorted(window_size,reverse=True)
#sentence_max_length = max([len(sentence) for sentence in tokenized_indexed_sentence ])


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

# In[18]:

# Changes to the model to be done here
model = Sequential()
model.add(Embedding(input_dim=word2vec_weights.shape[0], output_dim=word2vec_weights.shape[1], weights=[word2vec_weights]))
model.add(LSTM(512,return_sequences =True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dense(word2vec_weights.shape[1], activation='relu'))
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[19]:

model_weights_path = "../weights/LSTM_1_Layer"
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')


# ## Train Model

# In[20]:

model.fit(seq_in, seq_out, epochs=10, verbose=1, validation_split=0.2, batch_size=128, callbacks=[checkpoint])


# ### model predict

# In[21]:

start = 97
pattern = list(seq_in[start])
print("\"",' '.join(index2word[index] for index in pattern))
for i in range(10):
    prediction = model.predict(np.array([pattern]))
    pred_word = word2vec_model.similar_by_vector(prediction[0])[0][0]
    sys.stdout.write(pred_word+" ")
    pattern.append(word2index[pred_word])
    pattern = pattern[1:len(pattern)]


# ## Accuracy

# In[22]:

def accuracy(no_of_preds):
    correct=0
    wrong=0
    random.seed(1)
    for i in random.sample(range(0, seq_in.shape[0]), no_of_preds):
        start = i
        sentence = list(seq_in[start])
        prediction = model.predict(np.array([sentence]))
        pred_word = word2vec_model.similar_by_vector(prediction[0])[0][0]
        next_word_index = list(seq_in[start+1])
        next_word = index2word[next_word_index[-1]]
        sim = word2vec_model.similarity(pred_word,next_word)
        if (sim >= 0.6):
            correct +=1
        else : wrong +=1
    print('correct: '+str(correct)+(' wrong: ')+str(wrong))
    accur = float(correct/(correct+wrong))
    print('accuracy = ',float(accur))
    


# In[23]:

# n = no. of predictions
accuracy(400)


# In[ ]:



