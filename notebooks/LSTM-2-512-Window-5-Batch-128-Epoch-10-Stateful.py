
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


# In[2]:

np.mean([1, 2, 3])


# ## Instantiate Embeddings 

# In[ ]:

embeddings = Embeddings(100, 4, 1, 4)


# ### getting data from preprocessing

# In[ ]:

word2vec_weights = embeddings.get_weights()
word2index, index2word = embeddings.get_vocabulary()
word2vec_model = embeddings.get_model()
tokenized_indexed_sentences = embeddings.get_tokenized_indexed_sentences()


# ### generating training data

# In[ ]:

window_size = 5
vocab_size = len(word2index)
print(vocab_size)
#sorted(window_size,reverse=True)
#sentence_max_length = max([len(sentence) for sentence in tokenized_indexed_sentence ])


# ## Defining model

# In[ ]:

model_weights_path = "../weights/LSTM-2-512-Window-5-Batch-1-Epoch-10-Stateful"
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)


# In[ ]:

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


# In[ ]:

subsamples = np.array([len(seq) for seq in seq_in])
print(np.sum(subsamples))


# In[ ]:

subsamples_in = np.array([s for seq in seq_in for s in seq])
subsamples_out = np.array([s for seq in seq_out for s in seq])


# ## Train Model

# In[ ]:

np.expand_dims(seq_in[0][0], axis=1)


# In[ ]:

total_batches = int(subsamples_in.shape[0] / 256)


# In[ ]:

batch_len = []
for i in range(total_batches):
    batch_len.append(len(subsamples_in[i::total_batches]))
min_batch_len = min(batch_len)


# In[ ]:

# Changes to the model to be done here
model = Sequential()
model.add(Embedding(input_dim=word2vec_weights.shape[0], output_dim=word2vec_weights.shape[1], weights=[word2vec_weights], batch_input_shape=(min_batch_len, 5)))
model.add(LSTM(512, return_sequences=True, stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.1))
model.add(Dense(word2vec_weights.shape[1], activation='sigmoid'))
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:

print("Train")
for epoch in range(15):
    print("Epoch {0}/{1}".format(epoch+1, 15))
    mean_tr_accuracy = []
    mean_tr_loss = []
    for i in range(total_batches):
        print("Done with {0}/{1} batches".format(i, total_batches))
        train_accuracy, train_loss = model.train_on_batch(subsamples_in[i::total_batches][:min_batch_len], subsamples_out[i::total_batches][:min_batch_len])
        mean_tr_accuracy.append(train_accuracy)
        mean_tr_loss.append(train_loss)
        model.reset_states()
    mean_accuracy = np.mean(mean_tr_accuracy)
    mean_loss = np.mean(mean_tr_loss)
    print("Mean Accuracy", mean_tr_accuracy)
    print("Mean Loss", mean_tr_loss)
    filepath = "../weights/LSTM-2-512-Window-5-Batch-128-Epoch-10-Stateful/weights-{0}-{1}".format(epoch+1, mean_accuracy, mean_loss)
    model.save_weights(filepath)


# ### model predict

# In[ ]:

start = 97
pattern = list(subsamples_in[start])
print("\"",' '.join(index2word[index] for index in pattern))
for i in range(10):
    prediction = model.predict(np.array([pattern]))
    pred_word = word2vec_model.similar_by_vector(prediction[0])[0][0]
    sys.stdout.write(pred_word)
    pattern.append(word2index[pred_word])
    pattern = pattern[1:len(pattern)]


# ## Accuracy

# In[ ]:

def accuracy(no_of_preds):
    correct=0
    wrong=0
    random.seed(1)
    for i in random.sample(range(0, subsamples_in.shape[0]), no_of_preds):
        start = i
        sentence = list(seq_in[start])
        prediction = model.predict(np.array([sentence]))
        pred_word = word2vec_model.similar_by_vector(prediction[0])[0][0]
        next_word_index = list(subsamples_out[start+1])
        next_word = index2word[next_word_index[-1]]
        sim = word2vec_model.similarity(pred_word,next_word)
        if (sim >= 0.85):
            correct +=1
        else : wrong +=1
    print('correct: '+str(correct)+(' wrong: ')+str(wrong))
    accur = float(correct/(correct+wrong))
    print('accuracy = ',float(accur))
    


# In[ ]:

# n = no. of predictions
print(accuracy(9000))


# In[ ]:

batch_len = []
for i in range(total_batches):
    batch_len.append(len(subsamples_in[i::total_batches]))


# In[ ]:

max(np.array(batch_len))


# In[ ]:



