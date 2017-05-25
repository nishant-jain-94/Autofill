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

word2index

tokenized_indexed_sentences[0]

tokenized_indexed_sentences = [np.array(sentence) + 1 for sentence in tokenized_indexed_sentences if len(sentence) > 0]

new_weights = np.zeros((1, word2vec_weights.shape[1]))

new_weights = np.append(new_weights, word2vec_weights, axis = 0)

# ### generating training data
window_size = 5
vocab_size = len(word2index)
print(vocab_size)

maxlen = max([len(sentence) for sentence in tokenized_indexed_sentences])
tokenized_indexed_sentences = sequence.pad_sequences(tokenized_indexed_sentences)

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
# Changes to the model to be done here
model = Sequential()
model.add(Embedding(input_dim = new_weights.shape[0], output_dim = new_weights.shape[1], weights = [new_weights], mask_zero = True))
model.add(LSTM(1024, return_sequences = True))
model.add(LSTM(1024, return_sequences = True))
model.add(LSTM(300, return_sequences = True))
model.compile(loss='cosine_proximity', optimizer='adam',metrics=['accuracy'])
model.summary()

model_weights_path = "../weights/lstm-3-1024-1024-batchsize-512-epochs-30-Sequence"
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/weights.{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=False, mode='max')


# ## Train Model
model.fit(seq_in, seq_out, epochs=30, verbose=1, batch_size=512, callbacks=[checkpoint])


# ### model predict
start = 0
sentence_test = "In which regions in particular did"
indexed_sentences = embeddings.get_indexed_query(sentence_test)
print("indexed_sentences ",indexed_sentences)
sent = np.array(indexed_sentences)
pattern = list(sent)
print("\"",' '.join(index2word[index] for index in pattern))
for i in range(10):
    prediction = model.predict(np.array([pattern]))
    pred_word = word2vec_model.similar_by_vector(prediction[0][prediction.shape[1] - 1])[0][0]
    sys.stdout.write(pred_word+" ")
    pattern.append(word2index[pred_word])
    pattern = pattern[:len(pattern)]

# ## Accuracy
def accuracy():
    count = 0
    correct = 0
    for sub_sample_in, sub_sample_out in zip(seq_in, seq_out):
        ypred = model.predict_on_batch(np.expand_dims(sub_sample_in, axis = 0))[0]
        ytrue = sub_sample_out
        pred_word = word2vec_model.similar_by_vector(ypred)[0][0]
        true_word = word2vec_model.similar_by_vector(ytrue)[0][0]
        similarity = word2vec_model.similarity(pred_word, true_word)
        if similarity == 1:
            correct += 1
        count += 1
    print("Accuracy {0}".format(correct/count))