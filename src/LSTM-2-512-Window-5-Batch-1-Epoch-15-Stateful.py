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

embeddings = Embeddings(100, 4, 1, 4)


# getting data from preprocessing
word2vec_weights = embeddings.get_weights()
word2index, index2word = embeddings.get_vocabulary()
word2vec_model = embeddings.get_model()
tokenized_indexed_sentences = embeddings.get_tokenized_indexed_sentences()

window_size = 5
vocab_size = len(word2index)
print(vocab_size)

model = Sequential()
model.add(Embedding(input_dim = word2vec_weights.shape[0], output_dim = word2vec_weights.shape[1], weights = [word2vec_weights], batch_input_shape = (1, 5)))
model.add(LSTM(512, return_sequences = True, stateful = True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.1))
model.add(Dense(word2vec_weights.shape[1], activation = 'sigmoid'))
model.compile(loss = 'mse', optimizer = 'adam',metrics = ['accuracy'])
model.summary()


model_weights_path = "../weights/LSTM-2-512-Window-5-Batch-1-Epoch-10-Stateful"
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)


seq_in = []
seq_out = []

# generating dataset
for sentence in tokenized_indexed_sentences:
    sentence_seq_in = []
    sentence_seq_out = []
    for i in range(len(sentence) - window_size - 1):
        x = sentence[i : i + window_size]
        y = sentence[i + window_size]
        sentence_seq_in.append(x)
        sentence_seq_out.append(word2vec_weights[y])
    seq_in.append(sentence_seq_in)
    seq_out.append(sentence_seq_out)

# converting seq_in and seq_out into numpy array
seq_in = np.array(seq_in)
seq_out = np.array(seq_out)
n_samples = len(seq_in)
print ("Number of samples : ", n_samples)


seq_in.shape
np.expand_dims(seq_in[0][0], axis=1)


print("Train")
for epoch in range(15):
    print("Epoch {0}/{1}".format(epoch + 1, 15))
    mean_tr_accuracy = []
    mean_tr_loss = []
    for i in range(len(seq_in)):
        if i % 100 == 0:
            print("Done with {0}/{1}".format(i, len(seq_in)))
        for j in range(len(seq_in[i])):
            train_accuracy, train_loss = model.train_on_batch(np.expand_dims(seq_in[i][j], axis = 0), np.expand_dims(seq_out[i][j], axis = 0))
            mean_tr_accuracy.append(train_accuracy)
            mean_tr_loss.append(train_loss)
        model.reset_states()
    mean_accuracy = np.mean(mean_tr_accuracy)
    mean_loss = np.mean(mean_tr_loss)
    print("Mean Accuracy", mean_tr_accuracy)
    print("Mean Loss", mean_tr_loss)
    filepath = "../weights/LSTM-2-512-Window-5-Batch-1-Epoch-10-Stateful/weights-epoch-{0}-acc-{1}-loss-{2}".format(epoch + 1, mean_accuracy, mean_loss)
    model.save_weights(filepath)

# model predict

start = 20
samples_in = seq_in[start]
sample_out = seq_out[start]
for index, sample in enumerate(samples_in):
    predictions = model.predict_on_batch(np.expand_dims(sample, axis = 0))
    for pred_index, prediction in enumerate(predictions):
        print(' '.join(index2word[pred_index] for index in samples[index]))
        pred_word = word2vec_model.similar_by_vector(prediction)[0][0]
        sys.stdout.write("*" + pred_word + " \n")


# Accuracy
def accuracy():
    count = 0
    correct = 0
    for sub_samples_in, sub_samples_out in zip(seq_in, seq_out):
        for sub_sample_in, sub_sample_out in zip(sub_samples_in, sub_samples_out):
            ypred = model.predict_on_batch(np.expand_dims(sub_sample_in, axis = 0))[0]
            ytrue = sub_sample_out
            pred_word = word2vec_model.similar_by_vector(ypred)[0][0]
            true_word = word2vec_model.similar_by_vector(ytrue)[0][0]
            similarity = word2vec_model.similarity(pred_word, true_word)
            if similarity >= 0.85:
                correct += 1
            count += 1
    print("Accuracy {0}".format(correct/count))


print(accuracy())
