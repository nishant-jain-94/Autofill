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


# ## Setting Parameters
word_embedding_dimension = 300
word_embedding_window_size = 4
batch_size = 128 # 32, 64, 128
epochs = 25 # 10, 15, 30
window_size = 5 # 3, 4, 5
accuracy_threshold = 1
activation = 'relu' # sigmoid, relu, softmax
custom_accuracy = 0
loss_function = 'mse' # mse

model_name = "lstm-2-1024-512-epochs-25-batch_size-128"


# ## Instantiate Embeddings 
embeddings = Embeddings(word_embedding_dimension, word_embedding_window_size, 1, 4)


# ### getting data from preprocessing
word2vec_weights = embeddings.get_weights()
word2index, index2word = embeddings.get_vocabulary()
word2vec_model = embeddings.get_model()
tokenized_indexed_sentences = embeddings.get_tokenized_indexed_sentences()


# ### generating training data
vocab_size = len(word2index)
print(vocab_size)
#sorted(window_size,reverse=True)
#sentence_max_length = max([len(sentence) for sentence in tokenized_indexed_sentence ])

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
model = Sequential()
model.add(Embedding(input_dim = word2vec_weights.shape[0], output_dim = word2vec_weights.shape[1], weights = [word2vec_weights]))
model.add(LSTM(1024, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(512))
#model.add(Dropout(0.2))
model.add(Dense(word2vec_weights.shape[1], activation = activation))
model.compile(loss = loss_function, optimizer = 'adam',metrics = ['accuracy'])
model.summary()

model_weights_path = "../weights/"+model_name
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath = checkpoint_path, monitor = 'val_acc', verbose = 1, save_best_only = False, mode = 'max')


# ## Train Model
model_fit_summary = model.fit(seq_in, seq_out, epochs = epochs, verbose = 1, validation_split = 0.2, batch_size = batch_size, callbacks = [checkpoint])


# ### model predict
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
def accuracy():
    count = 0
    correct = 0
    for sub_sample_in, sub_sample_out in zip(seq_in, seq_out):
        ypred = model.predict_on_batch(np.expand_dims(sub_sample_in, axis = 0))[0]
        ytrue = sub_sample_out
        pred_word = word2vec_model.similar_by_vector(ypred)[0][0]
        true_word = word2vec_model.similar_by_vector(ytrue)[0][0]
        similarity = word2vec_model.similarity(pred_word, true_word)
        if similarity >= accuracy_threshold:
            correct += 1
        count += 1
    print("Accuracy {0}".format(correct/count))

custom_accuracy = accuracy()

x = model.layers[1]


x.get_config()


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


model_results