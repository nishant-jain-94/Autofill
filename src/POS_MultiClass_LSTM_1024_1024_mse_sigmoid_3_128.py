
# coding: utf-8

# In[1]:

import numpy as np
import os
import sys
import h5py
import datetime
import json
import pandas as pd
import itertools 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from embeddings import Embeddings
from keras.utils import to_categorical


# ## Setting Parameters

# In[2]:

word_embedding_dimension = 300
word_embedding_window_size = 4
batch_size = 128 # 32, 64, 128
epochs = 25 # 10, 15, 30
window_size = 4 # 3, 4, 5
accuracy_threshold = 0.85
activation = 'sigmoid' # sigmoid, relu, softmax
custom_accuracy = 0
loss_function = 'mse' # mse


# In[3]:

model_name = 'POS_MultiClass_LSTM' + '_1024_1024_' + loss_function + "_" + activation + "_" + str(window_size) + "_" + str(batch_size) #MODEL_NAME #POS-LSTM


# In[4]:

with open('../data/word_tokenized_sentence_300_4_1_4.json', 'r') as myfile:
    raw_data = json.load(myfile)


# In[6]:

embeddings = Embeddings(word_embedding_dimension, word_embedding_window_size, 1, 4)
pos2index, index2pos = embeddings.get_pos_vocabulary()


# In[8]:

test_data = embeddings.find_POS(raw_data) #find_POS(raw_data)


# In[ ]:

whole_test_data = [word for sent in test_data for word in sent]


# In[ ]:

new_data = []
for i in range(len(whole_test_data)-window_size-1):
    x = whole_test_data[i:i + window_size + 1]
    new_data.append(x)


# In[ ]:

new_data = [[data[:3], data[3]] for data in new_data]


# In[ ]:

vocab = ['PUNCT','SYM','X','ADJ','VERB','CONJ','NUM','DET','ADV','PROPN','NOUN','PART','INTJ','CCONJ','SPACE','ADP','SCONJ','AUX', 'PRON']


# In[ ]:

new_data.sort()
new_seq_in = []
new_seq_out = []
for i,j in itertools.groupby(new_data, lambda x: x[0]):
    ex = set(list(zip(*list(j)))[1])
    
    inputs = [to_categorical(pos2index[x_pos], num_classes = len(vocab)) for x_pos in i]
    new_seq_in_each = [each[0] for each in inputs] 
    new_seq_in.append(new_seq_in_each)
    
    outputs = [(to_categorical(pos2index[y_pos], num_classes = len(vocab))).tolist()[0] for y_pos in ex]
    new_seq_out_each = [each for each in outputs]
    new_seq_out_each = np.sum(new_seq_out_each, axis=0)
    new_seq_out.append(new_seq_out_each)
    

new_seq_in = np.array(new_seq_in)
new_seq_out = np.array(new_seq_out)


# In[ ]:

# Changes to the model to be done here
model = Sequential()
model.add(LSTM(1024, input_shape=(new_seq_in.shape[1], new_seq_in.shape[2]), return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(1024))
#model.add(Dropout(0.2))
model.add(Dense(len(vocab), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:

model_weights_path = "../weights/"+ model_name
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/pos_weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')


# In[ ]:

model_fit_summary = model.fit(new_seq_in, new_seq_out, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.25, callbacks=[checkpoint])


# In[ ]:

check_ori = 0
check_pre = 0
counter = 0
test_start = 0
test_end = 100
list_for_hist_words = []
list_for_hist_index = []
list_for_hist_words_ori = []
list_for_hist_index_ori = []
for i in range(test_start, test_end):
    test_no = i
    to_predict = new_seq_in[test_no:test_no+1]
    y_ans = model.predict(to_predict)
    
    for word, corr_int in pos2index.items():
        if corr_int == np.argmax(y_ans):
            #print ("pridicted: ",word, corr_int)
            check_pre = corr_int
            list_for_hist_words.append(word)
            list_for_hist_index.append(corr_int)
        if corr_int == np.argmax(new_seq_out[test_no:test_no+1]):
           #print ("original: ",word, corr_int)
            check_ori = corr_int
            list_for_hist_words_ori.append(word)
            list_for_hist_index_ori.append(corr_int)
    if check_ori == check_pre :
        counter += 1
    #print('\n')

print("Correct predictions: ",counter, '\nTotal Predictions: ',test_end - test_start)
custom_accuracy = counter/(test_end-test_start)


# In[ ]:

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

