import numpy as np
import os
import sys
import h5py
import datetime
import json
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from embeddings import Embeddings

# ## Setting Parameters
word_embedding_dimension = 100
word_embedding_window_size = 4
batch_size = 128 
epochs = 15 
window_size = 3 
accuracy_threshold = 0.85
activation = 'relu' 
custom_accuracy = 0
loss_function = 'mse'

model_name = 'POS_LSTM_' + '_1024_1024_' + loss_function + "_" + activation + "_" + str(window_size) + "_" + str(batch_size)

#MODEL_NAME #POS-LSTM
embeddings = Embeddings(word_embedding_dimension, word_embedding_window_size, 1, 4)
tokenized_pos_sentences = embeddings.get_pos_categorical_indexed_sentences()
pos2index, index2pos = embeddings.get_pos_vocabulary()
no_of_unique_tags = len(pos2index)

seq_in = []
seq_out = []
# generating dataset
for sentence in tokenized_pos_sentences:
    
    for i in range(len(sentence)-window_size-1):
        
        x = sentence[i:i + window_size]
        y = sentence[i + window_size]
        seq_in.append(x)
        seq_out.append(y)
        
# converting seq_in and seq_out into numpy array
seq_in = np.array(seq_in)
seq_out = np.array(seq_out)
n_samples = len(seq_in)
print ("Number of samples : ", n_samples)

x_data = seq_in
y_data = seq_out

# Changes to the model to be done here
model = Sequential()
model.add(LSTM(1024, input_shape = (x_data.shape[1], x_data.shape[2]), return_sequences = True))
model.add(LSTM(1024))
model.add(Dense(no_of_unique_tags, activation = 'relu'))
model.compile(loss='categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
model.summary()

model_weights_path = "../weights/"+ model_name
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/pos_weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath = checkpoint_path, monitor = 'val_acc', verbose = 1, save_best_only = False, mode = 'max')

model_fit_summary = model.fit(x_data, y_data, epochs = epochs, batch_size = batch_size, verbose = 1, validation_split = 0.25, callbacks = [checkpoint])

check_ori = 0
check_pre = 0
counter = 0
test_start = 0
test_end = 10000
list_for_hist_words = []
list_for_hist_index = []
list_for_hist_words_ori = []
list_for_hist_index_ori = []

for i in range(test_start, test_end):
    
    test_no = i
    to_predict = x_data[test_no:test_no+1]
    y_ans = model.predict(to_predict)
    
    for word, corr_int in pos2index.items():
        
        if corr_int == np.argmax(y_ans):
            
            check_pre = corr_int
            list_for_hist_words.append(word)
            list_for_hist_index.append(corr_int)
            
        if corr_int == np.argmax(y_data[test_no:test_no+1]):
            
            check_ori = corr_int
            list_for_hist_words_ori.append(word)
            list_for_hist_index_ori.append(corr_int)
            
    if check_ori == check_pre :
        counter += 1

print("Correct predictions: ",counter, '\nTotal Predictions: ',test_end - test_start)
custom_accuracy = counter/(test_end-test_start)

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