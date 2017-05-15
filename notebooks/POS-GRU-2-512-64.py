
# coding: utf-8

# In[ ]:

import numpy as np
import os
import sys
import h5py
import datetime
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.callbacks import ModelCheckpoint
from embeddings import Embeddings


# In[ ]:

embeddings = Embeddings(100, 4, 1, 4)
tokenized_pos_sentences = embeddings.get_pos_categorical_indexed_sentences()
pos2index, index2pos = embeddings.get_pos_vocabulary()
no_of_unique_tags = len(pos2index)
window_size = 2


# In[ ]:

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


# In[ ]:

x_data = seq_in
y_data = seq_out


# In[ ]:

# Changes to the model to be done here
model = Sequential()
model.add(GRU(512, input_shape=(x_data.shape[1], x_data.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(512))
model.add(Dropout(0.2))
model.add(Dense(no_of_unique_tags, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:

model_weights_path = "../weights/POS_GRU_2_Layer"
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
checkpoint_path = model_weights_path + '/pos_weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')


# In[ ]:

model.fit(x_data, y_data, epochs=10, batch_size=64, verbose=1, validation_split=0.2, callbacks=[checkpoint])


# In[ ]:

#test_no = 31
#to_predict = x_data[test_no:test_no+1]


# In[ ]:

#y_ans = model.predict(to_predict)


# In[ ]:

#y_ans


# In[ ]:

#for word, corr_int in pos2index.items():
#    if corr_int == np.argmax(y_ans[0]):
#        print ("pridicted: ",word, corr_int)
#    if corr_int == np.argmax(y_data[test_no:test_no+1][0]):
#        print ("original: ",word, corr_int)


# In[ ]:




# In[ ]:



