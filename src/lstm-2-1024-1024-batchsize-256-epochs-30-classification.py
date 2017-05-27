import os
import json
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM

from intersect_embeddings import Embeddings


MODEL_NAME = "lstm-2-1024-1024-batchsize-256-epochs-30-classification"
WORD_EMBEDDING_DIMENSION = 300
WORD_EMBEDDING_WINDOW_SIZE = 4
BATCH_SIZE = 256
EPOCHS = 30
WINDOW_SIZE = "None"
ACCURACY_THRESHOLD = 1
ACTIVATION = 'softmax'
CUSTOM_ACCURACY = 0
LOSS_FUNCTION = 'cosine_proximity'

class LSTMModel():
    def __init__(self):

        # Instantiate Embeddings
        self.embeddings = Embeddings(WORD_EMBEDDING_DIMENSION, WORD_EMBEDDING_WINDOW_SIZE, 1, 4)

        # Gets word2vec_model, word2index, index2word, word2vec_weights, tokenized_indexed_sentences
        self.word2vec_model = self.embeddings.get_intersected_model()
        word2index = self.embeddings.get_vocabulary()[0]
        word2vec_weights = self.word2vec_model.wv.syn0
        indexed_sentences = self.embeddings.get_indexed_sentences()

        # Shifting the indexes by 1 so as to reserve space for Masking
        self.word2index = {word:index + 1 for word, index in word2index.items()}
        self.index2word = {index:word for word, index in self.word2index.items()}
        self.vocab_size = len(word2index)
        indexed_sentences = [np.array(sentence) + 1
                             for sentence in indexed_sentences
                             if len(sentence) > 0]

        # Creating a zero vector for masking and then appending with word2vec_weights
        mask_vector = np.zeros((1, word2vec_weights.shape[1]))
        self.word2vec_weights = np.append(mask_vector, self.word2vec_weights, axis=0)

        # Padding Sentences
        sentence_with_max_len = max([len(sentence) for sentence in indexed_sentences])
        self.indexed_sentences = sequence.pad_sequences(indexed_sentences,
                                                        maxlen=sentence_with_max_len,
                                                        padding='post')

    def generate_sequences(self):

        for seq_in in self.indexed_sentences:
            seq_in_len = len(seq_in)
            seq_out = np.append(seq_in[1:], seq_in[seq_in_len - 1])
            one_hot_encoded_y = [np_utils.to_categorical(index, num_classes=self.vocab_size)
                                 for index in seq_out]
            yield (seq_in, one_hot_encoded_y)

    def train_model(self):
        
        # Defining model
        model = Sequential()
        model.add(Embedding(input_dim=self.word2vec_weights.shape[0],
                            output_dim=self.word2vec_weights.shape[1],
                            weights=[self.word2vec_weights], mask_zero=True))
        model.add(LSTM(1024, return_sequences=True))
        model.add(LSTM(1024))
        model.add(Dense(self.vocab_size, activation='sigmoid'))
        model.compile(loss='cross_entropy', optimizer='adam', metrics=['accuracy'])
        model_weights_path = "../weights/lstm-2-1024-1024-batchsize-256-epochs-30-classification"
        if not os.path.exists(model_weights_path):
            os.makedirs(model_weights_path)
        checkpoint_path = model_weights_path + '/weights.{epoch:02d}.hdf5'
        checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                     verbose=1,
                                     save_best_only=False,
                                     mode='max')
        print("Model Summary")
        print(model.summary())
        model.fit(seq_in, seq_out, epochs=EPOCHS, verbose=1, batch_size=BATCH_SIZE, callbacks=[checkpoint])
        return model
    
    def predict(self):
        model = self.train_model()
        sentence_test = "In which regions in particular did"
        indexed_sentences = self.embeddings.get_indexed_query(sentence_test)
        sent = np.array(indexed_sentences) + 1
        pattern = list(sent)
        print(' '.join(self.index2word[index] for index in pattern))
        # for i in range(10):
        prediction = model.predict(np.array([pattern]))
        pred_word = self.word2vec_model.similar_by_vector(prediction[0][prediction.shape[1] - 1])[0][0]
        sys.stdout.write(pred_word + " ")
        pattern.append(self.word2index[pred_word])
        pattern = pattern[:len(pattern)]

    def accuracy(self):
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

    def model_summary(self):
        # model_results = model_fit_summary.history
        # model_results.update(model_fit_summary.params)
        model_results["word_embedding_dimension"] = WORD_EMBEDDING_DIMENSION
        model_results["word_embedding_window_size"] = WORD_EMBEDDING_WINDOW_SIZE
        model_results["window_size"] = WINDOW_SIZE
        model_results["batch_size"] = BATCH_SIZE
        model_results["epochs"] = EPOCHS
        model_results["model_name"] = MODEL_NAME
        model_results["accuracy_threshold"] = ACCURACY_THRESHOLD
        model_results["activation"] = ACTIVATION
        model_results["custom_accuracy"] = CUSTOM_ACCURACY
        model_results["loss_function"] = LOSS_FUNCTION
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