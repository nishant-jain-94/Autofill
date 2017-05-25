import re
import os
import numpy as np
import json
import pickle
import datetime
import spacy
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize,sent_tokenize
from load_squad_wiki_data import get_squad_data, get_squad_wiki_data
from gensim.models import Word2Vec
nlp = spacy.load('en', parser = False, entity = False, matcher = False, add_vectors = False)

class Embeddings:
    def __init__(self, size, window, min_count, workers):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        base_file_name = '_'.join([str(number) for number in [size, window, min_count, workers]])
        self.path_word2vec_model = '../data/word2vec_model_{0}.pickle'.format(base_file_name)
        self.path_indexed_sentences = '../data/indexed_sentences_{0}.json'.format(base_file_name)
        self.path_word_embeddings = '../data/word_embeddings_{0}.npz'.format(base_file_name)
        self.path_indexed_vocabulary = '../data/indexed_vocabulary_{0}.json'.format(base_file_name)
        self.path_pos_categorical_indexed_sentences = '../data/pos_categorical_indexed_sentences_{0}.json'.format(base_file_name)
        self.path_pos_indexed_vocabulary = '../data/pos_indexed_vocabulary_{0}.json'.format(base_file_name)
        self.load_embeddings()
    
    def tokenize_sentence(self, raw_text):
        sentences = sent_tokenize(raw_text)
        sentences = [re.sub(r'[^\w\'\+\-\=\*\s\^]', '', sent) for sent in sentences]
        tokenized_sentences = [word_tokenize(sent) for sent in sentences]
        return tokenized_sentences
    
    def tokenize_index_sentence(self, sentence):
        word2index, index2word = self.get_vocabulary()
        tokenized_sentences = self.tokenize_sentence(sentence.lower())
        indexed_sentences = [[word2index[word] for word in sent] for sent in tokenized_sentences]
        return indexed_sentences
        
        
    def tag_sentence(self, text):
        tokenized_sentences = self.tokenize_sentence(text.lower())
        tokenized_pos_sentences = self.find_POS(tokenized_sentences)
        return tokenized_pos_sentences
    
    def find_POS(self, tokenized_sentences):
        final_pos_sents = []
        for sent in tokenized_sentences:
            doc = nlp(' '.join(sent))
            pos = []
            for word in doc:
                pos.append(word.pos_)
            final_pos_sents.append(pos)
        return final_pos_sents 

        
    def preprocessor(self, raw_text, size, window, min_count, workers):  
        tokenized_sentences = self.tokenize_sentence(raw_text)
        tokenized_pos_sentences = self.find_POS(tokenized_sentences)
        vocab = ['PUNCT', 'SYM', 'X', 'ADJ', 'VERB', 'CONJ', 'NUM', 'DET', 'ADV', 'PROPN', 'NOUN', 'PART', 'INTJ', 'CCONJ', 'SPACE', 'ADP', 'SCONJ', 'AUX', 'PRON']
        vocab = dict((word, index) for index, word in enumerate(vocab))
        with open(self.path_pos_indexed_vocabulary, 'w') as outfile:
            json.dump(vocab, outfile)
        model = Word2Vec(sentences = tokenized_sentences, size = size, window = window, min_count = min_count, workers = workers)
        vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
        with open(self.path_indexed_vocabulary, 'w') as outfile:
            json.dump(vocab, outfile)
        weights = model.wv.syn0
        np.save(open(self.path_word_embeddings, 'wb'), weights)
        with open(self.path_word2vec_model, 'wb') as output:
            pickle.dump(model, output)

    def get_raw_text(self, dataset):
        raw_text = ""
        question_text = ""
        question_text_list = []
        for data in dataset:
            for ques in data['Question']:
                ques = "SQUADSTART " + ques + " SQUADEND"
                question_text_list.append(ques)                
        question_dict = {}
        question_dict['Questions'] = question_text_list
        with open("../data/squad_train_dev_question.json", "w") as write:
            json.dump(question_dict, write)
        question_text = " ".join(question_text_list)
        raw_text = question_text
        raw_text = raw_text.lower()
        return raw_text
    
    def load_embeddings(self):
        if not (os.path.isfile(self.path_word2vec_model) and 
                os.path.isfile(self.path_word_embeddings) and 
                os.path.isfile(self.path_indexed_vocabulary) and
                os.path.isfile(self.path_pos_indexed_vocabulary)):
            print("Loading embeddings....")
            dataset = get_squad_wiki_data()
            raw_text = self.get_raw_text(dataset) 
            self.preprocessor(raw_text, self.size, self.window, self.min_count, self.workers)
        print("Loading the embeddings from the cache")
        if not (os.path.isfile(self.path_pos_categorical_indexed_sentences) and 
            os.path.isfile(self.path_indexed_sentences)):
            print("Starting tokenized, pos squad data.....")
            squad_data = get_squad_data()
            raw_text = self.get_raw_text(squad_data) 
            self.create_tokenized_squad_corpus(raw_text)
            self.create_pos_tokenized_squad_corpus(raw_text)
         
    def get_weights(self):
        weights = np.load(open(self.path_word_embeddings,'rb'))
        return weights

    def get_vocabulary(self):
        with open(self.path_indexed_vocabulary, 'r') as f:
            data = json.load(f)
        word2idx = data
        idx2word = dict([(v, k) for k, v in data.items()])
        return word2idx, idx2word

    def get_model(self):
        with open(self.path_word2vec_model, 'rb') as output:
            model = pickle.load(output)
        return model

    def get_tokenized_indexed_sentences(self):
        with open(self.path_indexed_sentences, 'r') as f:
            tokenized_sentences = json.load(f)
        return tokenized_sentences
    
    def get_pos_vocabulary(self):
        with open(self.path_pos_indexed_vocabulary, 'r') as f:
            data = json.load(f)
        pos2idx = data
        idx2pos = dict([(v, k) for k, v in data.items()])
        return pos2idx, idx2pos
    
    def get_pos_categorical_indexed_sentences(self):
        with open(self.path_pos_categorical_indexed_sentences, 'r') as f:
            tokenized_pos_sentences = json.load(f)
        return tokenized_pos_sentences
    
    def create_tokenized_squad_corpus(self, squad_corpus):
        print("Creating Tokenized Squad Corpus")
        tokenized_indexed_sentences = self.tokenize_index_sentence(squad_corpus)
        with open(self.path_indexed_sentences, "w") as f:
            json.dump(tokenized_indexed_sentences, f)
        
    def create_pos_tokenized_squad_corpus(self, squad_corpus):
        print("Creating Tokenized Squad Corpus")
        tokenized_pos_sentences = self.tag_sentence(squad_corpus)
        pos2idx, idx2pos = self.get_pos_vocabulary()
        categorical_pos_sentences = [to_categorical([pos2idx[word] for word in sent], num_classes = len(pos2idx)).tolist() for sent in tokenized_pos_sentences] 
        with open(self.path_pos_categorical_indexed_sentences, "w") as f:
            json.dump(categorical_pos_sentences, f)
            
    def load_google_word2vec_model(self):
        print("LOADING TRAINED SQUAD AND WIKI WORD2VEC MODEL.....")
        wiki_word2vec_model = self.get_model()
        print("INTERSECTING GOOGLES WORD2VEC MODEL WITH WORD2VEC MODEL")
        wiki_word2vec_model.intersect_word2vec_format(fname = '../model/GoogleNews-vectors-negative300.bin' , lockf = 1.0, binary = True)        

