import re
import os
import numpy as np
import json
import pickle
import datetime
import spacy
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize, sent_tokenize
from load_squad_wiki_data import get_squad_data, get_squad_wiki_data
from gensim.models import Word2Vec
from spacy.en import English
nlp = spacy.load('en', parser = False, matcher = False, add_vectors = False)
nlp_en = English()

class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)
    
class Embeddings:
    def __init__(self, size, window, min_count, workers):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        base_file_name = '_'.join([str(number) for number in [size, window, min_count, workers]])
        self.path_word2vec_model = '../data/word2vec_model_{0}.pickle'.format(base_file_name)
        self.path_word_tokenized_sentence = '../data/word_tokenized_sentence_{0}.json'.format(base_file_name)
        self.path_indexed_sentences = '../data/indexed_sentences_{0}.json'.format(base_file_name)
        self.path_vocabulary = '../data/vocabulary_{0}.json'.format(base_file_name)
        self.path_google_intersected = '../data/google_intersected_model_{0}.pickle'.format(base_file_name)
        self.index_sentences()
    
    def tokenize_sentences(self):
        tokenized_sentences = []
        embeddings_generator = self.load_embeddings()
        for sentence in embeddings_generator:
            tokenized_sentence = word_tokenize(sentence.lower())
            tokenized_sentences.append(tokenized_sentence)
        with open(self.path_word_tokenized_sentence, "w") as outfile:
            json.dump(tokenized_sentences, outfile)
        
    def create_embeddings(self):
        sentences = self.get_tokenized_sentences()
        word2vec_model = Word2Vec(sentences, size = self.size, window = self.window, min_count = self.min_count, workers = self.workers)
        word2index = dict([(k, v.index) for k, v in word2vec_model.wv.vocab.items()])
        with open(self.path_vocabulary, "w") as output:
            json.dump(word2index, output)
        with open(self.path_word2vec_model, 'wb') as output:
            pickle.dump(word2vec_model, output)
    
    def create_google_intersected_embeddings(self):
        word2vec_model = self.get_model()
        intersected_model = self.load_google_word2vec_model(word2vec_model) 
        with open(self.path_google_intersected, "wb") as output:
            pickle.dump(intersected_model, output)
                    
    def index_sentences(self):
        if not os.path.isfile(self.path_indexed_sentences):
            tokenized_sentences = self.get_tokenized_sentences()
            word2vec_model = self.get_intersected_model()
            word2index, index2word = self.get_vocabulary() 
            indexed_sentences = [[word2index[word] for word in sent] for sent in tokenized_sentences]
            with open(self.path_indexed_sentences, "w") as outfile:
                json.dump(indexed_sentences, outfile)

    def get_raw_text(self, dataset):
        question_text = ""
        for data in dataset:
            for question in data['Question']:
                question = self.noun_chunkers(question)
                question = "SQUADSTART " + re.sub(r'[^\w\'\+\-\=\*\s\^]', '', question) + " SQUADEND"
                yield question
    
    def load_embeddings(self):
        print("Loading embeddings....")
        dataset = get_squad_wiki_data()
        return self.get_raw_text(dataset)

    def get_vocabulary(self):
        with open(self.path_vocabulary, 'r') as f:
            data = json.load(f)
        word2idx = data
        idx2word = dict([(v, k) for k, v in data.items()])
        return word2idx, idx2word

    def get_model(self):
        if not os.path.isfile(self.path_word2vec_model):
            print("Creating Embeddings...")
            self.create_embeddings()
        print("Loading Embeddings...")
        with open(self.path_word2vec_model, 'rb') as output:
            model = pickle.load(output)
        return model
    
    def get_tokenized_sentences(self):
        if not os.path.isfile(self.path_word_tokenized_sentence):
            print("Creating Tokenized Sentences...")
            self.tokenize_sentences()
        print("Loading Indexed Sentences...")
        with open(self.path_word_tokenized_sentence, "r") as file:
            tokenized_sentences = json.load(file)
        return tokenized_sentences

    def get_indexed_sentences(self):
        if not os.path.isfile(self.path_indexed_sentences):
            print("Creating Indexed Sentences...")
            self.index_sentences()
        print("Loading Indexed Sentences...")
        with open(self.path_indexed_sentences, 'r') as f:
            indexed_sentences = json.load(f)
        return indexed_sentences
        
    def load_google_word2vec_model(self, model):
        print("INTERSECTING GOOGLES WORD2VEC MODEL WITH ORIGINAL WORD2VEC MODEL")
        model.intersect_word2vec_format(fname = '../model/GoogleNews-vectors-negative300.bin' , lockf = 1.0, binary = True)        
        return model
    
    def get_intersected_model(self):
        if not os.path.isfile(self.path_google_intersected):
            self.create_google_intersected_embeddings()
        with open(self.path_google_intersected, "rb") as output:
            intersected_model = pickle.load(output)
        return intersected_model
            
    def noun_chunkers(self, raw_text):
        doc = nlp_en(raw_text)
        for entity in doc.ents:
            raw_text = raw_text.replace(str(entity), "_".join(str(entity).split()))
        return raw_text
    
    def get_indexed_query(self, query):
        query = self.noun_chunkers(query)
        query = "SQUADSTART " + re.sub(r'[^\w\'\+\-\=\*\s\^]', '', query)
        word_tokenized_query = word_tokenize(query.lower())
        word2index, index2word = self.get_vocabulary()
        indexed_query = [word2index[word] for word in word_tokenized_query if word in word2index.keys()]
        return indexed_query
