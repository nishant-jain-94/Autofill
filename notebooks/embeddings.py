
# coding: utf-8

# In[ ]:

import re
import os
import numpy as np
import json
import pickle
import datetime
from nltk.tokenize import word_tokenize,sent_tokenize
from gensim.models import Word2Vec
from load_squad_wiki_data import get_squad_wiki_data
from get_POS_tags import find_POS


# In[ ]:

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
        self.path_pos_indexed_sentences = '../data/pos_indexed_sentences_{0}.json'.format(base_file_name)
        self.path_pos_indexed_vocabulary = '../data/pos_indexed_vocabulary_{0}.json'.format(base_file_name)
        self.load_embeddings()
        
    def preprocessor(self, raw_text, size, window, min_count, workers):
        # removes all the punctuations and retains only alphabets,digits,fullstop,apostrophe
        clean_text = re.sub(r'[^\w\'\.\+\-\=\*\s\^]','' ,raw_text)
        # sentence tokenize clean text
        sentences = sent_tokenize(clean_text)
        # replace full stops from each sentence in sentences list by null
        sentences = [sent.replace('.','') for sent in sentences]
        # finding tokenized pos sentences
        tokenized_pos_sentences = find_POS(sentences)
        # finding out the vocabulary of pos tagged sentences
        vocab = []
        for sent in tokenized_pos_sentences:
                vocab.extend(sent) 
        vocab = sorted(list(set(vocab)))
        vocab = dict((word, index) for index, word in enumerate(vocab))
        # Storeing the vocab2index of pos tags in a seperate file
        with open(self.path_pos_indexed_vocabulary,'w') as outfile:
            json.dump(vocab,outfile)
        # replacing each word of tokenized_pos_sentences by coressponding vocabulary index
        indexed_pos_sentences = [[vocab[word] for word in sent] for sent in tokenized_pos_sentences]
        # storeing indexed_pos_sentences(tokenized pos sentences) 
        with open(self.path_pos_indexed_sentences,'w') as outfile:
            json.dump(indexed_pos_sentences,outfile)
        # word tokenize each sentence in sentences list 
        tokenized_sentences = [word_tokenize(sent) for sent in sentences]
        # initialize word2vector model
        model = Word2Vec(sentences = tokenized_sentences, size = size, window = window, min_count = min_count, workers = workers)
        # finding out the vocabulary of raw_text with index     
        vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
        # Storeing the vocab2index in a seperate file
        with open(self.path_indexed_vocabulary,'w') as outfile:
            json.dump(vocab,outfile)
        # replacing each word of tokenized_sentences by coressponding vocabulary index
        indexed_sentences = [[vocab[word] for word in sent] for sent in tokenized_sentences]
        # storeing indexed_sentences(tokenized sentences) in indexed.json file
        with open(self.path_indexed_sentences,'w') as outfile:
            json.dump(indexed_sentences,outfile)
        # finding gensim weights
        weights = model.wv.syn0
        # storeing weights in wordembeddings.npz file
        np.save(open(self.path_word_embeddings, 'wb'), weights)
        # dump the word2vec model in dump file word2vec_model
        with open(self.path_word2vec_model, 'wb') as output:
            pickle.dump(model, output)

    def load_embeddings(self):
        if not (os.path.isfile(self.path_word2vec_model) or 
                os.path.isfile(self.path_word_embeddings) or 
                os.path.isfile(self.path_indexed_sentences) or
                os.path.isfile(self.path_indexed_vocabulary)):
            dataset = get_squad_wiki_data()
            raw_text = ""
            passage_text = "" 
            question_text = ""
            passage_text_list = []
            question_text_list = []
            for data in dataset:
                passage_text_list.append(data['Paragraph'])
                question_text_list.extend(data['Question'])                
            passage_text = "".join(passage_text_list)
            question_text = ".".join(question_text_list)
            raw_text = passage_text + " " + question_text
            raw_text = raw_text.lower()
            self.preprocessor(raw_text, self.size, self.window, self.min_count, self.workers)        

    # Will load and return weights from the existing embedding.npz file
    def get_weights(self):
        weights = np.load(open(self.path_word_embeddings,'rb'))
        return weights

    # Returns word2Index and index2word
    def get_vocabulary(self):
        with open(self.path_indexed_vocabulary, 'r') as f:
            data = json.load(f)
        word2idx = data
        idx2word = dict([(v, k) for k, v in data.items()])
        return word2idx, idx2word

    # Returns the pickled model
    def get_model(self):
        with open(self.path_word2vec_model,'rb') as output:
            model = pickle.load(output)
        return model

    # Returns the tokenized sentences
    def get_tokenized_indexed_sentences(self):
        with open(self.path_indexed_sentences, 'r') as f:
            tokenized_sentences = json.load(f)
        return tokenized_sentences
    
    # Returns pos2Index and index2pos
    def get_pos_vocabulary(self):
        with open(self.path_pos_indexed_vocabulary, 'r') as f:
            data = json.load(f)
        pos2idx = data
        idx2pos = dict([(v, k) for k, v in data.items()])
        return pos2idx, idx2pos
    
    # Returns the tokenized pos sentences
    def get_pos_tokenized_indexed_sentences(self):
        with open(self.path_pos_indexed_sentences, 'r') as f:
            tokenized_pos_sentences = json.load(f)
        return tokenized_pos_sentences


# In[ ]:

start_date = datetime.datetime.now()
e = Embeddings(5,4,1,4)
end_date = datetime.datetime.now()
print("TOTAL TIME ELAPSED IN EMBEDDINGS.PYNB")
print(((end_date - start_date).hour)," HOURS ",((end_date - start_date).minute)," MINUTES ",((end_date - start_date).second)," SECONDS ")


# In[ ]:

#e.get_model()


# In[ ]:

#e.get_weights()


# In[ ]:

#e.get_vocabulary()


# In[ ]:

#e.get_tokenized_indexed_sentences()


# In[ ]:

#vocab = ['PUNCT','SYM','X','ADJ','VERB','CONJ','NUM','DET','ADV','PROPN','NOUN','PART','INTJ','CCONJ','','']

