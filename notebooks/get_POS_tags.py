
# coding: utf-8

# In[9]:

import spacy
import re
from nltk.tokenize import word_tokenize, sent_tokenize


# In[10]:

nlp = spacy.load('en')


# In[14]:

def find_POS(raw_text_list):
    final_pos_sents = []
    for sent in raw_text_list:
        doc = nlp(sent)
        pos = []
        for word in doc:
            pos.append(word.pos_)
        final_pos_sents.append(pos)
    return final_pos_sents 

