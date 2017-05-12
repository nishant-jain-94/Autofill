import spacy
import re

nlp = spacy.load('en')

def find_POS(raw_text_list):
    final_pos_sents = []
    for sent in raw_text_list:
        doc = nlp(sent)
        pos = []
        for word in doc:
            pos.append((word, word.pos_))
        final_pos_sents.append(pos)
    return final_pos_sents 

