
# coding: utf-8

# # N-gram-Approach

# ### Importing require packages

# In[1]:

import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.util import ngrams
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist
import json


# ### Loading Data

# In[2]:

with open("../data/squad_wiki_data.json","r") as outfile:
    dataset = json.load(outfile)


# In[3]:

questions = dataset[0]['Question']
questions = ' '.join(questions)


# ### Generatng n_grams-bigrams

# In[4]:

def generate_conditional_prob_dist(training_passage, n):
    """Given a passage generates ngrams and then subsequently decrements n, where n >= 2 """
    
    ## removing special character and symbols and converting to lower case
    training_passage = re.sub(r"[^\w\'\?]", ' ', training_passage).lower()
    
    ## tokenizing the sanitized passage
    words = nltk.word_tokenize(training_passage)
    
    cfdist_list = []
    cpdist_list = []
    
    ## generating cpdist and n_grams for n_plus_one to bigrams
    for i in range(n, 1, -1):
        ## generating n_plus_one_grams and converting into list
        n_grams_generated = list(ngrams(words, i))
        
        ## converting into (n_gram, n+1 words) for prediction
        n_grams_for_predict = [(n_gram[:-1], n_gram[-1]) for n_gram in n_grams_generated] 
        
        ## calculating conditionalfrequency for all n_grams
        cfdist = ConditionalFreqDist(n_grams_for_predict)
        
        ## calculating conditional probablitlity of next word for all n_grams
        cpdist = ConditionalProbDist(cfdist, MLEProbDist)
        
        cfdist_list.append(cfdist)
        cpdist_list.append(cpdist)
    
    return cpdist_list


# In[5]:

cp_list = generate_conditional_prob_dist(questions, 5)


# ### Predict the next word function

# In[6]:

def predict_next_using_n_grams(n_grams, cpdist_list, mode="nsent"):
    
    next_prediction = None
    residue = ""
    
    # n_gram = tuple of n_words #input to the function
    len_n_grams = len(n_grams)
    
    # to end the recursion
    if(len_n_grams==0): return #no prediction available
    
    len_cpdist = len(cpdist_list)
    
    #handling sentence with length more than the n_grams generated
    if len_n_grams > len_cpdist: 
        residue = ' '.join(n_grams[:-len_cpdist])
        n_grams = n_grams[-len_cpdist:]
        len_n_grams = len(n_grams)
    
    # possible predictions
    possible_pred = list(cpdist_list[len_cpdist-len_n_grams][n_grams].samples())
    
    # number of possible prediciton for the provided n_grams
    n_possible_pred = len(possible_pred)
    
    if n_possible_pred > 0:
        
        if(mode == 'nword'):
            if(n_possible_pred == 1):
                next_prediction = '\n'.join(possible_pred[:5])
            
        if(mode == 'nsent'):
            possible_predictions = []
            for pred in possible_pred[:1]:
                print(pred)
                pred_words = list(n_grams)
                next_pred = pred
                while next_pred != '?':
                    pred_words.append(next_pred)
                    candidate_pred = list(cpdist_list[len_cpdist-len_n_grams][tuple(pred_words[-len_n_grams:])].samples())
                    next_pred = candidate_pred[0] if "?" not in candidate_pred else "?"
                pred_words.append('?')
                possible_predictions.append(' '.join(pred_words))
            next_prediction = '\n'.join(possible_predictions)
    
    else:
        # If prediciton is not available for the provided n_grams backoff
        residue = residue + " " + n_grams[0] 
        n_grams = n_grams[1:]
        next_prediction = predict_next_using_n_grams(n_grams, cpdist_list, mode)
        
    return residue + " " + next_prediction
    


# In[7]:

def generate_prediction(n_grams, mode="nsent"):
    n_grams = re.sub("[^\w\']", ' ', n_grams).lower()
    n_grams = tuple(nltk.word_tokenize(n_grams))
    return predict_next_using_n_grams(n_grams, cp_list, mode)


# In[ ]:



