
# coding: utf-8

# # N-gram-Approach

# ### importing require packages

# In[14]:

import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.util import ngrams
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist
import json


# ### Loading Data

# In[15]:

with open("../data/squad_wiki_data.json","r") as outfile:
    dataset = json.load(outfile)


# In[16]:

questions = dataset[0]['Question']
questions = ' '.join(questions)


# ### Generatng n_grams-bigrams

# In[17]:

def cpdist_training_n_plus_1_grams(training_passage, n_plus_one):
    
    ## removing special character and symbols and converting to lower case
    training_passage = re.sub(r"(^\w|\')", ' ', training_passage).lower()
    
    ## tokenizing the sanitized passage
    words = nltk.word_tokenize(training_passage)
    
    cfdist_list = []
    cpdist_list = []
    
    ## generating cpdist and n_grams for n_plus_one to bigrams
    for n in range(n_plus_one,1,-1):
        ## generating n_plus_one_grams and converting into list
        n_grams_generated = list(ngrams(words, n))
        ## converting into (n_gram, n+1 words) for prediction
        n_grams_for_predict = [(n_gram[:-1],n_gram[-1]) for n_gram in n_grams_generated] 
        
        ## calculating conditionalfrequency for all n_grams
        cfdist = ConditionalFreqDist(n_grams_for_predict)
        
        ## calculating conditional probablitlity of next word for all n_grams
        cpdist = ConditionalProbDist(cfdist,MLEProbDist)
        
        cfdist_list.append(cfdist)
        cpdist_list.append(cpdist)
    
    return cpdist_list


# In[18]:

cp_list = cpdist_training_n_plus_1_grams(questions, 5)


# In[33]:

#len(cp_list)


# ### Predict the next word function

# In[220]:

def predict_next_using_n_grams(n_grams, cpdist_list, mode="nsent"):
    # n_gram = tuple of n_words #input to the function
    len_n_grams = len(n_grams)
    
    # to end the recursion
    if(len_n_grams==0):return #no prediction available
    
    #handlind sentence with length more than the n_grams generated
    if len_n_grams > len(cpdist_list): 
        n_grams = n_grams[-len(cpdist_list):]
        len_n_grams = len(n_grams)
    
    # possible predictions
    possible_pred = list(cpdist_list[len(cpdist_list)-len_n_grams][n_grams].samples())
    # number of possible prediciton for the provided n_grams
    n_possible_pred = len(possible_pred)
    
    if n_possible_pred!=0:
        
        if(mode == 'nword'):
            if(n_possible_pred == 1):
            #pred = possible_pred[0]
            #tuple(list(n_grams).append(pred)[1:])
                return possible_pred[0]
            else:
            ## trying combining multiple probability
                return '\n'.join(possible_pred[0:5])
            
            
        if(mode == 'nsent'):
            if(n_possible_pred == 1):
                pred_words = []
                n_grams_list = list(n_grams)
                while(possible_pred[0]!='?'):
                    pred_words.append(possible_pred[0])
                    n_grams_list.append(possible_pred[0])                
                    n_grams_list = n_grams_list[1:]
                    possible_pred = list(cpdist_list[len(cpdist_list)-len_n_grams][tuple(n_grams_list)].samples())
                pred_words.append('?')
                return ' '.join(pred_words)
            
            else:
            ## returning 2 sentence
                preds = []
                pred1, pred2 = [],[]
                n_grams_list1, n_grams_list2 = list(n_grams), list(n_grams)
                
                possible_pred1 = possible_pred[0]
                possible_pred2 = possible_pred[1]
                
                while(possible_pred1!='?'):
                    pred1.append(possible_pred1)
                    n_grams_list1.append(possible_pred1)                
                    n_grams_list1 = n_grams_list1[1:]
                    possible_pred1 = list(cpdist_list[len(cpdist_list)-len_n_grams][tuple(n_grams_list1)].samples())[0]
                    
                pred1.append('?')
                preds.append(' '.join(pred1))
                
                while(possible_pred2!='?'):
                    pred2.append(possible_pred2)
                    n_grams_list2.append(possible_pred2)                
                    n_grams_list2 = n_grams_list2[1:]
                    possible_pred2 = list(cpdist_list[len(cpdist_list)-len_n_grams][tuple(n_grams_list2)].samples())[0]
                    
                pred2.append('?')
                preds.append(' '.join(pred2))
                return ' '.join(preds)
            
    else:
        #if prediciton is not available for the provided n_grams backoff
        n_grams = n_grams[1:]
        return predict_next_using_n_grams(n_grams, cpdist_list,mode)
        
    


# In[225]:

def generate_predcition(n_grams, mode="nsent"):
    n_grams = re.sub("[^\w\']", ' ', n_grams).lower()
    n_grams = tuple(nltk.word_tokenize(n_grams))
    ans = ' '.join(n_grams)+'\n'
    ans+= predict_next_using_n_grams(n_grams, cp_list, mode)
    return ans


# In[227]:

generate_predcition('this is the final','nword')


# In[11]:

#t = ("who","is","the","oldest")
#' '.join(t)


# In[112]:

#"who","is","the","oldest","quarterback","to","play","in","a","super","bowl","by","the","time","they","reached","super","bowl","50"


# In[93]:

#t =("who","is","the","oldest","quarterback")
## check why this is still predicting

#'super', 'bowl', '50', '?']
#13


# In[94]:

#t = list(t)
#t.append('a')
#t


# In[92]:

#type(t)


# ### keeping only a-z and 0-9

# In[6]:

#questions = re.sub(r"(^\w|\')", ' ', questions)


# In[7]:

#questions = questions.lower()


# In[8]:

#words = nltk.word_tokenize(questions)


# ### n-grams : 4 grams

# In[17]:

#_4grams = ngrams(words,4)


# In[18]:

#_4grams = list(_4grams)


# In[19]:

#_4grams


# In[106]:

#cfdist = ConditionalFreqDist()


# In[20]:

#_4grams_to_predict_3grams =  [(_4gram[:-1],_4gram[-1]) for _4gram in _4grams]


# In[32]:

#_4grams_to_predict_3grams


# In[21]:

#cfdist = ConditionalFreqDist(_4grams_to_predict_3grams)


# In[22]:

#cfdist
# ('able',): FreqDist({'to': 1}),


# In[172]:

#cfdist[("stamp",)]
#did not stamp out the 


# In[111]:

#cpdist = ConditionalProbDist(cfdist,MLEProbDist)


# In[ ]:

#nltk.probability.MLEProbDist.freqdist


# In[112]:

#def samples(tup):
#    return list(cpdist[tup].samples())
    #print(key)


# In[113]:

#samples(("stamp", "out", "the"))


# In[125]:

#cpdist[("empire", "during", "much")].prob('of')


# In[119]:

#list(range(4,1,-1))

