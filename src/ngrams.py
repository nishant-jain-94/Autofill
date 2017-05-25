import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.util import ngrams
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist
import json


# ### Loading Data
with open("../data/squad_wiki_data.json","r") as outfile:
    dataset = json.load(outfile)

questions = dataset[0]['Question']
questions = ' '.join(questions)


# ### Generatng n_grams-bigrams
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
cp_list = cpdist_training_n_plus_1_grams(questions, 5)

# ### Predict the next word function
def predict_next_using_n_grams(n_grams, cpdist_list, mode = "nsent"):
    # n_gram = tuple of n_words #input to the function
    len_n_grams = len(n_grams)
    
    # to end the recursion
    if(len_n_grams == 0):return #no prediction available
    
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
        
def generate_prediction(n_grams, mode="nsent"):
    n_grams = re.sub("[^\w\']", ' ', n_grams).lower()
    n_grams = tuple(nltk.word_tokenize(n_grams))
    ans = ' '.join(n_grams)+'\n'
    ans+= predict_next_using_n_grams(n_grams, cp_list, mode)
    return ans