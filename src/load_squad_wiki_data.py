
# coding: utf-8

# In[1]:

import os
import json
import requests
from load_from_wiki import load_data


# In[2]:

def processing_squad_data(dataset):
    raw_data = []
    for data in dataset['data']:
        paragraphs = data['paragraphs']       
        for paragraph in paragraphs:
            para_ques_dict = {}
            para_ques_dict['Passages'] = paragraph['context']
            ques_list = []
            for questions in paragraph['qas']:
                ques_list.append(questions['question'])
            para_ques_dict['Question'] = list(set(ques_list)) 
            raw_data.append(para_ques_dict)
    return raw_data


# In[3]:

def combine_squad_dev_train():    
    with open('../data/dev-v1.1.json') as data_file:
        dataset = json.load(data_file)
        dev_set = processing_squad_data(dataset)
    with open('../data/train-v1.1.json') as data_file:
        dataset = json.load(data_file)
        train_set = processing_squad_data(dataset)
    dev_set.extend(train_set)
    with open('../data/squad_data.json', 'w') as outfile:
        json.dump(dev_set , outfile)    


# In[4]:

def merge_file():
    """Merges two files squad_data and wiki_data and generates an merged file squad_wiki_data.json  """
    with open('../data/squad_data.json') as data_file:
        dataset1 = json.load(data_file)
    with open('../data/wiki_data.json') as data_file:
        dataset2 = json.load(data_file)
    final_dict = {}
    final_para = []
    final_question = []
    for data in dataset1:
        final_para.append(data['Passages'])
        final_question.extend(data['Question'])
    for data in dataset2:
        final_para.append(data['Passage'])
        final_question.extend(data['Question'])
    final_dict['Paragraph'] = ''.join(final_para)
    final_dict['Question'] = final_question
    final_data = []
    final_data.append(final_dict)
    with open('../data/squad_wiki_data.json','w') as outfile:
        json.dump(final_data , outfile)


# In[5]:

def load_squad_wiki_data():
    if not os.path.isfile("../data/squad_wiki_data.json"):
        # Check if the train-v1.1.json exists
        if not os.path.isfile("../data/train-v1.1.json"):
            print("Loading Squad Training Data")
            response = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json")
            with open("../data/train-v1.1.json", "wb") as outfile:
                for data in response.iter_content():
                    outfile.write(data)

        # Check if the dev-v1.1.json exists
        if not os.path.isfile("../data/dev-v1.1.json"):
            print("Loading Squad Dev Data")
            response = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
            with open("../data/dev-v1.1.json", "wb") as outfile:
                for data in response.iter_content():
                    outfile.write(data)

        # Check if the squad_data exists if not generate squad_data.json
        if not os.path.isfile("../data/squad_data.json"):
            print("Combining Squad Data")
            combine_squad_dev_train()

        # Check if the wiki_data exists else call the respective script to load it
        if not os.path.isfile("../data/wiki_data.json"):
            print("Loading Wiki Data")
            load_data()

        merge_file()  


# In[6]:

def get_squad_wiki_data():
    load_squad_wiki_data()
    with open("../data/squad_wiki_data.json", "r") as dataset:
        squad_wiki_data = json.load(dataset)
    return squad_wiki_data


# In[8]:

# data = get_squad_wiki_data()


# In[11]:

# type(data[0]["Question"])


# In[ ]:



