import re                                            # Regular python module for regex functions.
import json                                          # Output into Json file.
from wikipedia import WikipediaPage                  # Provides some attributes and function to read data from wikipedia. i.e. Titles, Summary, Context, Images.
from wikipedia import DisambiguationError, PageError # Error thrown in case of Disambiguity.

def normalize_passage(text_str):

    """ Remove some unnecessary characters from the data using regular python regex module."""

    text_str = re.sub(r"(    )|(\n)|(\\displaystyle)|(\\)|(Edit ===)|(Edit ==)|(===)|(==)", "", text_str)

    return text_str

def read_from_wiki(titles):

    """ This function will except a list(list of Strings) of all the titles
    and we will use these strings to get test from wikipedia."""

    out_data_list = []                                                                   # List to append the dictionary elements(i.e. Required data with keys and values.) into one list.

    for index, title in enumerate(titles):
        out_data_dict = {'Title': title ,'Passage':'', "Question": [] ,"Error" : None }  # Will store our processed text into dictionary. {key:'Passage', value:'Text'}

        try:
            get_wiki_data = WikipediaPage(title = title)                                 # Get all the data from wikipedia.

        except DisambiguationError:
            # If there is any disambiguity in the Title name.
            out_data_dict["Error"] = ("There is Disambigity in the title : " + title + ". Please provide more precise title.")

        except PageError:
            # If no page found with the given title.
            out_data_dict["Error"] = ("Page id " + title + " does not match any pages. Try another id!")

        if not out_data_dict["Error"]:
            # If there is no error then store the passage.
            content_only = get_wiki_data.content             # Store main content into a variable.
            processed_text = normalize_passage(content_only) # Process text using normalize_passge().
            out_data_dict['Passage'] = processed_text        # Store received text into dictionary.
            out_data_list.append(out_data_dict)              # Now append each dictionary into List.

    return out_data_list

def write_to_json(data):

    """ Convert list of dictionary into a json file and save it into data/wiki_text.json file. """

    with open('../data/wiki_data.json', 'w', encoding='utf8') as outfile:
        data_dump = json.dumps(data, indent=4, separators=(',', ': '))
        outfile.write(data_dump)

topics = ['Super Bowl 50', 'Warsaw', 'Normans', 'Nicola Tesla', 'Computational complexity theory', 'Teacher', 'Martin Luther',
     'Southern California', 'Sky (United Kingdom)', 'Victoria (Australia)', 'Huguenot', 'Steam engine', 'Oxygen',
    '1973 oil crisis', 'Apollo program', 'European Union Law', 'Amazon rainforest', 'Ctenophora','Fresno California',
    'Packet switching', 'Black Death', 'Geology', 'Newcastle upon Tyne' , 'Victoria and Albert Museum', 'American Broadcasting Company',
    'Genghis Khan', 'Pharmacy', 'Immune system', 'Civil disobedience', 'Construction' , 'Private school', 'Harvard University',
    'Jacksonville, Florida', 'Economic inequality', 'Doctor Who', 'University of Chicago', 'Yuan dynasty', 'Kenya', 'Intergovernmental Panel on Climate Change',
    'Chloroplast', 'Prime number', 'Rhine' , 'Scottish Parliament', 'Islamism', 'Imperialism', 'United Methodist Church', 'French and Indian War',
    'Force']

def load_data():
    """Loads data from Wikipedia"""
    wiki_data = read_from_wiki(topics)
    write_to_json(wiki_data)
