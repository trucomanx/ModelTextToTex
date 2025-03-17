from datasets import load_dataset

import nltk
nltk.download('punkt_tab') # needed to use sent_tokenize
from nltk.tokenize import sent_tokenize # A função é usada para dividir um texto em sentenças ou frases.     
#from nltk.tokenize import word_tokenize

#

def load_wiki_dataset(max_len=50000):
    dataset = load_dataset("wikipedia", "20220301.simple")

    #print(dataset)
    #print(dataset['train'])

    

    # divide o texto em sentenças
    text = []
    for data in dataset['train']:
      text.extend(sent_tokenize(data['text']))
    
    if max_len<len(text):
        return text[:max_len]
    
    return text
