import re
import spacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from pprint import pprint

def get_amounts(string):
    """Returns all the amounts found in the string 'sent' in a list

    Getting the amounts for example: RM10 RM10.00 RM100,000.00

    Parameters
    ----------
    sent : str
        original string without any prior preprocessing

    """
    split = string.split(' ')
    #Matches any string that begins with RM
    #contains only numbers, comma, period
    r = re.compile("\ARM[0-9,.]+")
    amount = list(filter(r.match, split))
    return amount

def get_bank_accounts(string):
    """Returns all the amounts found in the string 'sent' in a list

    Getting the amounts for example: RM10 RM10.00 RM100,000.00

    Parameters
    ----------
    sent : str
        original string without any prior preprocessing

    """
    split = string.split(' ')
    #Matches any string that contains only numbers or hyphens,
    #Minimum length of 6 characters, maximum length of 18 characters
    r = re.compile("[0-9\-]{6,18}")
    bank_accounts = list(filter(r.match, split))
    return bank_accounts

def get_entities(sent):
    """Returns all the entities found in the string 'sent' in a dictionary
    
    Doesn't do any prior preprocessing at all

    Parameters
    ----------
    sent : str
        string in which all the entities need to be extracted from

    """
    res = {}
    #Using spacy's pre-trained model to do high level classification of entities
    entities_dict = dict([(str(x), x.label_) for x in nlp(sent).ents])

    #Getting the amounts for example: RM10 RM10.00 RM100,000.00
    amounts = get_amounts(sent)
    bank_accounts = get_bank_accounts(sent)

    #Removing all amounts and bank_accounts from the dictionary created by spacy
    for key in amounts:
        if key in entities_dict:
            del entities_dict[key]
    for key in bank_accounts:
        if key in entities_dict:
            del entities_dict[key]

    #Grouping the keys according to their values
    res = {n:[k for k in entities_dict.keys() if entities_dict[k] == n] for n in set(entities_dict.values())}
    #Adding AMOUNT and BANK_ACC into the dictionary
    res['AMOUNT'] = amounts
    res['BANK_ACC'] = bank_accounts
    return res

if __name__ == '__main__':
    # Code mostly modified from https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
    # To test whether if it works just run this file: "py entity_extraction.py"
    from bs4 import BeautifulSoup
    import requests
    import re

    def url_to_string(url):
        res = requests.get(url)
        html = res.text
        soup = BeautifulSoup(html, 'html5lib')
        for script in soup(["script", "style", 'aside']):
            script.extract()
        return " ".join(re.split(r'[\n\t]+', soup.get_text()))

    ny_bb = url_to_string('https://www.theedgemarkets.com/article/bursa-malaysia-1q-net-profit-rm121m-rm65m-year-earlier')
    test_string = 'I want to deposit RM10 RM10.00 RM100,000.00 into my bank account 9990-6000-43141 99694039900 541421'
    entities_dict = get_entities(ny_bb)
    pprint(entities_dict)
    test_dict = get_entities(test_string)
    pprint(test_dict)