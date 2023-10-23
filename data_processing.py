import pandas as pd
import re
import spacy
import string

from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from xml.etree import ElementTree as ET

lemmatizer = spacy.load("en_core_web_sm")
stopwords = stopwords.words("english")

def data_preprocessing_transformer(data: str) -> list:
    """
    Minimal preprocessing for transformers. This includes:
    * lowercasing
    * replacement of all whitespace characters with " "
    * and removing any surplus " " characters
    * tokenization based on single " "
    """
    data = data.lower()
    data = re.sub(r"\s+", " ", data)
    data = data.strip()
    return data.split(" ")

# main processing: keep compound nouns, lemmatize, stop word removal, remove numbers
def data_preprocessing(data: str) -> list:
    """
    Main preprocessing pipeline. Tokenize a given sentence such that:
    * all tokens are lower-cased
    * compound nouns are kept (e.g. hand-arm)
    * the tokens are lemmatized
    * any stop words, isolated punctuation, and tokens with numeric characters are removed
    """
    data = data.lower()
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ from string.punctuation
    # keep compund nouns
    data = regexp_tokenize(data, r"[\w]+(?:[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~][\w]+[()*[\]{}]?){1,}|[\w]+|(?:[([{]\w+[)\]}])+")

    # lemmatize
    aux_data = data
    data = " ".join(data)
    lemmatized_data = lemmatizer(data)
    data = [token.lemma_ for token in lemmatized_data]

    # remove all punctuation
    data = [word for word in data if len(word) > 1]
    
    # some of the compound-nouns are separated by the lemmatizer, as such it is necessary to readd those tokens 
    # back as they were, but still in their lemmatized forms
    aux = []
    for aux_token in aux_data:
        found = False
        for token in lemmatized_data:
            if token.text == aux_token:
                aux.append(token.lemma_)
                found = True
            if found:
                break
        if not found:
            aux.append(aux_token)
    data = aux

    # remove stop words, any remaining punctuation, and tokens containing digits
    data = [word for word in data if word not in stopwords and word not in string.punctuation and not re.search(r"[0-9]+", word)]

    return data

def get_text_from_xml(xml_file: str, preprocessing_func: callable) -> pd.DataFrame:    
    """
    Obtain two datasets of articles from a given XML file path, one for the sentence transformer model
    and the second for all other models
    """
    
    parsed_dir = ET.parse(xml_file)
    parsed_dir = parsed_dir.getroot()

    data = {"ID": [], "Text": []}
    for child_node in parsed_dir:
        id = child_node.attrib["id"]
        if "P" != id[0] and id[0] != "A":
            id = "P" + id
        data["ID"].append(id)
        data["Text"].append(child_node.text)

    data = pd.DataFrame(data)
    data["Text"] = data["Text"].apply(preprocessing_func)

    return data

if __name__ == "__main__":
    directive_data = get_text_from_xml("DIR_EN_32002L0044.xml")
    provision_data = get_text_from_xml("NIM_EN.xml")

    directive_data.head()
