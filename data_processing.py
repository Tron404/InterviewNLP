import pandas as pd
import re
import spacy
import string

from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from xml.etree import ElementTree as ET

lemmatizer = spacy.load("en_core_web_sm")
stopwords = stopwords.words("english")

# minimal preprocessing for transformers
def data_preprocessing_transformer(data: str) -> list:
    data = data.lower()
    data = re.sub(r"\s+", " ", data)
    data = data.strip()
    return data.split(" ")

# main processing: keep compound nouns, lemmatize, stop word removal, remove numbers
def data_preprocessing(data: str) -> list:
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
