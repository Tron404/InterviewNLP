import pandas as pd
import re
import spacy
import string

from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from xml.etree import ElementTree as ET

lemmatizer = spacy.load("en_core_web_sm")
stopwords = stopwords.words("english")

# main processing: keep compound nouns, lemmatize, stop word removal, remove numbers
def data_preprocessing(data: str) -> list:
    data = data.lower()
    # @TODO try without compound nouns
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ from string.punctuation
    # keep compund nouns
    data = regexp_tokenize(data, r"[\w]+(?:[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~][\w]+[()*[\]{}]?){1,}|[\w]+|(?:[([{]\w+[)\]}])+")

    ### @TODO: 
    # * lemmatize
    aux_data = data
    data = " ".join(data)
    lemmatized_data = lemmatizer(data)

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

    # stop word removal
    data = [word for word in data if word not in stopwords]

    #### remove numbers
    # data = [word for word in data if not re.search(r"[0-9]+", word)]

    return data

def get_text_from_xml(xml_file: str) -> pd.DataFrame:    
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
    data["Text"] = data["Text"].apply(data_preprocessing)

    return data

if __name__ == "__main__":
    directive_data = get_text_from_xml("DIR_EN_32002L0044.xml")
    provision_data = get_text_from_xml("NIM_EN.xml")

    directive_data.head()
