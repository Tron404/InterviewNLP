import numpy as np
import pandas as pd

from collections import defaultdict

def encode_sentence_bow(sentence: str, vocab: defaultdict|dict) -> np.ndarray:
     return np.asarray([int(vocab_token in sentence) for vocab_token in vocab.keys()])

def create_vocabulary(data: pd.Series) -> defaultdict:
     vocabulary = defaultdict(int)

     for sentence in data:
          for token in sentence:
               vocabulary[token] += 1

     vocabulary = dict(sorted(vocabulary.items())) # avoid situations such as [1 1 1 1 0 0 ...]

     return vocabulary

def encode_bow(x_data: pd.DataFrame, y_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
     vocab = create_vocabulary(x_data["Text"])
     x_data["Encoded_Text"] = x_data["Text"].apply(func = encode_sentence_bow, args = (vocab,))
     y_data["Encoded_Text"] = y_data["Text"].apply(func = encode_sentence_bow, args = (vocab,))

     return x_data, y_data