# InterviewNLP

This Jupyter notebook ("main.py") is entirely plug and use. It only requires the presence of "data.zip" in the current working directory.

Four models + baseline
* Jaccard for lexical similarity (baseline)
* TF-IDF
* BERT embeddings with pooling
* Sentence-trained DistilRoBERTa (from sentence_transformers)
* Doc2vec

Other models were also considered, but were discarded due to low peformance compared to the chosen approaches:
* Word2vec with pooling
* BERT embbdings with pooling and weighted by TF-IDF scores
* One-hot encoding BoW