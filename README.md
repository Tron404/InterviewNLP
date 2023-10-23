# InterviewNLP

This Jupyter notebook ("main.py") is entirely plug-in and use. It only requires the presence of "data.zip" in the current working directory. Everything that was developed in this repository uses Python `3.11.5`, though any Python version from the `3.11.x` family should work. There is a `requirements.txt` file as well that can be used to install all of the required modules in a python virtual environment or conda environment. The implementation can be summarized as follows:

*Four models + baseline*
* Jaccard for lexical similarity (baseline)
* TF-IDF
* BERT embeddings with pooling
* Sentence-trained DistilRoBERTa (from sentence_transformers)
* Doc2vec

*Other models were also considered, but were discarded due to low peformance compared to the chosen approaches:*
* Word2vec with pooling
* BERT embbdings with pooling and weighted by TF-IDF scores
* One-hot encoding BoW
