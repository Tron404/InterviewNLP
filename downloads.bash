# download a spacy model
python -m spacy download en_core_web_sm -q

# download the set of stopwords from nltk
python -m nltk.downloader stopwords -q

# required to download BERT models
git lfs install

# unzip directive/provision data
if [ ! -d $1 ]; then
    mkdir $1
    unzip dataset.zip -d $1
else
    echo "The data were already unpacked"
fi