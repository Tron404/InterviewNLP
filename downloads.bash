# download a spacy model
python -m spacy download en_core_web_sm

# unzip directive/provision data
if [ ! -d $1 ]; then
    mkdir $1
    unzip dataset.zip -d $1
else
    echo "The data were already unpacked"
fi