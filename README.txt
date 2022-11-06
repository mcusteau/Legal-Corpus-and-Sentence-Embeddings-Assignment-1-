Michel Custeau,  8658589
Beril Borali, 300036112


PART 1:

For the necessary libraries, please see the "requirements.txt" file. The data folder needs to be named as "CUAD_v1" and be in the main directory . You need Python 3.10 to run the program. For running the program:

1. cd into the main folder (Legal-Corpus-and-Sentence-Embeddings-Assignment-1-)

2. Enter "python corpus_processing.py"



PART 2:

Make sure you are running on python 3.10 

Make sure to run python3 -m spacy download en_core_web_sm in the terminal. Also make sure to run in a command prompt nltk.download('punkt'). We have listed all necessary libraries in the "requirements.txt" file which is located in the main directory.  Make sure that the initial data folder sts2016-english-with-gs-v1.0 is located inside the Sentence Similarity folder. The correlation-noconfidence.pl will have to be set as executable, you can do this with the command chmod +x correlation-noconfidence.pl

To run the sentence similarity script, simply run "python3 sentence_similarity.py" inside the Sentence Similarity folder and once you run it, you will be prompted to enter the name of the model you want to use. Once "sentence_similarity.py" is done running, it will print in the terminal the resulting Pearson Correlation of the model you chose. If you want to run the InferSent model, you will be required to download certain files in order for it to work. Below I've included the steps to download the files to run InferSent.

Steps to download the required files for InferSent:

1. First cd into the Sentence Similarity Folder

2. Enter the following commands:

mkdir fastText
curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip fastText/crawl-300d-2M.vec.zip -d fastText/

mkdir encoder
curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl

3. InferSent should now successfully run next time you run sentence_similarity.py and prompt InferSent
