PART 1:



PART 2:

Simply run sentence_similarity.py which you will find in the Sentence Similarity folder and you will be prompted to enter the name of the model you want to use. Once sentence_similarity.py is done running, it will print in the terminal the resulting Pearson Correaltion of the model you chose. All these models should work given that you have all nescesary libraries installed (which we have specified in requirements.txt). The only exception to this is the InferSent model which will require you to download certain files in order for it to work. Below Ive included the steps to download the files to run InferSent.

Steps to download the required files for InferSent:

1. First cd into the Sentence Similarity Folder

2. Enter the following comands:

mkdir fastText
curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip fastText/crawl-300d-2M.vec.zip -d fastText/

mkdir encoder
curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl

3. InferSent should now successfully run next time you run sentence_similarity.py and prompt InferSent
