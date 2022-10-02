import spacy
import os
import torch
import subprocess
import numpy as np
import nltk
import torch
import tensorflow as tf
import tensorflow_hub as hub

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from models import InferSent



nlp_sent = spacy.load('en_core_web_sm') 
datasets = ["answer-answer", "headlines", "plagiarism", "postediting", "question-question"]

# extract sentences from txt file
# returns list that contains 5 sets of sentence pairs, and a list of 5 sets of gold standards for each sentence pair of each set
def readSentences(directory="./sts2016-english-with-gs-v1.0/"):
	gs_directories = [directory+"STS2016.gs.answer-answer.txt", directory+"STS2016.gs.headlines.txt", directory+"STS2016.gs.plagiarism.txt",
						directory+"STS2016.gs.postediting.txt", directory+"STS2016.gs.question-question.txt"]

	input_directories = [directory+"STS2016.input.answer-answer.txt", directory+"STS2016.input.headlines.txt", directory+"STS2016.input.plagiarism.txt",
							directory+"STS2016.input.postediting.txt", directory+"STS2016.input.question-question.txt"]
	
	sentence_datasets = []
	gold_standards_datasets = []
	# parse through each dataset
	for i in range(len(input_directories)):
		sentence_pairs = []
		gold_standards = []
		with open(input_directories[i]) as f:
			sentences_lines = f.readlines()
		with open(gs_directories[i]) as g:
			gold_standard_lines = g.readlines()
		# parse through each sentence pair of dataset and create a tuple out of them, as well as saving their similarity label
		for j in range(len(sentences_lines)):
			sentence_pair = sentences_lines[j].split("\t")
			gold_standard = gold_standard_lines[j]
			sentence_pairs.append((sentence_pair[0], sentence_pair[1]))
			gold_standards.append(gold_standard.replace("\n", ""))
		sentence_datasets.append(sentence_pairs)
		gold_standards_datasets.append(gold_standards)
	return sentence_datasets, gold_standards_datasets


########### model preprocessing functions


def d2v_preprocess(sentences, dataset):
	# parse through our 5 datasets of sentence pairs
	vocab_data = []
	tagged_data = []
	for dataset_num in range(len(datasets)):
		similarity_scores = []
		print("Processing:", datasets[dataset_num])
		# prase through each sentence pair of dataset
		for i in tqdm(range(len(sentences[dataset_num]))):
			vocab_data = vocab_data+[sentences[dataset_num][i][0], sentences[dataset_num][i][1]]

		tagged_data += [TaggedDocument(words=word_tokenize(_d.lower()), tags=[datasets[dataset_num]+"_"+str(i)]) for i, _d in enumerate(vocab_data)]
	
	d2v = Doc2Vec(vector_size=100,alpha=0.025, min_count=1)
	d2v.build_vocab(tagged_data)

	for epoch in tqdm(range(100)):
		d2v.train(tagged_data,
                total_examples=d2v.corpus_count,
                epochs=d2v.epochs)
    

	return d2v

def Bert_preprocess(model_name):

	model = SentenceTransformer(model_name)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.empty_cache()
	model.to(device)

	return model

def InferSent_preprocess(sentences, datasets):
	V=2
	MODEL_PATH = 'encoder/infersent%s.pkl' % V
	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
	                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
	infersent = InferSent(params_model)
	infersent.load_state_dict(torch.load(MODEL_PATH))

	#W2V_PATH = 'GloVe/glove.840B.300d.txt'
	W2V_PATH = 'fastText/crawl-300d-2M.vec'
	infersent.set_w2v_path(W2V_PATH)

	vocab_data = []
	for dataset_num in range(len(datasets)):
		similarity_scores = []
		print("Processing:", datasets[dataset_num])
		# prase through each sentence pair of dataset
		for i in tqdm(range(len(sentences[dataset_num]))):
			vocab_data = vocab_data+[sentences[dataset_num][i][0], sentences[dataset_num][i][1]]

	infersent.build_vocab(vocab_data)
	print(vocab_data)
	return infersent

def UniversalSentenceEncoder_preprocess():
	module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
	model = hub.load(module_url)
	print ("module %s loaded" % module_url)
	return model


########### cosine similarity functions


def FindSimilarityWithInferSent(sentence_pair, dataset_num, model, datasets):

	# create embeddigns of our two sentences

	query_embedding = model.encode(sentence_pair[0], tokenize=True)[0]
	passage_embedding = model.encode(sentence_pair[1], tokenize=True)[0]

	# Since I'm unsure on wether the embeddings are normalised we'll use the full cosine similarity function to be safe
	similarity = np.dot(query_embedding, passage_embedding)/(np.linalg.norm(query_embedding)*np.linalg.norm(passage_embedding))
	
	return similarity

def FindSimilarityWithD2V(dataset_num, model, datasets, i):
	
	similarity = np.dot(model.dv.get_vector(datasets[dataset_num]+"_"+str(i), norm=True), model.dv.get_vector(datasets[dataset_num]+"_"+str(i+1), norm=True))
	return similarity


def FindSimilarityWithTransformer(sentence_pair, dataset_num, model, datasets):
	
	# create embeddigns of our two sentences
	query_embedding = model.encode(sentence_pair[0])
	passage_embedding = model.encode(sentence_pair[1])

	# find their cosine similarity by calculating dot score of their embeddings
	similarity = util.dot_score(query_embedding, passage_embedding)[0][0]
	
	return similarity.item()


def FindSimilarityWithUSE(sentence_pair, dataset_num, model, datasets):

	# create embeddigns of our two sentences
	query_embedding = model([sentence_pair[0]])[0]
	passage_embedding = model([sentence_pair[1]])[0]

	similarity = np.dot(query_embedding, passage_embedding)/(np.linalg.norm(query_embedding)*np.linalg.norm(passage_embedding))

	return similarity

########### Evaluation function


# evaluate similarity for sentences with given model
def EvaluateSimilarity(sentences, datasets, model_name):

	match model_name:
		case "Bert":
			model = Bert_preprocess("all-mpnet-base-v2")
		case "D2V":
			model = d2v_preprocess(sentences, datasets)
		case "InferSent":
			model = InferSent_preprocess(sentences, datasets)
		case "USE":
			model = UniversalSentenceEncoder_preprocess()
		case _:
			raise Exception("Invalid Model Name") 


	similarity_score = []

	# parse through our 5 datasets of sentence pairs
	for dataset_num in range(len(datasets)):
		similarity_scores = []
		print("Processing:", datasets[dataset_num])
		# prase through each sentence pair of dataset
		for i in tqdm(range(len(sentences[dataset_num]))):
			# calculate their cosine similarity
			match model_name:
				case "Bert":
					similarity_scores.append(str(FindSimilarityWithTransformer(sentences[dataset_num][i], dataset_num, model, datasets)))
				case "D2V":
					similarity_scores.append(str(FindSimilarityWithD2V(dataset_num, model, datasets, i)))
				case "InferSent":
					similarity_scores.append(str(FindSimilarityWithInferSent(sentences[dataset_num][i], dataset_num, model, datasets)))
				case "USE":
					similarity_scores.append(str(FindSimilarityWithUSE(sentences[dataset_num][i], dataset_num, model, datasets)))
		# write down our results in txt file
		with open("./sts2016-english-with-gs-v1.0/SYSTEM_OUT."+datasets[dataset_num]+".txt", 'w') as f:
			for sim in similarity_scores:
				f.write(sim)
				f.write("\n")




sentences, labels = readSentences()


# Choices of model names: Bert, D2V, InferSent, USE
model = EvaluateSimilarity(sentences, datasets, "USE")


subprocess.run("./correlation-noconfidence.pl STS2016.gs.headlines.txt SYSTEM_OUT.headlines.txt", shell=True, cwd='./sts2016-english-with-gs-v1.0')


