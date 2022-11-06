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
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from models import InferSent



nlp_sent = spacy.load('en_core_web_sm') 
dataset_2016 = ["answer-answer", "headlines", "plagiarism", "postediting", "question-question"]
dataset_2012_test = ["MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews"]
dataset_2012_train = ["MSRpar", "MSRvid", "SMTeuroparl"]
dataset_2012_trial = ["trial"]
dataset_2013 = ["FNWN", "headlines", "OnWN"]
dataset_2014 = ["OnWN", "images", "deft-news", "headlines", "deft-forum", "tweet-news"]
dataset_2015 = ["answers-forums", "answers-students", "belief", "headlines", "images"]

datasets = {
'2012_test':('./data/STS2012-en-test/test-gold/', dataset_2012_test, 'STS'),
'2012_train':('./data/STS2012-en-train/train/', dataset_2012_train, 'STS'),
'2012_trial':('./data/STS2012-en-trial/trial/', dataset_2012_trial, 'STS'),
'2013':('./data/STS2013-en-test/test-gs/', dataset_2013, 'STS'),
'2014':('./data/STS2014-en-test/sts-en-test-gs-2014/', dataset_2014, 'STS'),
'2015_raw':('./data/STS2015-en-rawdata-scripts/sts2015-en-post/data', dataset_2015, 'STS'),
'2015_test':('./data/STS2015-en-test/test_evaluation_task2a/', dataset_2015, 'STS'),
'2016':('./data/sts2016-english-with-gs-v1.0/', dataset_2016, 'STS2016')}

# extract sentences from txt file
# returns list that contains 5 sets of sentence pairs, and a list of 5 sets of gold standards for each sentence pair of each set
def readSentences(dataset_year):
	directory=datasets[dataset_year][0]
	dataset=datasets[dataset_year][1]
	year=datasets[dataset_year][2]
	
	if(dataset_year=='2015_raw'):
		gs_directories = [directory+'/gs/'+year+".gs." +group+".txt" for group in dataset]
		input_directories = [directory+'/filter/'+group for group in dataset]
	else:
		gs_directories = [directory+year+".gs." +group+".txt" for group in dataset]
		input_directories = [directory+year+".input."+group+".txt" for group in dataset]
	
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
			sentence_pair = sentences_lines[j].replace("\n", "").split("\t")
			if(dataset_year=='2012_trial'):
				gold_standard = gold_standard_lines[j].split("\t")[1].replace("\n", "")
			else:
				gold_standard = gold_standard_lines[j].replace("\n", "")
			if(dataset_year=='2015_raw'):
				sentence_pairs.append((sentence_pair[4], sentence_pair[5]))
			elif(dataset_year=='2012_trial'):
				sentence_pairs.append((sentence_pair[1], sentence_pair[2]))
			else:
				sentence_pairs.append((sentence_pair[0], sentence_pair[1]))
			gold_standards.append(gold_standard)
		sentence_datasets.append(sentence_pairs)
		gold_standards_datasets.append(gold_standards)
	return sentence_datasets, gold_standards_datasets, dataset


########### model preprocessing functions


def d2v_preprocess(sentences, dataset):
	# parse through our 5 datasets of sentence pairs
	# vocab_data = []
	# tagged_data = []
	# for dataset_num in range(len(datasets)):
	# 	similarity_scores = []
	# 	print("Processing:", datasets[dataset_num])
	# 	# prase through each sentence pair of dataset
	# 	for i in tqdm(range(len(sentences[dataset_num]))):
	# 		vocab_data = vocab_data+[sentences[dataset_num][i][0], sentences[dataset_num][i][1]]

	# 	tagged_data += [TaggedDocument(words=word_tokenize(_d.lower()), tags=[datasets[dataset_num]+"_"+str(i)]) for i, _d in enumerate(vocab_data)]
	
	# # create our model
	# d2v = Doc2Vec(vector_size=50,alpha=0.025, min_count=1)
	# d2v.build_vocab(tagged_data)

	# # create embedding of sentences
	# for epoch in tqdm(range(20)):
	# 	d2v.train(tagged_data,
 #                total_examples=d2v.corpus_count,
 #                epochs=d2v.epochs)
    
	# d2v.save("./d2v_model")
	return Doc2Vec.load("./d2v_model")

def transformer_preprocess(model_name):

	#load model
	model = SentenceTransformer(model_name)
	
	# check if we can use GPU for training
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

	# Build Vocabulary
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
	# Load Model
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
	
	# find their cosine similarity by calculating dot score of their embeddings
	similarity = np.dot(model.dv.get_vector(datasets[dataset_num]+"_"+str(i), norm=True), model.dv.get_vector(datasets[dataset_num]+"_"+str(i+1), norm=True))
	return similarity


def FindSimilarityWithTransformer(sentence_pair, dataset_num, model, datasets):
	
	# create embeddigns of our two sentences
	query_embedding = model.encode(sentence_pair[0], normalize_embeddings=True)
	passage_embedding = model.encode(sentence_pair[1], normalize_embeddings=True)

	# find their cosine similarity by calculating dot score of their embeddings
	similarity = util.dot_score(query_embedding, passage_embedding)[0][0]
	
	return similarity.item()


def FindSimilarityWithUSE(sentence_pair, dataset_num, model, datasets):

	# create embeddigns of our two sentences
	query_embedding = model([sentence_pair[0]])[0]
	passage_embedding = model([sentence_pair[1]])[0]

	# find the cosine similarity of the embeddings
	similarity = np.dot(query_embedding, passage_embedding)/(np.linalg.norm(query_embedding)*np.linalg.norm(passage_embedding))

	return similarity

########### Training Function function

def TrainTransformer(sentence_pairs, labels):

	train_examples = []
	for i in range(len(sentence_pairs)):
		if(labels[i]!=''):
			train_examples.append(InputExample(texts=[sentence_pairs[i][0], sentence_pairs[i][1]], label=float(labels[i])/5))

	train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
	train_loss = losses.CosineSimilarityLoss(model)

	model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100, output_path="./models/new_model_test")

def InitiateTraining(model, year):
	sentences, labels, dataset_groups = readSentences(year)
	for dataset_num in tqdm(range(len(dataset_groups))):
		sentence_pairs = sentences[dataset_num]
		gs = labels[dataset_num]
		TrainTransformer(sentence_pairs, gs)

########### Evaluation function


# evaluate similarity for sentences with given model
def EvaluateSimilarity(sentences, datasets, model_name):

	match model_name:
		case "Mpnet":
			model = transformer_preprocess("all-mpnet-base-v2")
		case "D2V":
			model = d2v_preprocess(sentences, datasets)
		case "InferSent":
			model = InferSent_preprocess(sentences, datasets)
		case "USE":
			model = UniversalSentenceEncoder_preprocess()
		case "Roberta":
			model = transformer_preprocess("usc-isi/sbert-roberta-large-anli-mnli-snli")
		case "trained":
			model = transformer_preprocess("./models/model_2012_train_8epochs_batch8")
		case _:
			raise Exception("Invalid Model Name") 



	# parse through our 5 datasets of sentence pairs
	for dataset_num in range(len(datasets)):
		similarity_scores = []
		print("Processing:", datasets[dataset_num])
		# parse through each sentence pair of dataset
		for i in tqdm(range(len(sentences[dataset_num]))):
			# calculate their cosine similarity
			match model_name:
				case "Mpnet":
					similarity_scores.append(str(FindSimilarityWithTransformer(sentences[dataset_num][i], dataset_num, model, datasets)))
				case "D2V":
					similarity_scores.append(str(FindSimilarityWithD2V(dataset_num, model, datasets, i)))
				case "InferSent":
					similarity_scores.append(str(FindSimilarityWithInferSent(sentences[dataset_num][i], dataset_num, model, datasets)))
				case "USE":
					similarity_scores.append(str(FindSimilarityWithUSE(sentences[dataset_num][i], dataset_num, model, datasets)))
				case "Roberta":
					similarity_scores.append(str(FindSimilarityWithTransformer(sentences[dataset_num][i], dataset_num, model, datasets)))
				case "trained":
					similarity_scores.append(str(FindSimilarityWithTransformer(sentences[dataset_num][i], dataset_num, model, datasets)))
		# write down our results in txt file
		with open("./data/sts2016-english-with-gs-v1.0/SYSTEM_OUT."+datasets[dataset_num]+".txt", 'w') as f:
			for sim in similarity_scores:
				f.write(sim)
				f.write("\n")


def main():

	model_name = input("\nEnter your choice of model: Mpnet, D2V, InferSent, USE, Roberta\n")

	sentences, labels, datasets = readSentences('2016')

	# Choices of model names: Mpnet, D2V, InferSent, USE, Roberta
	model = EvaluateSimilarity(sentences, datasets, model_name)

	# calcualte Pearson correlation 
	with open('./data/sts2016-english-with-gs-v1.0/outputs.txt', 'w') as outfile:
	    for dataset_num in range(len(datasets)):
	        with open("./data/sts2016-english-with-gs-v1.0/SYSTEM_OUT."+datasets[dataset_num]+".txt") as infile:
	            for line in infile:
	                outfile.write(line)

	with open('./data/sts2016-english-with-gs-v1.0/inputs.txt', 'w') as outfile:
	    for dataset_num in range(len(datasets)):
	        with open("./data/sts2016-english-with-gs-v1.0/STS2016.gs."+datasets[dataset_num]+".txt") as infile:
	            for line in infile:
	                outfile.write(line)

	subprocess.run("./correlation-noconfidence.pl inputs.txt outputs.txt", shell=True, cwd='./data/sts2016-english-with-gs-v1.0')


main()

# model = transformer_preprocess("usc-isi/sbert-roberta-large-anli-mnli-snli")

# InitiateTraining(model, '2012_train')