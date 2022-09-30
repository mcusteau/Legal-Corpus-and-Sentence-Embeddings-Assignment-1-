import spacy
import os
import torch
import subprocess
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util



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

# cosine similarity function
def FindSimilarityWithTransformer(sentence_pair, dataset_num, model, datasets):
	
	# create embeddigns of our two sentences
	query_embedding = model.encode(sentence_pair[0])
	passage_embedding = model.encode(sentence_pair[1])

	# find their cosine similarity by calculating dot score of their embeddings
	similarity = util.dot_score(query_embedding, passage_embedding)[0][0]
	
	return similarity.item()

def SentTransformer(sentences, model_name, datasets):
	# initiate model
	model = SentenceTransformer(model_name)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.empty_cache()
	model.to(device)

	similarity_score = []

	# parse through our 5 datasets of sentence pairs
	for dataset_num in range(len(datasets)):
		similarity_scores = []
		print("Processing:", datasets[dataset_num])
		# prase through each sentence pair of dataset
		for i in tqdm(range(len(sentences[dataset_num]))):
			# calculate their cosine similarity
			similarity_scores.append(str(FindSimilarityWithTransformer(sentences[dataset_num][i], dataset_num, model, datasets)))
		# write down our results in txt file
		with open("./sts2016-english-with-gs-v1.0/SYSTEM_OUT."+datasets[dataset_num]+".txt", 'w') as f:
			for sim in similarity_scores:
				f.write(sim)
				f.write("\n")


sentences, labels = readSentences()
similarities = SentTransformer(sentences, "all-mpnet-base-v2", datasets)
subprocess.run("./correlation-noconfidence.pl STS2016.gs.headlines.txt SYSTEM_OUT.headlines.txt", shell=True, cwd='./sts2016-english-with-gs-v1.0')

