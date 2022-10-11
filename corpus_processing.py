import os
import spacy
# from collections import Counter
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH
import re
# import time
import io


files = ""

nlp = English()

def mergeFiles(directory="./CUAD_v1/full_contract_txt/"):

	files = " "
	for filename in os.listdir(directory):
		with open(os.path.join(directory,  filename)) as f:
			files += f.read() + " "
	return files		


file_string = mergeFiles()


# def merge2Files(nfile,directory="./CUAD_v1/full_contract_txt/"):

# 	files = " "
# 	count=0
# 	for filename in os.listdir(directory):
# 		while count<nfile:
# 			with open(os.path.join(directory,  filename)) as f:
# 				files += f.read() + " "
# 				count+=1
# 	return files	

# file_string = merge2Files(2)

def tokenise1(doc):
	# build our tokenizer
	tokenizer = Tokenizer(nlp.vocab)
	# tokenize document
	doc = tokenizer(doc)
	tokens = [token.text for token in doc]
	return tokens

def get_unique_tokens(tokens):
	u_tokens = []
	visited = set()
	for word in tokens:
		lword=word.lower()
		if lword not in visited:
			u_tokens.append(lword)
		visited.add(lword)
	return u_tokens
	# now out has "unique" tokens

def word_count(tokens):
	# word_freq = Counter(tokens)
	# common_words = word_freq.most_common(5)
	# print (common_words)
	# { _key : _value(_key) for tokens in _container }

	u_tokens=get_unique_tokens(tokens)
	token_dict=dict.fromkeys(u_tokens,0)

	for token in tokens:
		ltoken=token.lower()
		token_dict[ltoken]+=1

	return token_dict 

def sort_frequencies(token_dict):
	token_dict_sorted= {k: v for k, v in sorted(token_dict.items(), key=lambda  pair: pair[1], reverse=True)}

	return token_dict_sorted


def print_results(token_dict):
	# no line separation
	# file = io.open('Results.txt', 'w')
	# file.write(f'{token_dict} run1\n')

	# prints to a new line
	with open("tokens.txt", 'w') as file: 
	    for key, value in token_dict.items(): 
	        file.write('%s:%s\n' % (key, value))

	file.close()



# create tokens
def tokenise(doc):
	
	#reg = '''[\.'](?=[\s\t])|[\[\("\]\)\:\?\!\:]|(?<=[\s\t])[']'''

	find_abbreviations = r'''(?<=[\s\t])[A-Z][\.a-zA-Z]*\.(?=[\s\t])''' # finds abbreviations like "St." and "P.H.D." by looking for words that start with capital letter and end with dot as those are probably abbreviations

	find_abbreviations2 = r'''(?<=[\s\t])[a-zA-Z\d]+\.[a-zA-Z\.\d]+(?=[\s\t])''' #finds abbreviations like "i.e." or numbers like "10.19" by looking for words that contain at least one dot in the middle of them

	find_numbers_with_commas = r'''\d+,[\d,\.]+''' # finds numbers with commas like "1,000,000.12"

	find_contractions_and_possesive = r'''[a-zA-Z]+'[a-zA-Z]+''' # finds contractions like "don't" and possesive words like "supplier's"    

	#find_plurial_possisive = r'''(?<=[\s\t])[^\'][a-zA-Z]+\'(?=[\s\t])''' # finds contractions like James' but looking for words that end however this is probematic

	# dictionary of special words that we dont want edited during tokenization
	special_case_dict = {}

	abbreviations = re.findall(find_abbreviations, doc)
	# print(abbreviations)
	for abb in abbreviations:
		special_case_dict[abb] = [{ORTH: abb}]

	abbreviations2 = re.findall(find_abbreviations2, doc)
	# print(abbreviations2)
	for abb in abbreviations2:
		special_case_dict[abb] = [{ORTH: abb}]

	comma_numbers = re.findall(find_numbers_with_commas, doc)
	# print(comma_numbers)
	for num in comma_numbers:
		special_case_dict[num] = [{ORTH: num}]

	contractions = re.findall(find_contractions_and_possesive, doc)
	# print(contractions)
	for con in contractions:
		special_case_dict[con] = [{ORTH: con}]

	# Note that these next regex expressions will not impact our special words we added to dictioanry

	prefix_regex = re.compile(r'''^[\[\("']''') # seperates words from punctuation marks: [ ( " '     

	suffix_regex = re.compile(r'''[\]\)\:\?\!\:\.,"']''') # seperates words from punctuation marks: ] ) : ? ! . " , '     

	http_catcher = re.compile(r'''https:[^\s\t]+''') # catches URLs that start with https:

	# build our tokenizer
	tokenizer = Tokenizer(nlp.vocab, rules = special_case_dict, prefix_search = prefix_regex.search, suffix_search=suffix_regex.search, url_match=http_catcher.search)

	# tokenize document
	doc = tokenizer(doc)

	tokens = [token.text for token in doc]
	
	return tokens

def remove_stopwords(tokens):
	with open('stopwords.txt', 'r') as f:
		stop_words = [line.strip() for line in f]

	tokens_nostop=[]
	for token in tokens:
		if token not in stop_words:
			tokens_nostop.append(token.lower())

	return tokens_nostop


def compute_bigrams(tokens):
	# bigrams = [b for l in tokens for b in zip(l,l)]
	bigrams=[]
	n=len(tokens)
	print("Bigrams are:\n")
	for i in range(n-1):
		bigram=(tokens[i],tokens[i+1])
		bigrams.append(bigram)
	# print(bigrams)
	return bigrams

	# bigrams_without_stopwords = [(a,b) for a,b in nltk.bigrams(brown.words(categories="romance")) if a not in stopwords.words('english') and b not in stopwords.words('english')]
	# bigrams_without_stopwords_fd = nltk.FreqDist(bigrams_without_stopwords)
	# print(bigrams_without_stopwords_fd.most_common(50))

#FIX THIS
def bigram_frequency(bigrams):

	# u_tokens=get_unique_tokens(tokens)
	# bigram_dict=dict.fromkeys(bigrams,0)
	bigram_dict={}
	for bigram in bigrams:
		# bigram_dict[bigram]+=1
		if (bigram not in bigram_dict.keys()):
			bigram_dict.update({bigram: 1})
		else:       
			bigram_dict[bigram] += 1

	return bigram_dict 

def sort_bigram_frequencies(bigram_dict):
	bigram_dict_sorted= {k: v for k, v in sorted(bigram_dict.items(), key=lambda  pair: pair[1], reverse=True)}

	return bigram_dict_sorted

def process_corpus(file_string, raw=False, exclude_stopwords=True):
# raw=True then tokens are not modified
# exclude_stopwords=True then punctuations and stopwords are removed else only punctuations are removed

	# Part 1
	# b)
	# tokens = tokenise1(file_string)

	if (raw):
		tokens = tokenise1(file_string)
	else:
		tokens = tokenise(file_string)

	if (exclude_stopwords):
		tokens=remove_stopwords(tokens)

	unique_tokens=get_unique_tokens(tokens)

	n_tokens_in_the_corpus="# of tokens in the corpus: "+str(len(tokens))
	n_unique_tokens_in_the_corpus="# unique tokens in the corpus: "+str(len(unique_tokens))
	type_token_ratio="# of types (unique tokens) / token ratio: "+str(len(tokens)/len(unique_tokens))

	# start_time = time.time()

	token_dict=word_count(tokens)
	sorted_freq=sort_frequencies(token_dict)
	print_results(sorted_freq)
	# print("--- %s seconds ---" % (time.time() - start_time))

	count_one=0
	for once in sorted_freq.values():
		if once==1:
			count_one+=1
	n_tokens_appeared_once=str(count_one)+" tokens appeared only once in the corpus."
	
	bigrams=compute_bigrams(tokens)
	bigram_dict=bigram_frequency(bigrams)
	# print("Bigram frequencies are:\n",bigram_dict)
	sorted_bigram_dict=sort_frequencies(bigram_dict)
	print("\n\n\n\nSORTED bigram frequencies are:\n",sorted_bigram_dict)

	if (raw):
		filename='Report_results_raw.txt'
		results = io.open(filename, 'w')
		results.write(f'{n_tokens_in_the_corpus}\n{n_unique_tokens_in_the_corpus}\n{type_token_ratio}\n{n_tokens_appeared_once}\n')
		results.close()
	else:
		if (exclude_stopwords):
			filename='Report_results_no_stopwords.txt'
			results = io.open(filename, 'w')
			results.write(f'{n_tokens_in_the_corpus}\n{n_unique_tokens_in_the_corpus}\n{type_token_ratio}\n{n_tokens_appeared_once}\n')
			results.close()
		else:
			filename='Report_results_no_punctuation.txt'
			results = io.open(filename, 'w')
			results.write(f'{n_tokens_in_the_corpus}\n{n_unique_tokens_in_the_corpus}\n{type_token_ratio}\n{n_tokens_appeared_once}\n')
			results.close()

	

# process_corpus(file_string,True,False)
# process_corpus(file_string,exclude_stopwords=False)
process_corpus(file_string)




