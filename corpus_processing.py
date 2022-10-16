import os
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH
import re
import io



files = ""
nlp = English()

def mergeFiles(directory="./CUAD_v1/full_contract_txt/"):
	""" Reads all the documents and returns a string of the files
	"""
	files = " "
	for filename in os.listdir(directory):
		with open(os.path.join(directory,  filename)) as f:
			files += f.read() + " "
	return files		

file_string = mergeFiles()


def lowercase(tokens):
	"""Returns the lowercased list of tokens
	"""
	n=len(tokens)
	for i in range(n):
		tokens[i]=tokens[i].lower()
	return tokens


def word_count(tokens):
	"""Creates a dictionary with the tokens and their frequencies in the corpus.
		The tokens need to be lowercased before calling this function for a higher accuracy in the results.
	"""
	token_dict={}
	for token in tokens:
		if (token not in token_dict.keys()):
			token_dict.update({token: 1})
		else:       
			token_dict[token] += 1
	return token_dict 


def sort_frequencies(token_dict):
	""" Returns the sorted dictionary of the tokens in descending order
	"""
	token_dict_sorted= {k: v for k, v in sorted(token_dict.items(), key=lambda  pair: pair[1], reverse=True)}
	return token_dict_sorted


def appeard_n_times(token_dict,n):	
	"""Returns the # of the tokens that appeard n times
	"""
	count_n=0
	for once in token_dict.values():
		if once==n:
			count_n+=1
	return count_n


def print_tokens(tokens,filename):
	""" Prints the list of the tokens to a new txt file
	"""
	# prints to a new line
	with open(filename, 'w') as file: 
	    for token in tokens: 
	    	file.write('%s\n' % token)
	file.close()


def print_freq(token_dict,filename):
	""" Prints the dictinary of the tokens/bigrams to a new txt file
	"""
	# prints to a new line
	count=0
	with open(filename, 'w') as file: 	
	    for key, value in token_dict.items(): 
	        file.write('%s:%s\n' % (key, value))
	        count+=1
	file.close()


# create tokens
def tokenise(doc, raw=False):
	""" This is our main tokeniser function.
	"""

	# takes the tokens as they are	
	if (raw):
		# build our tokenizer
		tokenizer = Tokenizer(nlp.vocab)

	# extracts the words 	
	else:
		with open('contractions.txt', 'r') as f:
			contractions = f.read().split(",")

		find_abbreviations = r'''(?<=[\s\t\n])[A-Z][\.a-zA-Zéèàëôùç]*\.(?=[\s\t\n])''' # finds abbreviations like "St." and "P.H.D." by looking for words that start with capital letter and end with dot as those are probably abbreviations

		find_abbreviations2 = r'''(?<=[\s\t\n])[a-zA-Zéèàëôùç\d]+\.[a-zA-Zéèàëôùç\.\d]+(?=[\s\t\n])''' #finds abbreviations like "i.e." or numbers like "10.19" by looking for words that contain at least one dot in the middle of them

		find_numbers_with_commas = r'''\d+,[\d,\.]+''' # finds numbers with commas like "1,000,000.12"


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

		for con in contractions:
			#print(con)
			special_case_dict[con] = [{ORTH: con}]
			special_case_dict[con.capitalize()] = [{ORTH: con.capitalize()}]
			special_case_dict[con.upper()] = [{ORTH: con.upper()}]


		# Note that these next regex expressions will not impact our special words we added to dictioanry

		prefix_regex = re.compile(r'''^[{\[\("']''') # seperates words from punctuation marks: [ ( " ' that are located at the beginning of the word

		suffix_regex = re.compile(r'''[\]\)\:\?\!\:\.,"';}&\-]$''') # seperates words from punctuation marks: ] ) : ? ! . " , ' ; that are located at the end of the word

		infix_regex = re.compile(r'''(?<=[a-zA-Zéèàëôùç\[\]\(\)\*.\$])[\-'\\\:\(\/&;](?=[a-zA-Zéèàëôùç\[\]\(\)\*.\$])|(?<=[\d])['](?=[a-zA-Zéèàëôùç])|(?<=[a-zA-Z\[\]\(\)\*.])'-(?=[a-zA-Z\[\]\(\)\*.])''') # splits words in two when seeing - ' \ : ( ) / & ;  between two letters

		http_catcher = re.compile(r'''https:[^\s\t\n]+''') # catches URLs that start with https:

		# build our tokenizer
		tokenizer = Tokenizer(nlp.vocab, rules = special_case_dict, prefix_search = prefix_regex.search, infix_finditer= infix_regex.finditer, suffix_search=suffix_regex.search, url_match=http_catcher.search)

	# tokenize document
	doc = tokenizer(doc)
	tokens = [token.text for token in doc]
		
	return tokens

def remove_punctuations(tokens):
	""" Removes the empty tokens or the tokens that consists of punctuation or symbols. Returns the new enhanced list of tokens.
	"""
	# list of all the symbols and punctuations   
	punctuation = '''!"#$%&\'-()...***...…***…*****+,.///:;<=>?@[·[*\\]^_`—{|}·•\'\'●§§§~...***...]%\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\s\s\s\s\s\s\s\s\s\r\r\r\r\r\r\r\r'''
	tokens_nopunc=[]
	for token in tokens:
		if (token not in punctuation) and (token.strip()!=""):
			tokens_nopunc.append(token.strip())
	return tokens_nopunc


def remove_stopwords(tokens):
	""" Creates a list with all the stopwords from the given text file.
		Removes the tokens that consists of stopwords. 
		Returns the new enhanced list of tokens.
	"""
	with open('stopwords.txt', 'r') as f:
		stop_words = [line.strip() for line in f]

	tokens_nostop=[]
	for token in tokens:
		
		if token not in stop_words:
			tokens_nostop.append(token)
			
	return tokens_nostop


def compute_bigrams(tokens):
	""" Returns a list of tuples created with every 2 consecutive words in the list of tokens.
	"""
	bigrams=[]
	n=len(tokens)
	# print("Bigrams are:\n")
	for i in range(n-1):
		bigram=(tokens[i],tokens[i+1])
		bigrams.append(bigram)
	# print(bigrams)
	return bigrams


def process_corpus(file_string):
	"""
	There are 3 types of results while processing the corpus :
	1) Raw tokens
	2) punctuations removed
	3) punctuations and stopwords removed 
	"""


	####################
	# process raw
	####################
	raw=True
	lower=True
	tokens = tokenise(file_string,raw) # tokenising the document

	if (lower):
		tokens=lowercase(tokens)	# lowercasing the tokens

	token_dict=word_count(tokens)	# dictionnary with word frequencies
	sorted_freq=sort_frequencies(token_dict)

	print_tokens(tokens,"outputs.txt")
	print_freq(sorted_freq,"tokens.txt")

	title="Raw tokens"
	islowercase="lower case before processing: "+str(lower)
	n_unique_tokens=len(token_dict)
	n_tokens_in_the_corpus="# of tokens in the corpus: "+str(len(tokens))
	n_unique_tokens_in_the_corpus="# unique tokens in the corpus: "+str(n_unique_tokens)
	type_token_ratio="# of types (unique tokens) / token ratio: "+str(n_unique_tokens/len(tokens))
	n_tokens_appeared_once=str(appeard_n_times(sorted_freq,1))+" tokens appeared only once in the corpus."

	results = io.open("Report_results.txt", 'w')
	results.write(f'{title}\n{islowercase}\n{n_tokens_in_the_corpus}\n{n_unique_tokens_in_the_corpus}\n{type_token_ratio}\n{n_tokens_appeared_once}\n')
	results.close()


	####################
	# no punctuation
	####################
	# raw=False
	lower=True
	tokens = tokenise(file_string) # tokenising the document

	if (lower):
		tokens=lowercase(tokens)

	tokens=remove_punctuations(tokens) 	# remove punctutation
	token_dict=word_count(tokens)
	sorted_freq=sort_frequencies(token_dict)

	print_tokens(tokens,"outputs_enhanced_no_punct.txt")
	print_freq(sorted_freq,"tokens_enhanced_no_punct.txt")

	title="\n\nPunctuation excluded"
	islowercase="lower case before processing: "+str(lower)
	n_unique_tokens=len(token_dict)
	n_tokens_in_the_corpus="# of tokens in the corpus: "+str(len(tokens))
	n_unique_tokens_in_the_corpus="# unique tokens in the corpus: "+str(n_unique_tokens)
	type_token_ratio="(\"lexical diversity\", # of types (unique tokens) / token ratio : "+str(n_unique_tokens/len(tokens))
	n_tokens_appeared_once=str(appeard_n_times(sorted_freq,1))+" tokens appeared only once in the corpus."

	results = io.open("Report_results.txt", 'a')
	results.write(f'{title}\n{islowercase}\n{n_tokens_in_the_corpus}\n{n_unique_tokens_in_the_corpus}\n{type_token_ratio}\n{n_tokens_appeared_once}\n')
	results.close()


	####################
	# no punctuation, no stopwords
	####################
	# raw=False
	lower=True
	tokens = tokenise(file_string) # tokenising the document

	if (lower):
		tokens=lowercase(tokens)

	tokens=remove_punctuations(tokens) # remove punctutation
	tokens=remove_stopwords(tokens)	# remove stopwords
	token_dict=word_count(tokens)
	sorted_freq=sort_frequencies(token_dict)

	bigrams=compute_bigrams(tokens) # create bigrams
	bigram_dict=word_count(bigrams)
	sorted_bigram_dict=sort_frequencies(bigram_dict)

	print_tokens(tokens,"outputs_enhanced_no_punct_stopw.txt")
	print_freq(sorted_freq,"tokens_enhanced_no_punct_stopw.txt")
	print_freq(sorted_bigram_dict,"bigrams.txt")

	title="\n\nPunctuation AND Stopwords excluded"
	islowercase="lower case before processing: "+str(lower)
	n_unique_tokens=len(token_dict)
	n_tokens_in_the_corpus="# of tokens in the corpus: "+str(len(tokens))
	n_unique_tokens_in_the_corpus="# unique tokens in the corpus: "+str(n_unique_tokens)
	type_token_ratio="\"lexical density\", # of types (unique tokens) / token ratio: "+str(n_unique_tokens/len(tokens))
	n_tokens_appeared_once=str(appeard_n_times(sorted_freq,1))+" tokens appeared only once in the corpus."

	results = io.open("Report_results.txt", 'a')
	results.write(f'{title}\n{islowercase}\n{n_tokens_in_the_corpus}\n{n_unique_tokens_in_the_corpus}\n{type_token_ratio}\n{n_tokens_appeared_once}\n')
	results.close()


process_corpus(file_string)


