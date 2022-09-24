import os
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH
import re

files = ""

nlp = English()

def mergeFiles(directory="./CUAD_v1/full_contract_txt/"):

	files = " "
	for filename in os.listdir(directory):
		with open(os.path.join(directory,  filename)) as f:
			files += f.read() + " "
	return files		


file_string = mergeFiles()

# create tokens
def tokenise(doc):
	
	#reg = '''[\.'](?=[\s\t])|[\[\("\]\)\:\?\!\:]|(?<=[\s\t])[']'''

	find_abbreviations = r'''(?<=[\s\t])[A-Z][\.a-zA-Z]*\.(?=[\s\t])''' # finds abbreviations like "St." and "P.H.D."

	find_abbreviations2 = r'''(?<=[\s\t])[a-zA-Z\d]+\.[a-zA-Z\.\d]+(?=[\s\t])''' #finds abbreviations like "i.e." or "10.19"

	find_numbers_with_commas = r'''\d+,[\d,\.]+''' # finds numbers with commas like "1,000,000.12"

	find_contractions_and_possesive = r'''[a-zA-Z]+'[a-zA-Z]+''' # finds contractions like "don't" and possesive words like "supplier's"

	special_case_dict = {}

	abbreviations = re.findall(find_abbreviations, doc)
	print(abbreviations)
	for abb in abbreviations:
		special_case_dict[abb] = [{ORTH: abb}]

	abbreviations2 = re.findall(find_abbreviations2, doc)
	print(abbreviations2)
	for abb in abbreviations2:
		special_case_dict[abb] = [{ORTH: abb}]

	comma_numbers = re.findall(find_numbers_with_commas, doc)
	print(comma_numbers)
	for num in comma_numbers:
		special_case_dict[num] = [{ORTH: num}]

	contractions = re.findall(find_contractions_and_possesive, doc)
	print(contractions)
	for con in contractions:
		special_case_dict[con] = [{ORTH: con}]

	prefix_regex = re.compile(r'''^[\[\("']''')

	suffix_regex = re.compile(r'''[\]\)\:\?\!\:\.,"']''')

	http_catcher = re.compile(r'''https:[^\s\t]+''')

	tokenizer = Tokenizer(nlp.vocab, rules = special_case_dict, prefix_search = prefix_regex.search, suffix_search=suffix_regex.search, url_match=http_catcher.search)

	doc = tokenizer(doc)

	tokens = [token.text for token in doc]
	
	return tokens


tokens = tokenise(file_string)

#print(tokens)