
import os
import spacy
# from collections import Counter
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH
import re
# import time
import io



nlp = English()

doc ="this\nsi; a! test' as(df (asdf) [1000,332,324.12] Jame's Jame(s New-York asdf:asdf don't aren't ar\\f asdf- {2012\\10\\10}"

with open('contractions.txt', 'r') as f:
		contractions = f.read().split(",")

find_abbreviations = r'''(?<=[\s\t\n])[A-Z][\.a-zA-Z]*\.(?=[\s\t\n])''' # finds abbreviations like "St." and "P.H.D." by looking for words that start with capital letter and end with dot as those are probably abbreviations

find_abbreviations2 = r'''(?<=[\s\t\n])[a-zA-Z\d]+\.[a-zA-Z\.\d]+(?=[\s\t\n])''' #finds abbreviations like "i.e." or numbers like "10.19" by looking for words that contain at least one dot in the middle of them

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

suffix_regex = re.compile(r'''[\]\)\:\?\!\:\.,"';}]$''') # seperates words from punctuation marks: ] ) : ? ! . " , ' ; that are located at the end of the word

infix_regex = re.compile(r'''(?<=[a-zA-Z])[\-'\\\:\(](?=[a-zA-Z])''') # splits words in two when seeing - ' \ : ( between two letters

http_catcher = re.compile(r'''https:[^\s\t\n]+''') # catches URLs that start with https:

# build our tokenizer
tokenizer = Tokenizer(nlp.vocab, rules = special_case_dict, prefix_search = prefix_regex.search, infix_finditer= infix_regex.finditer, suffix_search=suffix_regex.search, url_match=http_catcher.search)

# tokenize document
doc = tokenizer(doc)

tokens = [token.text for token in doc]
print(tokens)