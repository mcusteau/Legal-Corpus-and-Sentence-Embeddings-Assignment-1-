import os

files = ""

def mergeFiles(directory="./CUAD_v1/full_contract_txt/"):

	files = ""
	for filename in os.listdir(directory):
		with open(os.path.join(directory,  filename)) as f:
			files += " " + f.read()
	return files		


file_string = mergeFiles()

print(file_string)