import os
from embeddings import GloveEmbedding
import torch
from torch import FloatTensor
import json
import jsonlines


# -- Text utilities --

def first_word_from(text, word_begin):
    return text[word_begin:].split(maxsplit=1)[0].lower()

def word_from_positions(positions, fileContent):
    return fileContent[positions[0]:positions[1]]
 
def is_punctuation_or_stopword(word, stopwords):
    return (word.upos == 'PUNCT' or word.upos == 'SYM' or word.upos == 'X'
           or word.text.lower() in stopwords)
    
# -- Embedding utilities --

os.environ['HOME'] = os.path.join('..', 'GloVe')
glove = GloveEmbedding('common_crawl_48', d_emb=300, show_progress=True)

def glove_emb(word: str) -> FloatTensor:
    try:
        emb = glove.emb(word.lower())
        if emb[0] is not None:
            return FloatTensor(emb)
        else:
            return FloatTensor([0.] * 300)
    except TypeError:
        return FloatTensor([0.] * 300)

def emb_similarity(emb1: FloatTensor, emb2: FloatTensor) -> float:
    return torch.cosine_similarity(emb1, emb2, dim=0).item()

# -- File utilities --
    
def get_file_content(fileName, folder):
    filePath = os.path.join("..", folder, fileName)
    try:
        with open(filePath) as file:
            print("Loading content of", filePath, '...')
            content = file.read()
            file.close()
            print("File loaded !")
            print("-------------\n\n")
            return content
    except IOError:
        print("File", filePath, "not found.")
        return -1
        
def get_file_json(fileName, folder):
    filePath = os.path.join("..", folder, fileName)
    try:
        with open(filePath) as file:
            print("Loading content of", filePath, '...')
            contentJson = json.load(file)
            file.close()
            print("File loaded !")
            print("-------------\n\n")
            return contentJson
    except (IOError, ValueError):
        print("File", filePath, "not found or not JSON.")
        return -1

def get_file_jsonlines(fileName, folder):
    filePath = os.path.join("..", folder, fileName)
    jsonLines = []
    
    try:
        with jsonlines.open(filePath) as file:
            print("Loading content of", filePath, '...')
            for lineJson in file:
                jsonLines.append(lineJson)
            file.close()
            print("File loaded !")
            print("-------------\n\n")
            return jsonLines
    except IOError:
        print("File", filePath, "not found.")
        return -1
