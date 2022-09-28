import os
from embeddings import GloveEmbedding
import torch
from torch import FloatTensor


# -- Text utilities --


def first_word_from(text, word_begin):
    return text[word_begin:].split(maxsplit=1)[0].lower()


def word_from_positions(positions, fileContent):
    return fileContent[positions[0]:positions[1]]


def ignore_punctuation_and_stopwords(wordIterator, nextWord, stopwords):
    while (nextWord.upos == 'PUNCT' or nextWord.upos == 'SYM' or nextWord.upos == 'X'
           or nextWord.text.lower() in stopwords):
        try:
            nextWord = next(wordIterator)
        except StopIteration:
            return -1

    return nextWord
    
# -- Embedding utilities --


os.environ['HOME'] = os.path.join('..', 'GloVe')
glove = GloveEmbedding('common_crawl_48', d_emb=300, show_progress=True)


def glove_emb(word: str) -> FloatTensor:
    try:
        return FloatTensor(glove.emb(word))
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