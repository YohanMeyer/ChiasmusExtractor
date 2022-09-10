import stanza
from stanza.pipeline.core import DownloadMethod
import itertools
import copy
from nltk.tokenize import wordpunct_tokenize
# doc for stanza : https://stanfordnlp.github.io/stanza/data_objects#document
# Stopwords downloaded on https://www.ranks.nl/stopwords and manually modified

# -- Utility functions --
def first_word_from(text, word_begin):
    return text[word_begin:].split(maxsplit=1)[0].lower()

def ignore_punctuation_and_stopwords(wordIterator, nextWord, stopwords):
    while(nextWord.upos == 'PUNCT' or nextWord.upos == 'SYM' or nextWord.upos == 'X' 
            or nextWord.text.lower() in stopwords):
        try:
            nextWord = next(wordIterator)
        except StopIteration:
            return -1
    
    return nextWord

def word_from_positions(positions, fileContent):
    return fileContent[positions[0]:positions[1]]

# -- Initializing the project --

# fileName = input('Enter the name of the file to process : ')
# fileName = 'small-chiasmi.txt'
fileName = 'chiasmi.txt'
# fileName = 'test2.txt'

with open(fileName) as file:
    content = file.read()
    file.close()


# -- Initializing the Stanza pipeline for further processing --
stanza.download('en', processors='tokenize, lemma, pos')
processingPipeline = stanza.Pipeline('en', processors='tokenize, lemma, pos', download_method=DownloadMethod.REUSE_RESOURCES)

doc = processingPipeline(content)
wordsFront = doc.iter_words()
wordsBack = doc.iter_words()

stopwords = set(line.strip() for line in open('stopwords.txt'))

# we have corresponding lemmas for each word. We need to initialize a window of 30 lemmas and put them in a hash
# table. Every time we have a match -> verify if we have a match inside

# -- Processing function for each next word --

def process_next_word(currentTerm, currentId, storageTable, matchTable, startBlock, endBlock):
    if currentTerm in storageTable:
        # we have a match ! Let's update the storage table
        storageTable[currentTerm].append(currentId)

        # compute all possible pairs for the new match (A in A B B A)
        newPairs = [currentTerm, list(itertools.combinations(storageTable[currentTerm], 2))]

        # compute all possible pairs of old matches (B in A B B A)
        oldMatches = [(oldTerm, list(itertools.combinations(matchTable[oldTerm], 2))) for oldTerm in matchTable]
        
        # iterate over all pairs for the new match
        for newPair in newPairs[1]:
          # iterate over all old matches
          for oldMatch in oldMatches:
            oldTerm = oldMatch[0]
            # iterate over all pairs from the old match to check if it is inside the new match
            for oldPair in oldMatch[1]:
                if (oldPair[0] > newPair[0] and oldPair[1] < newPair[1]):
                    # found a chiasmus candidate                
                    # we need, for each candidate : 
                    #   - the position in the raw text of the first character of the first word of the block
                    #   - the position in the raw text of the last character of the 5th word coming after the block
                    #       -> currently, we take the subsequent 25 characters
                    #   - the position in the block of the words forming the candidate
                    candidateList.append([[startBlock, endBlock + 25], [[newPair[0], newPair[0] + len(currentTerm)], [oldPair[0], oldPair[0] + len(oldTerm)], [oldPair[1], oldPair[1] + len(oldTerm)], [newPair[1], newPair[1] + len(currentTerm)]]])

        # update the match table
        matchTable[currentTerm] = copy.deepcopy(storageTable[currentTerm])
    else:
        # no match, let's update the storage table
        storageTable[currentTerm] = [currentId]

# -- Creating the general variables --

# candidateList is a list containing lists of the form
#   [[info regarding the block], [info regarding the chiasm in itself]]
# where the first list contains :
#   [the position of the first, the position of the last character of the block]
# and the second list contains :
#   [[startFirstTerm, endFirstTerm], [startSecondTerm, endSecondTerm], ...]
#   each "Term" being a word that is part of the chiasmus candidate

candidateList = []
lemmaTable = {}
matchTable = {}

# -- Main part : make the window slide using wordsFront and wordsBack --

loop = 0
# foreach stops when wordsFront ends
for nextWord, oldWord in zip(wordsFront, wordsBack):    
    # handle the rear of the window
    if loop < 30:
        # initializing the window to have 30 words
        oldLemma = ''
        startBlock = 0
    else:
        # If we are currently processing a "punctuation or stop word", then we ignore it
        oldWord = ignore_punctuation_and_stopwords(wordsBack, oldWord, stopwords)
        oldLemma = oldWord.lemma
        startBlock = oldWord.parent.start_char
        
        # Delete the word exiting the sliding window from lemmaTable
        # not leaving an empty entry in the table
        if(len(lemmaTable[oldLemma]) <= 1):
            del lemmaTable[oldLemma]
        else:
            del lemmaTable[oldLemma][0]
        # Updating matchTable if necessary after this deletion
        if oldLemma in matchTable:
            # delete when only one occurrence is left - not a match anymore
            if(len(matchTable[oldLemma]) <= 2):
                del matchTable[oldLemma]
            else:
                del matchTable[oldLemma][0]
    loop += 1
    
    # handle the front of the window
    # If we are currently processing a "punctuation or stop word", then we ignore it
    nextWord = ignore_punctuation_and_stopwords(wordsFront, nextWord, stopwords)
    # if we reached the end of the file
    if(nextWord == -1):
        break

    # Processing the new word
    process_next_word(nextWord.lemma, nextWord.parent.start_char, lemmaTable, matchTable, startBlock, nextWord.parent.end_char)

print('-------\ncandidate list (', len(candidateList), ' candidates):')
for candidateBlock, candidateTerms in candidateList:
    print(word_from_positions(candidateBlock, content))
    print(word_from_positions(candidateTerms[0], content), word_from_positions(candidateTerms[1], content), word_from_positions(candidateTerms[2], content), word_from_positions(candidateTerms[3], content))
    print('-----')

