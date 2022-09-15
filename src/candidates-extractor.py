import itertools
import copy
import sys

import stanza
from stanza.pipeline.core import DownloadMethod
from nltk.tokenize import wordpunct_tokenize

from utility import *

# doc for stanza : https://stanfordnlp.github.io/stanza/data_objects#document
# Stopwords downloaded on https://www.ranks.nl/stopwords and manually modified

SIMILARITY_LIMIT = 0.7


# -- Initializing the project --

if __name__ == '__main__' and len(sys.argv) >= 2:
    fileName = sys.argv[1]
else:
    fileName = input('Enter the name of the file to process : ')

content = get_file_content(fileName, "../inputs/")
if(content == -1):
    exit(0)


# -- Initializing the Stanza pipeline --

stanza.download('en', processors='tokenize, lemma, pos')
processingPipeline = stanza.Pipeline('en', processors='tokenize, lemma, pos', download_method=DownloadMethod.REUSE_RESOURCES)

doc = processingPipeline(content)
wordsFront = doc.iter_words()
wordsBack = doc.iter_words()

stopwords = set(line.strip() for line in open('stopwords.txt'))

# We have corresponding lemmas for each word. We need to initialize a window of 30 lemmas and put them in a hash
# table. Every time we have a match -> verify if we have a match inside

# -- Processing function for each next word --


def append_to_candidates(candidateList: list, startBlock: int, endBlock: int,
                         A1: str, B1: str, B2: str, A2: str,
                         A1_len: int, B1_len: int, B2_len: int, A2_len: int):
    candidateList.append([
        [startBlock, endBlock + 25], [[A1, A1 + A1_len],
                                      [B1, B1 + B1_len],
                                      [B2, B2 + B2_len],
                                      [A2, A2 + A2_len]]
    ])



def process_next_word(currentWord, currentId, storageTableLemma, matchTableLemma,
                      storageTableEmbedding, matchTableEmbedding, startBlock, endBlock):
    # --- Search of chiasmi through lemma correspondence

    currentTerm = currentWord.lemma
    if currentTerm in storageTableLemma:
        # we have a match ! Let's update the storage table
        storageTableLemma[currentTerm].append(currentId)

        # compute all possible pairs for the new match (A in A B B A)
        newPairs = [currentTerm, list(itertools.combinations(storageTableLemma[currentTerm], 2))]

        # compute all possible pairs of old matches (B in A B B A)
        oldMatches = [(oldTerm, list(itertools.combinations(matchTableLemma[oldTerm], 2))) for oldTerm in matchTableLemma]
        
        # iterate over all pairs for the new match
        for newPair in newPairs[1]:
          # iterate over all old matches
          for oldMatch in oldMatches:
            oldTerm = oldMatch[0]
            # iterate over all pairs from the old match to check if it is inside the new match
            for oldPair in oldMatch[1]:
                if oldPair[0] > newPair[0] and oldPair[1] < newPair[1]:
                    # found a chiasmus candidate                
                    # we need, for each candidate : 
                    #   - the position in the raw text of the first character of the first word of the block
                    #   - the position in the raw text of the last character of the 5th word coming after the block
                    #       -> currently, we take the subsequent 25 characters
                    #   - the position in the block of the words forming the candidate
                    append_to_candidates(
                        candidateList, startBlock, endBlock,
                        newPair[0], oldPair[0], oldPair[1], newPair[1],
                        len(currentTerm), len(oldTerm), len(oldTerm), len(currentTerm)
                    )

        # update the match table
        matchTableLemma[currentTerm] = copy.deepcopy(storageTableLemma[currentTerm])
    else:
        # no match, let's update the storage table
        storageTableLemma[currentTerm] = [currentId]

    return  # Skip the embedding part for now
    # --- Search of chiasmi through embedding (semantic) similarity (!!! YET UNTESTED !!!)

    currentEmb = glove_emb(currentWord.text)
    currentLen = len(currentWord.text)

    # Search for possible matches
    for oldWordId, (emb, oldWordLen) in storageTableEmbedding.items():
        similarity = emb_similarity(currentEmb, emb)
        if similarity > SIMILARITY_LIMIT or similarity < -SIMILARITY_LIMIT:
            # We have a match! Searching for possible second pairs of matching words
            for oldPair1, matchedWords in matchTableEmbedding.items():
                # We need the second pairs to be contained withing the first pair,
                # i.e. its first word to be AFTER the word we orginally matched
                if oldWordId < oldPair1:
                    oldPair1Len = storageTableEmbedding[oldPair1][1]
                    for oldPair2 in matchedWords:
                        oldPair2len = storageTableEmbedding[oldPair2][1]
                        append_to_candidates(
                            candidateList, startBlock, endBlock,
                            oldWordId, oldPair1, oldPair2, currentId,
                            oldWordLen, oldPair1Len, oldPair2len, currentLen
                        )
            # Updating the embedding match table
            if oldWordId in matchTableEmbedding:
                matchTableEmbedding[oldWordId].append(currentId)
            else:
                matchTableEmbedding[oldWordId] = [currentId]

    # Updating the embedding storage table
    storageTableEmbedding[currentId] = (currentEmb, currentLen)



# -- Creating the general variables --

candidateList = []
lemmaTable = {}
lemmaMatchTable = {}
embeddingTable = {}
embeddingMatchTable = {}

# -- Initializing the sliding window over the first 30 characters --

for _ in range(30):
    nextWord = next(wordsFront)
    nextWord = ignore_punctuation_and_stopwords(wordsFront, nextWord, stopwords)
    
    # if we reached the end of the file
    if nextWord == -1:
        break
        
    process_next_word(nextWord, nextWord.parent.start_char, lemmaTable, lemmaMatchTable,
                      embeddingTable, embeddingMatchTable, 0, nextWord.parent.end_char)

# -- Main part : make the window slide using wordsFront and wordsBack --
#    (Same algorithm but delete info relevant to wordsBack when moving forward)

# candidateList is a list containing lists of the form
#   [[info regarding the block], [info regarding the chiasm in itself]]
# where the first list contains :
#   [the position of the first, the position of the last character of the block]
# and the second list contains :
#   [[startFirstTerm, endFirstTerm], [startSecondTerm, endSecondTerm], ...]
#   each "Term" being a word that is part of the chiasmus candidate

# -- Main part : make the window slide using wordsFront and wordsBack --

# foreach stops when wordsFront ends
for nextWord, oldWord in zip(wordsFront, wordsBack):
    # If we are currently processing a "punctuation or stop word", then we ignore it
    nextWord = ignore_punctuation_and_stopwords(wordsFront, nextWord, stopwords)
    # if we reached the end of the file
    if nextWord == -1:
        break

    oldWord = ignore_punctuation_and_stopwords(wordsBack, oldWord, stopwords)
    oldLemma = oldWord.lemma
    startBlock = oldWord.parent.start_char

    # Processing the front of the window
    process_next_word(nextWord, nextWord.parent.start_char, lemmaTable, lemmaMatchTable,
                      embeddingTable, embeddingMatchTable, startBlock, nextWord.parent.end_char)
    
    # handle the rear of the window
    # Delete the word exiting the sliding window from lemmaTable
    if len(lemmaTable[oldLemma]) <= 1:
        del lemmaTable[oldLemma]
    else:
        del lemmaTable[oldLemma][0]
    # Updating matchTable if necessary after this deletion
    if oldLemma in lemmaMatchTable:
        # delete when only one occurrence is left - not a match anymore
        if len(lemmaMatchTable[oldLemma]) <= 2:
            del lemmaMatchTable[oldLemma]
        else:
            del lemmaMatchTable[oldLemma][0]

# [TBD] : process the last 30 words of the file !


print('-------\ncandidate list (', len(candidateList), ' candidates):')
for candidateBlock, candidateTerms in candidateList:
    print(word_from_positions(candidateBlock, content))
    for term in candidateTerms:
        print(word_from_positions(term, content), end = " ")
    print('\n-----')

fileNameCandidates = os.path.join("..", "outputs", os.path.splitext(os.path.basename(fileName))[0] + "-candidates.txt")
with open(fileNameCandidates, 'w') as fileOut:
    for candidateBlock, candidateTerms in candidateList:
        # the newline character separates candidates
        fileOut.write(word_from_positions(candidateBlock, content).replace("\n", ""))
        fileOut.write('\n')
        fileOut.write(str(candidateBlock[0]) + ' ' + str(candidateBlock[1]))
        for termPair in candidateTerms:
            for term in termPair:
                term = term - candidateBlock[0]
                fileOut.write(' ' + str(term))
        fileOut.write('\n')
    fileOut.close()