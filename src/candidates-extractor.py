import itertools
import copy
import sys
import json

import stanza
from stanza.pipeline.core import DownloadMethod
from nltk.tokenize import wordpunct_tokenize

from utility import *

# doc for stanza : https://stanfordnlp.github.io/stanza/data_objects#document
# Stopwords downloaded on https://www.ranks.nl/stopwords and manually modified

SIMILARITY_LIMIT = 0.7
Word = stanza.models.common.doc.Word


# -- Initializing the project --

if __name__ == '__main__' and len(sys.argv) >= 2:
    fileName = sys.argv[1]
else:
    fileName = input('Enter the name of the file to process : ')

content = get_file_content(fileName, "inputs")
if(content == -1):
    exit(0)


# -- Initializing the Stanza pipeline --

stanza.download('en', processors='tokenize, lemma, pos, depparse')
processingPipeline = stanza.Pipeline('en', processors='tokenize, lemma, pos, depparse', download_method=DownloadMethod.REUSE_RESOURCES)

doc = processingPipeline(content)
wordsFront = doc.iter_words()
wordsBack = doc.iter_words()

stopwords = set(line.strip() for line in open('stopwords.txt'))

# We have corresponding lemmas for each word. We need to initialize a window of 30 lemmas and put them in a hash
# table. Every time we have a match -> verify if we have a match inside

# -- Processing function for each next word --

def append_to_candidates(candidateList: list, startBlock: int, endBlock: int,
                         A1: Word, B1: Word, B2: Word, A2: Word):
# def append_to_candidates(candidateList, startBlock, endBlock,
#                          A1, B1, B2, A2):
    candidateList.append([[startBlock, endBlock + 25], [A1, B1, B2, A2]])
    
def append_nested_to_candidates(candidateList: list, startBlock: int, endBlock: int,
                         newPair: list, nestedCandidate: list):
    newCandidate= [[startBlock, endBlock + 25], [newPair[0]]]
    for nestedWord in nestedCandidate[1]:
        newCandidate[1].append(nestedWord)
    
    newCandidate[1].append(newPair[1])
    candidateList.append(newCandidate)

def search_nested_chiasmi(currentPair, candidateList):
    # iterate from the end to get recent matches
    for candidate in reversed(candidateList):
        if(candidate[1][-1].parent.end_char <= currentPair[0].parent.start_char):
            break
        elif(candidate[1][0].parent.start_char > currentPair[0].parent.end_char
                and candidate[1][-1].parent.end_char < currentPair[1].parent.start_char):
            return candidate
    return -1

def process_next_word(currentWord, storageTableLemma, matchTableLemma,
                      storageTableEmbedding, matchTableEmbedding, startBlock, endBlock):
    # --- Search of chiasmi through lemma correspondence

    currentId = currentWord.parent.start_char
    currentLemma = currentWord.lemma
    currentLength = len(currentWord.parent.text)

    if currentLemma in storageTableLemma:
        # we have a new match !
        # compute all pairs for the new match (A in A B ... B A)
        newPairs = list((matchingWord, currentWord) for matchingWord in storageTableLemma[currentLemma])
        
        # Let's update the storage table
        storageTableLemma[currentLemma].append(currentWord)
        
        # compute all possible pairs of all old matches (B in A B ... B A)
        oldMatches = list(itertools.combinations(matchTableLemma[oldLemma], 2) for oldLemma in matchTableLemma)
        # if oldLemma != currentLemma (to avoid A A A A ?)
        
        oldMatches = list(list(x) for x in oldMatches)

        # iterate over all pairs for the new match
        for newPair in newPairs:

            # check if the chiasmus contains more than two pairs
            nestedCandidate = search_nested_chiasmi(newPair, candidateList)
            if(nestedCandidate != -1):
                append_nested_to_candidates(
                    candidateList, startBlock, endBlock,
                    newPair, nestedCandidate)
                continue
            
            # iterate over all old matches
            for oldMatchingPairs in oldMatches:
                # iterate over all pairs from the old match to check if it is inside the new match
                for oldPair in oldMatchingPairs:
                    if(oldPair[0].parent.start_char > newPair[0].parent.start_char 
                            and oldPair[1].parent.start_char < newPair[1].parent.start_char):
                        # found a chiasmus candidate                
                        # we need, for each candidate : 
                        #   - the position in the raw text of the first character of the first word of the block
                        #   - the position in the raw text of the last character of the 5th word coming after the block
                        #       -> currently, we take the subsequent 25 characters
                        #   - the position in the block of the words forming the candidate
                        append_to_candidates(
                            candidateList, startBlock, endBlock,
                            newPair[0], oldPair[0], oldPair[1], newPair[1])

        # update the match table
        matchTableLemma[currentLemma] = copy.deepcopy(storageTableLemma[currentLemma])
    else:
        # no match, let's update the storage table
        storageTableLemma[currentLemma] = [currentWord]

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
                        # append_to_candidates(
                        #     candidateList, startBlock, endBlock,
                        #     oldWordId, oldPair1, oldPair2, currentId,
                        #     oldWordLen, oldPair1Len, oldPair2len, currentLen
                        # )
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

initRange = 30
sentenceIndex = -1
if(doc.num_words <= 30):
    initRange = doc.num_words
    
for _ in range(initRange):
    nextWord = next(wordsFront)
        
    nextWord, sentenceIndex = ignore_punctuation_and_stopwords(wordsFront, nextWord, stopwords, sentenceIndex)
    print(sentenceIndex)
    
    # if we reached the end of the file
    if nextWord == -1:
        break
    
    nextWord.sentenceIndex = sentenceIndex
    
    process_next_word(nextWord, lemmaTable, lemmaMatchTable,
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
    nextWord, sentenceIndex = ignore_punctuation_and_stopwords(wordsFront, nextWord, stopwords, sentenceIndex)
    # if we reached the end of the file
    if nextWord == -1:
        break

    nextWord.sentenceIndex = sentenceIndex
    print(sentenceIndex)
    
    oldWord, _ = ignore_punctuation_and_stopwords(wordsBack, oldWord, stopwords, None)
    oldLemma = oldWord.lemma
    startBlock = oldWord.parent.start_char

    # Processing the front of the window
    process_next_word(nextWord, lemmaTable, lemmaMatchTable,
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

print(sentenceIndex)
print("--------")
# print('-------\ncandidate list (', len(candidateList), ' candidates):')
# for candidateBlock, candidateTerms in candidateList:
#     print(word_from_positions(candidateBlock, content))
#     for term in candidateTerms:
#         print(word_from_positions(term, content), end = " ")
#     print('\n-----')

fileNameCandidates = os.path.join("..", "annotation", os.path.splitext(os.path.basename(fileName))[0] + "-annotator.jsonl")

# format imposed by the usage of Deccano
# "entities" will contain the positions of the chiasmi terms and "cats" the annotation label
# Setting default label to "False" to speed up the annotation process
candidateJson = {"text" : "", "entities" : [], "cats" : "NotAChiasmus"}
termLetters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
nbSentences = len(doc.sentences)
print(nbSentences)
# print(doc.sentences[nbSentences-1])
print()

with open(fileNameCandidates, 'w') as fileOut:
    
    for candidateBlock, candidateWords in candidateList:
        candidateJson["text"] = word_from_positions(candidateBlock, content)
        candidateJson["entities"] = []
        
        startBlock = candidateBlock[0]
        
        for letterIndex, word in enumerate(candidateWords):
            wordSpecs = []
            wordSpecs.append(word.parent.start_char - startBlock)
            wordSpecs.append(word.parent.end_char - startBlock)
            
            if letterIndex < len(candidateWords)/2:
                wordSpecs.append(termLetters[letterIndex] + "-1")
            else:
                wordSpecs.append((termLetters[len(candidateWords) - letterIndex - 1]) + "-2")
            candidateJson["entities"].append(wordSpecs)
        
        # adding metadata useful for post-annotation processings
        candidateJson["startBlock"] = startBlock
        candidateJson["endBlock"] = candidateBlock[1]
        
        # adding metadata for candidates rating
        candidateJson["dep"] = [word.deprel for word in candidateWords]
        candidateJson["words"] = []
        candidateJson["lemmas"] = []
        
        print(candidateWords[0].sentenceIndex, candidateWords[-1].sentenceIndex)
        # if(candidateWords[-1].sentenceIndex == nbSentences - 1):
        #     candidateSentences = doc.sentences[candidateWords[0].sentenceIndex:]
        #     print("hey")
        # else:
        candidateSentences = doc.sentences[candidateWords[0].sentenceIndex:
                (candidateWords[-1].sentenceIndex + 1)]
        
        appendWord = False
        for sentence in candidateSentences:
            for word in sentence.words:
                candidateJson["lemmas"].append(word.lemma)
                if(word == candidateWords[0]):
                    appendWord = True
                if(appendWord):
                    candidateJson["words"].append(word.text)
                if(word == candidateWords[-1]):
                    appendWord = False
                    break
                

        fileOut.write(json.dumps(candidateJson))
        fileOut.write("\n")
    
    fileOut.close()

print("\n---------")
print("Candidates stored in", fileNameCandidates)