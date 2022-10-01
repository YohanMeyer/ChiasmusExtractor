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

EMBEDDING_SIMILARITY_LIMIT = 0.8

# -- Creating the general variables --

# candidateList is a list containing lists of the form
#   [[info regarding the block], [info regarding the chiasm in itself]]
# where the first list contains :
#   [the position of the first, the position of the last character of the block]
# and the second list contains :
#   [[startFirstTerm, endFirstTerm], [startSecondTerm, endSecondTerm], ...]
#   each "Term" being a word that is part of the chiasmus candidate

candidateList = []
storageTableLemma = {}
matchTableLemma = {}
storageTableEmbedding = {}
matchTableEmbedding = {}
lengthTable = {}


# -- Function adding a candidate to the current list --

def append_to_candidates(startBlock: int, endBlock: int,
                         A1: int, B1: int, B2: int, A2: int,
                         A1_len: int, B1_len: int, B2_len: int, A2_len: int):
    candidateList.append([
        [startBlock, endBlock + 25], [[A1, A1 + A1_len],
                                      [B1, B1 + B1_len],
                                      [B2, B2 + B2_len],
                                      [A2, A2 + A2_len]]
    ])

# -- Function adding a nested candidate (A B C ... C B A) to the current list --

def append_nested_to_candidates(startBlock: int, endBlock: int,
                                A1: int, A2: int, A1_len: int, A2_len: int,
                                nestedCandidate: list, alreadyDetectedCandidates: list):
    newCandidate = [[startBlock, endBlock + 25], [[A1, A1 + A1_len]]]
    for nestedTerm in nestedCandidate[1]:
        newCandidate[1].append(nestedTerm)
    
    newCandidate[1].append([A2, A2 + A2_len])
    candidateList.append(newCandidate)
    alreadyDetectedCandidates.append(tuple([A1] + [nestedTerm[0] for nestedTerm in nestedCandidate[1]] + [A2]))

# -- Function to find nested chiasmi (A B C ... C B A) in the current state --

def search_nested_chiasmi(currentMatch, candidateList):
    nestedChiasmi = []
    for candidate in reversed(candidateList):
        if(candidate[1][-1][1] <= currentMatch[0]):
            break
        elif(candidate[1][0][0] > currentMatch[0] and candidate[1][-1][1] < currentMatch[1]):
            nestedChiasmi.append(candidate)
    return nestedChiasmi

# -- Processing function for each next word --

def process_next_word(currentWord, currentId, startBlock, endBlock):
    alreadyDetectedCandidates = []

    # --- Search of chiasmi through lemma correspondence

    currentTerm = currentWord.lemma
    currentLength = len(currentWord.parent.text)

    if currentTerm in storageTableLemma:
        # we have a new match ! Let's update the storage table
        storageTableLemma[currentTerm].append(currentId)
        lengthTable[currentId] = currentLength

        # compute all pairs for the new match (A in A B ... B A)
        newPairs = list((termId, currentId) for termId in storageTableLemma[currentTerm] if termId != currentId)
        
        # compute all possible pairs of old matches (B in A B ... B A)
        oldMatches = list(itertools.combinations(matchTableLemma[oldTerm], 2) for oldTerm in matchTableLemma)
        # if oldTerm != currentTerm (to avoid A A A A ?)
        
        oldMatches = list(list(x) for x in oldMatches)

        # iterate over all pairs for the new match
        for newPair in newPairs:

            # check if the chiasmus contains more than two pairs
            nestedCandidates = search_nested_chiasmi(newPair, candidateList)
            for nested in nestedCandidates:
                append_nested_to_candidates(
                    startBlock, endBlock,
                    newPair[0], newPair[1],
                    lengthTable[newPair[0]], lengthTable[newPair[1]],
                    nested, alreadyDetectedCandidates
                )
            
            # iterate over all old matches
            for oldMatch in oldMatches:
                # iterate over all pairs from the old match to check if it is inside the new match
                for oldPair in oldMatch:
                    if oldPair[0] > newPair[0] and oldPair[1] < newPair[1]:
                        # found a chiasmus candidate                
                        # we need, for each candidate : 
                        #   - the position in the raw text of the first character of the first word of the block
                        #   - the position in the raw text of the last character of the 5th word coming after the block
                        #       -> currently, we take the subsequent 25 characters
                        #   - the position in the block of the words forming the candidate
                        append_to_candidates(
                            startBlock, endBlock,
                            newPair[0], oldPair[0], oldPair[1], newPair[1],
                            lengthTable[newPair[0]], lengthTable[oldPair[0]],
                            lengthTable[oldPair[1]], lengthTable[newPair[1]]
                        )
                        alreadyDetectedCandidates.append((newPair[0], oldPair[0], oldPair[1], newPair[1]))

        # update the match table
        matchTableLemma[currentTerm] = copy.deepcopy(storageTableLemma[currentTerm])
    else:
        # no match, let's update the storage table
        storageTableLemma[currentTerm] = [currentId]
        lengthTable[currentId] = currentLength

    # --- Search of chiasmi through embedding (semantic) similarity (!!! YET UNTESTED !!!)

    currentText = currentWord.text
    currentEmb = glove_emb(currentText)
    currentLen = len(currentText)

    # Search for possible matches
    for oldWordId, (emb, oldWordLen) in storageTableEmbedding.items():
        similarity = emb_similarity(currentEmb, emb)
        if similarity > EMBEDDING_SIMILARITY_LIMIT or similarity < -EMBEDDING_SIMILARITY_LIMIT:
            # We have a match! Searching for possible nested chiasmi first
            allNestedCandidates = search_nested_chiasmi((oldWordId, currentId), candidateList)
            for nestedCandidate in allNestedCandidates:
                candidateComposedId = tuple([oldWordId] + [nestedTerm[0] for nestedTerm in nestedCandidate[1]] + [currentId])
                if candidateComposedId not in alreadyDetectedCandidates:
                    append_nested_to_candidates(
                        startBlock, endBlock,
                        oldWordId, currentId,
                        oldWordLen, currentLen,
                        nestedCandidate, alreadyDetectedCandidates
                    )

            # No nested chiasmi, searching for regular chiasmi now
            for oldPair1, matchedWords in matchTableEmbedding.items():
                # We need the second pairs to be contained withing the first pair
                if oldWordId < oldPair1:
                    oldPair1Len = storageTableEmbedding[oldPair1][1]
                    for oldPair2 in matchedWords:
                        oldPair2len = storageTableEmbedding[oldPair2][1]
                        # Making sure we are only taking into account non-lemma chiasmi:
                        if (oldWordId, oldPair1, oldPair2, currentId) not in alreadyDetectedCandidates:
                            append_to_candidates(
                                startBlock, endBlock,
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


def main():
    
    # -- Initializing the project --

    if __name__ == '__main__' and len(sys.argv) >= 2:
        fileName = sys.argv[1]
    else:
        fileName = input('Enter the name of the file to process : ')
        if fileName == "":  # Quick hack to launch things faster when testing, to delete in the final version
            fileName = "small-chiasmi.txt"

    content = get_file_content(fileName, "inputs")
    if(content == -1):
        exit(0)


    # -- Initializing the Stanza pipeline --

    stanza.download('en', processors='tokenize, lemma, pos')
    processingPipeline = stanza.Pipeline('en', processors='tokenize, lemma, pos', download_method=DownloadMethod.REUSE_RESOURCES)

    doc = processingPipeline(content)
    wordsFront = doc.iter_words()
    wordsBack = doc.iter_words()

    stopwords = set(line.strip() for line in open('stopwords.txt'))
    
    # -- Initializing the sliding window over the first 30 characters --

    initRange = 30
    if(doc.num_words <= 30):
        initRange = doc.num_words
        
    for _ in range(initRange):
        nextWord = next(wordsFront)
        nextWord = ignore_punctuation_and_stopwords(wordsFront, nextWord, stopwords)
        
        # if we reached the end of the file
        if nextWord == -1:
            break

        process_next_word(nextWord, nextWord.parent.start_char,  0, nextWord.parent.end_char)

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
        oldId = oldWord.parent.start_char
        startBlock = oldWord.parent.start_char

        # Processing the front of the window
        process_next_word(nextWord, nextWord.parent.start_char, startBlock, nextWord.parent.end_char)
        
        # handle the rear of the window
        # Delete the word exiting the sliding window from lemmaTable
        if len(storageTableLemma[oldLemma]) <= 1:
            for _id in storageTableLemma[oldLemma]:
                del lengthTable[_id]
            del storageTableLemma[oldLemma]
        else:
            del lengthTable[storageTableLemma[oldLemma][0]]
            del storageTableLemma[oldLemma][0]
        # Updating matchTable if necessary after this deletion
        if oldLemma in matchTableLemma:
            # delete when only one occurrence is left - not a match anymore
            if len(matchTableLemma[oldLemma]) <= 2:
                del matchTableLemma[oldLemma]
            else:
                del matchTableLemma[oldLemma][0]

        # Deleting the word exiting the sliding window from the embedding tables
        del storageTableEmbedding[oldId]
        if oldId in matchTableEmbedding:
            for _ in range(len(matchTableEmbedding[oldId])):
                del matchTableEmbedding[oldId][0]
            del matchTableEmbedding[oldId]

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

    with open(fileNameCandidates, 'w') as fileOut:
        
        for candidateBlock, candidateTerms in candidateList:
            candidateJson["text"] = word_from_positions(candidateBlock, content)
            candidateJson["entities"] = []
            
            for letterIndex, termPair in enumerate(candidateTerms):
                newPair = []
                for term in termPair:
                    term = term - candidateBlock[0]
                    newPair.append(term)
                if letterIndex < len(candidateTerms)/2:
                    newPair.append(termLetters[letterIndex] + "-1")
                else:
                    newPair.append((termLetters[len(candidateTerms) - letterIndex - 1]) + "-2")
                candidateJson["entities"].append(newPair)
            
            # adding metadata useful for post-annotation processings
            candidateJson["startBlock"] = candidateBlock[0]
            candidateJson["endBlock"] = candidateBlock[1]

            fileOut.write(json.dumps(candidateJson))
            fileOut.write("\n")
        
        fileOut.close()

    print("\n---------")
    print("Candidates stored in", fileNameCandidates)
    
if __name__ == "__main__":
    main()