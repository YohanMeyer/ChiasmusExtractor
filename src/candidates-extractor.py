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
WINDOW_SIZE = 30
Word = stanza.models.common.doc.Word

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

# -- Function adding a candidate to the current list --

def append_to_candidates(candidateList: list, startBlock: int, endBlock: int,
                         A1: Word, B1: Word, B2: Word, A2: Word):
    candidateList.append([[startBlock, endBlock + 25], [A1, B1, B2, A2]])

# -- Function adding a nested candidate (A B C ... C B A) to the current list --

def append_nested_to_candidates(candidateList: list, startBlock: int, endBlock: int,
                         newPair: list, nestedCandidate: list, alreadyDetectedCandidates: list):
    newCandidate= [[startBlock, endBlock + 25], [newPair[0]]]
    for nestedWord in nestedCandidate[1]:
        newCandidate[1].append(nestedWord)
    
    newCandidate[1].append(newPair[1])
    candidateList.append(newCandidate)

# -- Function to find nested chiasmi (A B C ... C B A) in the current state --

def search_nested_chiasmi(currentPair, candidateList):
    # iterate from the end to get recent matches
    nestedChiasmi = []
    for candidate in reversed(candidateList):
        if(candidate[1][-1].parent.end_char <= currentPair[0].parent.start_char):
            break
        elif(candidate[1][0].parent.start_char > currentPair[0].parent.end_char
                and candidate[1][-1].parent.end_char < currentPair[1].parent.start_char):
            nestedChiasmi.append(candidate)
    return nestedChiasmi

# -- Processing function for each next word --

def process_next_word(currentWord, startBlock, endBlock):
    alreadyDetectedCandidates = []

    # --- Search of chiasmi through lemma correspondence

    currentId = currentWord.parent.start_char
    currentLemma = currentWord.lemma

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
            allNestedCandidates = search_nested_chiasmi(newPair, candidateList)
            for nestedCandidate in allNestedCandidates:
                append_nested_to_candidates(
                    candidateList, startBlock, endBlock,
                    newPair, nestedCandidate, alreadyDetectedCandidates
                )
            # iterate over all old matches
            for oldMatchingPairs in oldMatches:
                # iterate over all pairs from the old match to check if it is inside the new match
                for oldPair in oldMatchingPairs:
                    if(oldPair[0].parent.start_char > newPair[0].parent.start_char 
                            and oldPair[1].parent.start_char < newPair[1].parent.start_char):
                        # found a chiasmus candidate
                        append_to_candidates(
                            candidateList, startBlock, endBlock,
                            newPair[0], oldPair[0], oldPair[1], newPair[1]
                        )

        # update the match table
        matchTableLemma[currentLemma] = copy.deepcopy(storageTableLemma[currentLemma])
    else:
        # no match, let's update the storage table
        storageTableLemma[currentLemma] = [currentWord]

    # --- Search of chiasmi through embedding (semantic) similarity (!!! YET UNTESTED !!!)

    currentEmb = glove_emb(currentWord.text)
    # Search for possible matches
    for oldWordId, (oldWord, oldEmb) in storageTableEmbedding.items():
        similarity = emb_similarity(currentEmb, oldEmb)
        if similarity > EMBEDDING_SIMILARITY_LIMIT or similarity < -EMBEDDING_SIMILARITY_LIMIT:
            
            # We have a match! Searching for possible nested chiasmi first
            newPair = [oldWord, currentWord]
            
            # avoid duplicates with lemmas
            if(oldWord.lemma.lower() != currentWord.lemma.lower()):
                allNestedCandidates = search_nested_chiasmi(newPair, candidateList)
                for nestedCandidate in allNestedCandidates:
                    append_nested_to_candidates(
                        candidateList, startBlock, endBlock, newPair,
                        nestedCandidate, alreadyDetectedCandidates
                    )

            # No nested chiasmi, searching for regular chiasmi now
            for oldPairId1, oldPairMatchingIds in matchTableEmbedding.items():
                # We need the second pairs to be contained withing the first pair
                oldPairWord1 = storageTableEmbedding[oldPairId1][0]
                
                if oldWordId < oldPairId1:
                    for oldPairId2 in oldPairMatchingIds:
                        oldPairWord2 = storageTableEmbedding[oldPairId2][0]
                        # check that it is not already covered by lemmas
                        if(newPair[0].lemma.lower() != newPair[1].lemma.lower()
                                or oldPairWord1.lemma.lower() != oldPairWord2.lemma.lower()):
                            append_to_candidates(
                                candidateList, startBlock, endBlock,
                                newPair[0], oldPairWord1,
                                oldPairWord2, newPair[1]
                            )
            # Updating the embedding match table
            if oldWordId in matchTableEmbedding:
                matchTableEmbedding[oldWordId].append(currentId)
            else:
                matchTableEmbedding[oldWordId] = [currentId]

    # Updating the embedding storage table
    storageTableEmbedding[currentId] = (currentWord, currentEmb)


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
        print("File", fileName, "not found.")
        exit(0)

    # -- Initializing the Stanza pipeline --

    stanza.download('en', processors='tokenize, lemma, pos, depparse')
    processingPipeline = stanza.Pipeline(
            'en', processors='tokenize, lemma, pos, depparse', 
            download_method=DownloadMethod.REUSE_RESOURCES
    )

    doc = processingPipeline(content)
    wordsFront = doc.iter_words()
    wordsBack = doc.iter_words()

    stopwords = set(line.strip() for line in open('stopwords.txt'))
    
    # -- Initializing the sliding window over the first 30 characters --
    
    sentenceIndex = -1
    for _ in range(WINDOW_SIZE - 1):
        try:
            nextWord = next(wordsFront)
            if(nextWord.id == 1):
                sentenceIndex = sentenceIndex + 1
            nextWord.sentenceIndex = sentenceIndex
        except StopIteration:
            # if we reached the end of the file
            break
        
        if(not is_punctuation_or_stopword(nextWord, stopwords)):
            # only process if it is a valid word
            process_next_word(nextWord, 0, nextWord.parent.end_char)
        
    # -- Main part : make the window slide using wordsFront and wordsBack --
    
    # foreach stops when wordsFront ends
    for nextWord, oldWord in zip(wordsFront, wordsBack):
        startBlock = oldWord.parent.start_char
        
        if(nextWord.id == 1):
            sentenceIndex = sentenceIndex + 1
        nextWord.sentenceIndex = sentenceIndex

        # Processing the front of the window, only process if it is a valid word
        if(not is_punctuation_or_stopword(nextWord, stopwords)):
            process_next_word(nextWord, startBlock, nextWord.parent.end_char)
        
        # handle the rear of the window, only delete if it is a valid word
        if(not is_punctuation_or_stopword(oldWord, stopwords)):
            oldLemma = oldWord.lemma
            oldId = oldWord.parent.start_char
            
            # Delete the word exiting the sliding window from lemmaTable
            if len(storageTableLemma[oldLemma]) <= 1:
                del storageTableLemma[oldLemma]
            else:
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
    
    # we now have our full list of candidates, need to write it into a file
    fileNameCandidates = os.path.join("..", "candidates", os.path.splitext(os.path.basename(fileName))[0] + "-candidates.jsonl")

    # format imposed by the usage of Deccano
    # "entities" will contain the positions of the chiasmi terms and "cats" the annotation label
    # Setting default label to "False" to speed up the annotation process
    candidateJson = {"text" : "", "entities" : [], "cats" : "NotAChiasmus"}
    termLetters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    
    with open(fileNameCandidates, 'w') as fileOut:
        
        for candidateBlock, candidateWords in candidateList:
            candidateJson["text"] = word_from_positions(candidateBlock, content)
            candidateJson["entities"] = []
            candidateJson["index"] = []
            
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
            
            candidateSentences = doc.sentences[candidateWords[0].sentenceIndex:
                    (candidateWords[-1].sentenceIndex + 1)]
            
            appendWord = False
            lemmaIndex = 0
            ids = [id[0] + startBlock for id in candidateJson["entities"]]
            for sentence in candidateSentences:
                for word in sentence.words:
                    candidateJson["lemmas"].append(word.lemma)
                    candidateJson["words"].append(word.text)
                    
                    if(word.parent.start_char in ids):
                        candidateJson["index"].append(lemmaIndex)
                    lemmaIndex += 1
                    
                    if(word == candidateWords[-1]):
                        break

            fileOut.write(json.dumps(candidateJson))
            fileOut.write("\n")
        
        fileOut.close()

    print("\n---------")
    print("Candidates stored in", fileNameCandidates)
    
if __name__ == "__main__":
    main()