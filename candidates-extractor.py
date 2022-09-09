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
    
# -- Initializing the project --

# fileName = input('Enter the name of the file to process : ')
fileName = 'small-chiasmi.txt'
# fileName = 'chiasmi.txt'
# fileName = 'test.txt'

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

def process_next_word(currentTerm, currentId, storageTable, matchTable):
    if currentTerm in storageTable:
        # we have a match ! Let's update the storage table
        storageTable[currentTerm].append(currentId)

        # compute all possible pairs for the new match
        newMatches = list(itertools.combinations(storageTable[currentTerm], 2))

        # compute all possible pairs of old matches
        oldMatches = [(oldMatch, list(itertools.combinations(matchTable[oldMatch], 2))) for oldMatch in matchTable]

        for newMatch in newMatches:
            for oldMatch in oldMatches:
                # iterate over all pairs from the old match to check if it is inside the new match
                for oldPair in oldMatch[1]:
                    if (oldPair[0] > newMatch[0] and oldPair[1] < newMatch[1]):
                        # found a chiasmus candidate
                        candidateList.append([newMatch[0], oldPair[0], oldPair[1], newMatch[1]])

        # update the match table
        matchTable[currentTerm] = copy.deepcopy(storageTable[currentTerm])
    else:
        # no match, let's update the storage table
        storageTable[currentTerm] = [currentId]

# -- Creating the general variables --

candidateList = []
lemmaTable = {}
matchTable = {}

# -- Initializing the sliding window over the first 30 characters --

for _ in range(30):
    nextWord = next(wordsFront)
    nextWord = ignore_punctuation_and_stopwords(wordsFront, nextWord, stopwords)
    
    # if we reached the end of the file
    if(nextWord == -1):
        break
        
    process_next_word(nextWord.lemma, nextWord.parent.start_char, lemmaTable, matchTable)

# -- Main part : make the window slide using wordsFront and wordsBack --
#    (Same algorithm but delete info relevant to wordsBack when moving forward)

# loop stops when wordsFront ends
for nextWord, oldWord in zip(wordsFront, wordsBack):
    # If we are currently processing a "punctuation word", then we ignore it
    nextWord = ignore_punctuation_and_stopwords(wordsFront, nextWord, stopwords)
    oldWord = ignore_punctuation_and_stopwords(wordsBack, oldWord, stopwords)
    oldLemma = oldWord.lemma

    # if we reached the end of the file
    if(nextWord == -1):
        break
        
    # Processing the new word
    process_next_word(nextWord.lemma, nextWord.parent.start_char, lemmaTable, matchTable)

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
    
# -- Printing the result (simple printing, some error when extracting the word, unoptimized) --
candidateWordList = [
    [(candidate_begin, first_word_from(content, candidate_begin)) for candidate_begin in candidate]
    for candidate
    in candidateList
]
print('-------\ncandidate list (', len(candidateWordList), ' candidates):')
for candidate in candidateWordList:
    print(candidate)

