import stanza
from stanza.pipeline.core import DownloadMethod
import itertools
import string
import copy
# doc for stanza : https://stanfordnlp.github.io/stanza/data_objects#document


# -- Utility functions --
def first_word_from(text, word_begin):
    return text[word_begin:].split(maxsplit=1)[0].lower()


# -- Initializing the project --

# fileName = input('Enter the name of the file to process : ')
fileName = 'small-chiasmi.txt'

with open(fileName) as file:
    content = file.read()
    file.close()

stanza.download('en', processors='tokenize, lemma, pos')
processingPipeline = stanza.Pipeline('en', processors='tokenize, lemma', download_method=DownloadMethod.REUSE_RESOURCES)

doc = processingPipeline(content)
wordsFront = doc.iter_words()
wordsBack = doc.iter_words()

# sentence = doc.sentences[0]
# for word in sentence.words:
#     print(word.lemma)

# [TBD] : stopword and punctuation : either delete them with pre-processing or properly ignore them in the processing function

# we have corresponding lemmas for each word. We need to initialize a window of 30 lemmas and put them in a hash
# table. Every time we have a match -> verify if we have a match inside


# -- Creating the general variables --

candidateList = []
lemmaTable = {}
matchTable = {}


# -- Processing function for each next word --

def process_next_word(nextWord):
    currentLemma = nextWord.lemma

    # If we are currently processing a "punctuation word", then we ignore it
    # [TBD] : ignore rather than skip (currently, it messes with the window initialization, not going up to 30 words)
    if currentLemma in string.punctuation:
        return

    currentId = nextWord.parent.start_char
    # print('current id : ', [currentId], 'current lemma : ', currentLemma)
    if currentLemma in lemmaTable:
        # we have a match !
        # update lemma table
        lemmaTable[currentLemma].append(currentId)

        # compute all possible pairs for the new match
        newMatches = list(itertools.combinations(lemmaTable[currentLemma], 2))

        # compute all possible pairs of old matches
        oldMatches = [(oldMatch, list(itertools.combinations(matchTable[oldMatch], 2))) for oldMatch in matchTable]

        for newMatch in newMatches:
            # print('newMatch ! ; ids :', newMatch)
            for oldMatch in oldMatches:
                # print('oldMatch ! lemma : ', oldMatch[0], ' ; ids : ', oldMatch[1])
                # iterate over all pairs from the old match to check if it is inside the new match
                for oldPair in oldMatch[1]:
                    if (oldPair[0] > newMatch[0] and oldPair[1] < newMatch[1]):
                        # found a chiasmus candidate
                        candidateList.append([newMatch[0], oldPair[0], oldPair[1], newMatch[1]])
        #                 print("new candidate")
        # print('----')

        # update match table
        matchTable[currentLemma] = copy.deepcopy(lemmaTable[currentLemma])
    else:
        # update lemma table
        lemmaTable[currentLemma] = [currentId]


# -- Initializing the sliding window over the first 30 characters --

for _ in range(30):
    nextWord = next(wordsFront)
    process_next_word(nextWord)

# -- Main part : make the window slide using wordsFront and wordsBack --
#    (Same algorithm but delete info relevant to wordsBack when moving forward)

# loop stops when wordsFront ends
for nextWord, oldWord in zip(wordsFront, wordsBack):
    oldLemma = oldWord.lemma
    # print(lemmaTable)
    # print('--------')

    # Delete the word exiting the sliding window from lemmaTable (only non-punctuation words)
    if oldLemma not in string.punctuation:
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

    # Processing the new word
    process_next_word(nextWord)


# -- Printing the result (simple printing, some error when extracting the word, unoptimized) --

candidateWordList = [
    [(candidate_begin, first_word_from(content, candidate_begin)) for candidate_begin in candidate]
    for candidate
    in candidateList
]
print('-------\ncandidate list :')
for candidate in candidateWordList:
    print(candidate)

