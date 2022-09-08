import stanza
from stanza.pipeline.core import DownloadMethod
import itertools

# doc for stanza : https://stanfordnlp.github.io/stanza/data_objects#document

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

# [TBD] : remove lemmas whose words are in the stopword list ?

# we have corresponding lemmas for each word. We need to initialize a window of 30 lemmas and put them in a hash table. Every time we have a match -> verify if we have a match inside

candidateList = []
lemmaTable = {}
matchTable = {}

# initializing
for _ in range(30):
    currentWord = next(wordsFront)
    currentLemma = currentWord.lemma
    currentId = currentWord.parent.start_char
    # print('current id : ', [currentId], 'current lemma : ', currentLemma)
    if currentLemma in lemmaTable:
        # we have a match !
        # update lemma table
        lemmaTable[currentLemma].append(currentId)
        
        # compute all possible pairs for the new match
        newMatches = list(itertools.combinations(lemmaTable[currentLemma], 2))
        
        # compute all possible pairs of old matches
        oldMatches = list(((oldMatch, list(itertools.combinations(matchTable[oldMatch], 2))) for oldMatch in matchTable))
        
        for newMatch in newMatches:
            # print('newMatch ! ; ids :', newMatch)
            for oldMatch in oldMatches:
                # print('oldMatch ! lemma : ', oldMatch[0], ' ; ids : ', oldMatch[1])
                # iterate over all pairs from the old match to check if it is inside the new match
                for oldPair in oldMatch[1]:
                    if(oldPair[0] > newMatch[0] and oldPair[1] < newMatch[1]):
                        # found a chiasmus candidate
                        candidateList.append([newMatch[0], oldPair[0], oldPair[1], newMatch[1]])
        #                 print("new candidate")
        # print('----')
             
        
        # update match table
        matchTable[currentLemma] = lemmaTable[currentLemma]
    else:
        # update lemma table
        lemmaTable[currentLemma] = [currentId]
    

print('-------\ncandidate list :')
print(candidateList)
            

# initializing done, now for the "real work" : make the window slide using wordsFront and wordsBack, same algorithm but delete info relevant to wordsBack when moving forward    
