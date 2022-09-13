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
    
def get_file_content(fileName, possibleFolder):
    try:
        with open(fileName) as file:
            print("Loading content of ", fileName, '...')
            content = file.read()
            file.close()
            print("File loaded !")
            print("-------------")
            print()
            return content
    except IOError:
        try:
            secondFileName = possibleFolder + fileName
            with open(secondFileName) as file:
                print("Loading content of ", secondFileName, '...')
                content = file.read()
                file.close()
                print("File loaded !")
                print("-------------")
                print()
                return content
        except IOError:
            print("File ", fileName, " not found.")
            return -1