import os.path

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