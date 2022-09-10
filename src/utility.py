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