import nltk
import unicodedata
import numpy as np

from nltk.stem.snowball import FrenchStemmer
stemmer = FrenchStemmer()

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def tokenize(sentence):
    sentence = remove_accents(sentence)
    return nltk.word_tokenize(sentence)

def stem(word):
    word = remove_accents(word)
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
