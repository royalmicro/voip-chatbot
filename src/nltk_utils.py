from typing import List
import  nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence: str) -> list[str]:
    return nltk.word_tokenize(sentence)

def stem(word: str):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence: str, all_words: List[str]):
    
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # Initialize a list to hold the counts of each word
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Count the occurrences of each word in the tokenized sentence
    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag
    