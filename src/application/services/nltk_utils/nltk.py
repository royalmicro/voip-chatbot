from typing import List
import nltk.data
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer

class NltkUtils:
    nltk.download("perluniprops")
    nltk.download("nonbreaking_prefixes")
    

    def __init__(self) -> None:
        self.stemmer = SnowballStemmer("spanish")
        self.toktok = ToktokTokenizer()

    def tokenize(self, sentence: str) -> list[str] | str:
        return self.toktok.tokenize(sentence)

    def stem(self, word: str):
        return self.stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence: List[str] | str, all_words: List[str]):

        tokenized_sentence = [self.stem(w) for w in tokenized_sentence]

        # Initialize a list to hold the counts of each word
        bag = np.zeros(len(all_words), dtype=np.float32)

        # Count the occurrences of each word in the tokenized sentence
        for index, word in enumerate(all_words):
            if word in tokenized_sentence:
                bag[index] = 1.0

        return bag