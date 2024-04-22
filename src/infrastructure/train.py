import json
import pickle
from typing import Tuple, List
import numpy as np

import nltk

from tensorflow import keras  # noqa: F401

from config import Config
from nltk_utils import tokenize, stem, bag_of_words
from domain import Intent
from application import ChatResponsePredictionModel

nltk.download("punkt")
nltk.download("wordnet")


class Train:
    IGNORE_WORDS = ["?", "!", ".", ","]

    def __init__(self) -> None:
        self.config = Config()
        self.intents: list[Intent] = []
        self.chatModel = ChatResponsePredictionModel()

    def execute(self) -> None:
        with open(self.config.get_data_path() + "/intents.json", "r") as f:
            intents = json.load(f)

        intents = self._intents_from_json(intents)

        tags: list[str] = []
        all_words: list[str] = []
        xy: Tuple[List[str], str] = []
        X_train = []
        y_train = []

        for intent in intents:
            tag: str = intent.get_tag()
            tags.append(tag)

            for pattern in intent.get_patterns():
                w = tokenize(pattern)
                all_words.extend(w)
                xy.append((w, tag))

        all_words = [stem(w) for w in all_words if w not in self.IGNORE_WORDS]
        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        pickle.dump(all_words, open(self.config.get_data_path() + "/words.pkl", "wb"))
        pickle.dump(tags, open(self.config.get_data_path() + "/classes.pkl", "wb"))

        for pattern_sentence, tag in xy:
            bag = bag_of_words(pattern_sentence, all_words)
            X_train.append(bag)

            label = tags.index(tag)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.chatModel.execute(X_train, y_train)

    def _intents_from_json(self, intents_json: object) -> list[Intent]:

        intents: list[Intent] = []
        try:
            for intent in intents_json["intents"]:
                intent = Intent(
                    intent["tag"],
                    intent["patterns"],
                    intent["responses"],
                )
                intents.append(intent)
        except Exception as e:
            print(e)

        return intents
