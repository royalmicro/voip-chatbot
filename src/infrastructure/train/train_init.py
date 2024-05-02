import json
import pickle
import numpy as np

from typing import Tuple, List
from tensorflow import keras  # noqa: F401
from config import Config
from application.services import NltkUtils
from domain import Intent
from application.models import ChatResponsePredictionModel


class TrainInit:
    IGNORE_WORDS = ["?", "!", ".", ","]

    def __init__(self) -> None:
        self.config = Config()
        self.intents: list[Intent] = []
        self.chatModel = ChatResponsePredictionModel(self.config.MODEL_INIT)
        self.nltk_utils = NltkUtils()
        with open(
            self.config.get_intents_path()
            + "/"
            + self.config.MODEL_INIT
            + "/"
            + self.config.MODEL_INIT
            + ".intent.json",
            "r",
        ) as f:
            intents = json.load(f)

        self.intents = self._intents_from_json(intents)

    def execute(self) -> None:

        tags: list[str] = []
        all_words: list[str] = []
        xy: Tuple[List[str], str] = []
        X_train = []
        y_train = []

        for intent in self.intents:
            tag: str = intent.get_tag()
            tags.append(tag)

            for pattern in intent.get_patterns():
                w = self.nltk_utils.tokenize(pattern)
                all_words.extend(w)
                xy.append((w, tag))

        all_words = [
            self.nltk_utils.stem(w) for w in all_words if w not in self.IGNORE_WORDS
        ]
        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        pickle.dump(
            all_words,
            open(
                self.config.get_intents_path()
                + "/"
                + self.config.MODEL_INIT
                + "/words.pkl",
                "wb",
            ),
        )
        pickle.dump(
            tags,
            open(
                self.config.get_intents_path()
                + "/"
                + self.config.MODEL_INIT
                + "/classes.pkl",
                "wb",
            ),
        )

        for pattern_sentence, tag in xy:
            bag = self.nltk_utils.bag_of_words(pattern_sentence, all_words)
            X_train.append(bag)

            label = tags.index(tag)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.chatModel.execute(X_train, y_train)

    def _intents_from_json(self, intents_json: object) -> list[Intent]:

        intents: list[Intent] = []
        try:
            for intent in intents_json[self.config.MODEL_INIT]:
                intent = Intent(
                    intent["tag"],
                    intent["patterns"],
                    intent["responses"],
                )
                intents.append(intent)
        except Exception as e:
            print(e)

        return intents
