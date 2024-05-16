import json
import numpy as np

from typing import Tuple, List
from config import Config
from application.services import NltkUtils
from domain import Intent
from application.models import ChatResponsePredictionModel
from application.services import STOPWORDS_ES, SYMBOLS, ParseIntents, SaveVocabulary


class TrainInit:
    def __init__(self) -> None:
        self.config = Config()
        self.intents: list[Intent] = []
        self.chatModel = ChatResponsePredictionModel(self.config.MODEL_INIT)
        self.nltk_utils = NltkUtils()
        self.save_vocabulary = SaveVocabulary()
        self.intent_parser = ParseIntents()

    def execute(self) -> None:
        tags: list[str] = []
        all_words: list[str] = []
        xy: List[Tuple[List[str] | str, str]] = []
        X_train = []
        y_train = []

        with open(
            self.config.get_intents_path()
            + "/"
            + self.config.MODEL_INIT
            + "/"
            + self.config.MODEL_INIT
            + ".intent.json",
            "r",
        ) as f:
            json_intents = json.load(f)

        intents = self.intent_parser.execute(json_intents, self.config.MODEL_INIT)

        for intent in intents:
            tag: str = intent.get_tag()
            tags.append(tag)

            for pattern in intent.get_patterns():
                w = self.nltk_utils.tokenize(pattern)
                all_words.extend(w)
                xy.append((w, tag))

        all_words = [
            self.nltk_utils.stem(w)
            for w in all_words
            if w not in SYMBOLS and w not in STOPWORDS_ES
        ]
        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        self.save_vocabulary.execute(all_words, tags, self.config.MODEL_INIT)

        for pattern_sentence, tag in xy:
            bag = self.nltk_utils.bag_of_words(pattern_sentence, all_words)
            X_train.append(bag)

            label = tags.index(tag)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.chatModel.execute(X_train, y_train)
