import numpy as np
import json
import pickle
from keras.models import load_model
import random
from config import Config
from application.services import NltkUtils, GetVocabulary


class ChatBot:
    def __init__(self) -> None:
        self.config = Config()
        self.nltk_utils = NltkUtils()
        self.get_vocabulary = GetVocabulary()

    def execute(self) -> None:
        print("GO! Bot is running!")

        while True:
            message = input("")
            model_name = self.config.MODEL_INIT
            predicted_intents = self._predict_class(message, model_name)

            if predicted_intents[0]["intent"] == self.config.MODEL_CONFIGURATION:
                model_name = self.config.MODEL_CONFIGURATION
                predicted_intents = self._predict_class(message, model_name)
            elif predicted_intents[0]["intent"] == self.config.MODEL_INFORMATION:
                model_name = self.config.MODEL_INFORMATION
                predicted_intents = self._predict_class(message, model_name)

            intent_path = "/".join(
                [
                    self.config.get_intents_path(),
                    model_name,
                    model_name + ".intent.json",
                ]
            )

            with open(intent_path, "r") as f:
                intents = json.load(f)

            res = self._get_response(
                predicted_intents, intents_json=intents, model_name=model_name
            )
            print(res)

    def _predict_class(self, sentence, model_name: str):
        model = load_model(
            self.config.get_model_path() + "/" + model_name + "_model.h5"
        )
        words, classes = self.get_vocabulary.execute(model_name)
        sentence = self.nltk_utils.tokenize(sentence)
        bow = self.nltk_utils.bag_of_words(sentence, words)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.40
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

        return return_list

    def _get_response(self, predicted_intents, intents_json, model_name: str):
        tag = predicted_intents[0]["intent"]
        list_of_intents = intents_json[model_name]

        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break

        return result
