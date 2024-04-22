import numpy as np
import json
import pickle
from keras.models import load_model
from nltk_utils import bag_of_words
import random
from config import Config
from nltk_utils import tokenize

class ChatBot:
    def __init__(self) -> None:
        self.config = Config()
        self.words = pickle.load(open(self.config.get_data_path() + "/words.pkl", "rb"))
        self.classes = pickle.load(
            open(self.config.get_data_path() + "/classes.pkl", "rb")
        )
        self.model = load_model(self.config.get_model_path() + "/chatbot_model.h5")
        self.intents = json.loads(
            open(self.config.get_data_path() + "/intents.json").read()
        )

    def execute(self) -> None:
        print("GO! Bot is running!")
        while True:
            message = input("")
            ints = self.predict_class(message)
            res = self.get_response(ints, intents_json=self.intents)
            print(res)

    def predict_class(self, sentence):
        sentence = tokenize(sentence)
        bow = bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})

        return return_list

    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]

        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break

        return result
