import pickle
from config import Config


class GetVocabulary:
    def __init__(self) -> None:
        self.config = Config()

    def execute(self, model_name: str) -> tuple[any, any]:
        words = pickle.load(
            open(self.config.get_intents_path() + "/" + model_name + "/words.pkl", "rb")
        )
        classes = pickle.load(
            open(
                self.config.get_intents_path() + "/" + model_name + "/classes.pkl", "rb"
            )
        )

        return words, classes
