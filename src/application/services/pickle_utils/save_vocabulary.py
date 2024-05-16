import pickle
from config import Config

class SaveVocabulary:
    def __init__(self) -> None:
        self.config = Config()

    def execute(self, all_words: list[str], tags: list[str], model_name: str) -> None:
        pickle.dump(
            all_words,
            open(
                self.config.get_intents_path() + "/" + model_name + "/words.pkl",
                "wb",
            ),
        )
        pickle.dump(
            tags,
            open(
                self.config.get_intents_path() + "/" + model_name + "/classes.pkl",
                "wb",
            ),
        )
