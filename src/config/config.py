import os


class Config:
    ACTUAL_DIR = os.path.dirname(__file__)
    
    MODEL_INIT = "init"
    MODEL_CONFIGURATION = "configuration"

    def __init__(self) -> None:
        self.data_path = os.path.abspath(
            os.path.join((self.ACTUAL_DIR), "../..", "data")
        )
        self.model_path = os.path.abspath(
            os.path.join((self.ACTUAL_DIR), "../..", "model_storage")
        )
        
    def get_data_path(self) -> str:
        return self.data_path

    def get_intents_path(self) -> str:
        return os.path.abspath(
            os.path.join(self.data_path, "intents")
        )
    
    def get_model_path(self) -> str:
        return self.model_path