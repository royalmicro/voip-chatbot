import os


class Config:
    ACTUAL_DIR = os.path.dirname(__file__)

    def __init__(self) -> None:
        self.data_path = os.path.abspath(
            os.path.join((self.ACTUAL_DIR), "../..", "data")
        )
        self.model_path = os.path.abspath(
            os.path.join((self.ACTUAL_DIR), "../..", "model_storage")
        )
    def get_data_path(self) -> str:
        return self.data_path

    def get_model_path(self) -> str:
        return self.model_path