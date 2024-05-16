from infrastructure import ChatBot, TrainIvozConfig, TrainInit, TrainIvozInformation
import sys


def execute_train_configuration():
    train = TrainIvozConfig()
    train.execute()

def execute_train_information():
    train = TrainIvozInformation()
    train.execute()

def execute_train_init():
    train = TrainInit()
    train.execute()


def execute_chatbot():
    chatbot = ChatBot()
    chatbot.execute()


if __name__ == "__main__":

    if "chatbot" in sys.argv:
        execute_chatbot()
    else:
        if "train_config" in sys.argv:
            execute_train_configuration()

        if "train_information" in sys.argv:
            execute_train_information()

        if "train_init" in sys.argv:
            execute_train_init()