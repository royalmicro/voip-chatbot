from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from config import Config


class ChatResponsePredictionModel:
    def __init__(self, model_name: str) -> None:
        self.config = Config()
        self.model_name = model_name

    def execute(self, X_train: list, y_train: list) -> None:
        model = Sequential(
            [
                Input(shape=(X_train.shape[1],)),  # Input layer
                Dense(128, activation="relu"),  # Hidden layer with ReLU activation
                Dropout(0.5),
                Dense(
                    64, activation="relu"
                ),  # Another hidden layer with ReLU activation
                Dropout(0.5),
                Dense(
                    y_train.shape[0], activation="softmax"
                ),  # Output layer with softmax activation
            ]
        )

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        # Compile the model
        model.compile(
            optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        # Train the model
        hist = model.fit(
            X_train,
            y_train,
            epochs=500,
            batch_size=5,
            validation_split=0.2,
            verbose=1,
        )

        model.save(
            self.config.get_model_path() + "/" + self.model_name + "_model.h5", hist
        )
