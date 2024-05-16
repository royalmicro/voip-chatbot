from domain.entities.intent import Intent


class ParseIntents:
    def __init__(self) -> None:
        pass

    def execute(self, intents_json: object, model_name: str) -> list[Intent]:
        intents: list[Intent] = []
        try:
            for intent in intents_json[model_name]:
                intent = Intent(
                    intent["tag"],
                    intent["patterns"],
                    intent["responses"],
                )
                intents.append(intent)
        except Exception as e:
            print(e)

        return intents
