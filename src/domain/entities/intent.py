class Intent:
    def __init__(self, tag: str, patterns: list[str], responses: list[str]) -> None:
        self.tag = tag
        self.patterns = patterns
        self.responses = responses

    def get_tag(self):
        return self.tag

    def get_patterns(self):
        return self.patterns

    def get_responses(self):
        return self.responses
