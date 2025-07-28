from datasets import load_dataset

class TokenClassification:
    def __init__(self, name):
        self.name = name
        self.dataset = load_dataset("conll2003", trust_remote_code=True)

    def hello(self):
        return "ciao " + self.name
