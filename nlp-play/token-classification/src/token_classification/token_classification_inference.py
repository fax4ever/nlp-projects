from transformers import pipeline

class TokenClassificationInference:
    def __init__(self):
        model_checkpoint = "fax4ever/bert-finetuned-ner"
        self.token_classifier = pipeline(
            "token-classification", model=model_checkpoint, aggregation_strategy="simple"
        )

    def token_classify(self, sentence):
        return self.token_classifier(sentence)