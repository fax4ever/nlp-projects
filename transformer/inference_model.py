import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class InferenceModel:
    def __init__(self, repo, kind):
        self.model = AutoModelForSequenceClassification.from_pretrained(repo)
        self.tokenizer = AutoTokenizer.from_pretrained(kind)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, desc, wiki):
        self.model.eval()
        # no max length - we want to use the default of the base model
        # as we do in training
        encoding = self.tokenizer(desc, wiki, return_tensors='pt', padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)
        return prediction.item()