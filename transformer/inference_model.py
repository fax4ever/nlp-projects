import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class InferenceModel:
    def __init__(self, repo, kind):
        self.model = AutoModelForSequenceClassification.from_pretrained(repo)
        self.tokenizer = AutoTokenizer.from_pretrained(kind)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict_text(self, desc, wiki, input_ids_ds, attention_mask_ds, max_length=512):
        self.model.eval()
        encoding = self.tokenizer(desc, wiki, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        input_ids_ds = torch.tensor(input_ids_ds).to(self.device).view(1, -1)
        attention_mask_ds = torch.tensor(attention_mask_ds).to(self.device).view(1, -1)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)
            outputs2 = self.model(input_ids=input_ids_ds, attention_mask=attention_mask_ds)
            _, prediction_ds = torch.max(outputs2.logits, dim=1)
        return prediction.item(), prediction_ds.item()