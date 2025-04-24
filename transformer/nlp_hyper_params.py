import torch
import evaluate
import numpy as np
from transformers import set_seed

#List of LMs: ("bigbird", "distilbert", "roberta", "xlm_base", "xlm_large", "mdeberta_base", "mdeberta_large")
LMs = {
    "bigbird": {
        "model_name": "google/bigbird-roberta-base",
        "max_length": 4096,
    },
    "distilbert": {
        "model_name": "distilbert-base-uncased",
        "max_length": 512,
    },
    "roberta": {
        "model_name": "roberta-base",
        "max_length": 512,
    },
    "xlm_base": {
        "model_name": "xlm-roberta-base",
        "max_length": 512,
    },
    "xlm_large": {
        "model_name": "xlm-roberta-large",
        "max_length": 512,
    },
    "mdeberta_base": {
        "model_name": "microsoft/mdeberta-v3-base",
        "max_length": 512,
    },
    "mdeberta_large": {
        "model_name": "microsoft/mdeberta-v3-large",
        "max_length": 512,
    }, 
}
# Metrics
def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    return {"accuracy": accuracy}

class NLPHyperParams:
    def __init__(self):
        set_seed(42)
        self.language_model_name = LMs["bigbird"]["model_name"] 
        self.max_length = LMs["bigbird"]["max_length"]
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 0.001 # we could use e.g. 0.01 in case of very low and very high amount of data for regularization
        self.epochs = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: ", self.device)

