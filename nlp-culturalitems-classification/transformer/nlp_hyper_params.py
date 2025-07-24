import torch
import evaluate
import numpy as np
from transformers import set_seed

#List of LMs: ("bigbird", "distilbert", "roberta_base", "roberta_large", 
#              "xlm_base", "xlm_large", "mdeberta_base", "mdeberta_large")
LMs = {
    "bigbird": {
        "model_name": "google/bigbird-roberta-base",
        "max_length": 4096,
        "epochs": 8,
        "batch_size": 4
    },
    "distilbert": {
        "model_name": "distilbert-base-uncased",
        "max_length": 512,
        "epochs": 8,
        "batch_size": 4
    },
    "roberta_base": {
        "model_name": "roberta-base",
        "max_length": 512,
        "epochs": 8,
        "batch_size": 32
    },
    "roberta_large": {
        "model_name": "roberta-large",
        "max_length": 512,
        "epochs": 8,
        "batch_size": 32
    },
    "xlm_base": {
        "model_name": "xlm-roberta-base",
        "max_length": 512,
        "epochs": 8,
        "batch_size": 32
    },
    "xlm_large": {
        "model_name": "xlm-roberta-large",
        "max_length": 512,
        "epochs": 8,
        "batch_size": 32
    },
    "mdeberta_base": {
        "model_name": "microsoft/mdeberta-v3-base",
        "max_length": 512,
        "epochs": 8,
        "batch_size": 32
    },
    "mdeberta_large": {
        "model_name": "microsoft/mdeberta-v3-large",
        "max_length": 512,
        "epochs": 8,
        "batch_size": 32
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
        key = "roberta_base"
        self.language_model_name = LMs[key]["model_name"]
        self.max_length = LMs[key]["max_length"]
        self.batch_size = LMs[key]["batch_size"]
        self.learning_rate = 1e-5
        self.weight_decay = 0.01 # we could use e.g. 0.01 in case of very low and very high amount of data for regularization
        self.epochs = LMs[key]["epochs"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: ", self.device)

