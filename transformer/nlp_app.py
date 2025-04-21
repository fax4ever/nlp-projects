import os
from datasets import load_dataset
from nlp_hyper_params import NLPHyperParams, compute_metrics
from nlp_encoder_model import NLPEncoderModel
from nlp_trainer import NLPTrainer

def output_label(label):
    if label == 'cultural agnostic':
        return 0
    if label == 'cultural representative':
        return 1
    if label == 'cultural exclusive':
        return 2
    raise ValueError('label not suppoerted: ' + label)

def map_labels(sample):
    label = sample["label"]
    sample["label"] = output_label(label)
    return sample

def main():
    params = NLPHyperParams()
    dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
    print(dataset)
    dataset = dataset.map(map_labels)
    model = NLPEncoderModel(params)

    def tokenize_function(examples):
        return model.tokenizer(examples["description"], padding=True, truncation=True)
    print("Tokenize the dataset ...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print(tokenized_datasets)

    train_ = tokenized_datasets["train"]
    validation_ = tokenized_datasets["validation"]
    trainer = NLPTrainer(params, model, train_, validation_)
    trainer.train_and_evaluate()

    print(model.predict("Italian universal popular dish with a flat dough-based base and toppings"))
    print(model.predict("sicilian eggplant dish"))
    print(model.predict("baked food made of flour, water and other ingredients"))

if __name__ == "__main__":
    main()