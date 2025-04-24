import os, sys
from datasets import load_dataset
from matplotlib import pyplot as plt

from nlp_hyper_params import NLPHyperParams, compute_metrics
from nlp_encoder_model import NLPEncoderModel
from nlp_trainer import NLPTrainer

sys.path.insert(1, sys.path[0].replace("transformer", "no_transformer"))
from nlp_dataset import NLPDataset
from entity_factory import extract_entity_id

entity_dict = {}

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
    wiki_id = extract_entity_id(sample["item"])
    if wiki_id is not None and wiki_id in entity_dict:
        wiki_text = entity_dict[wiki_id].wiki_text
        sample["wiki_text"] = wiki_text if type(wiki_text) == str else ""
    else:
        sample["wiki_text"] = ""
    return sample

def main():
    nlp_dataset = NLPDataset()
    for entity in nlp_dataset.training_set:
        entity_dict[entity.entity_id] = entity
    for entity in nlp_dataset.validation_set:
        entity_dict[entity.entity_id] = entity

    params = NLPHyperParams()
    dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
    print(dataset)
    dataset = dataset.map(map_labels)
    model = NLPEncoderModel(params)

    def tokenize_function(items):
        return model.tokenizer(items["description"], items["wiki_text"], padding=True, truncation=True)
    print("Tokenize the dataset ...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print(tokenized_datasets)

    train_ = tokenized_datasets["train"]
    validation_ = tokenized_datasets["validation"]
    trainer = NLPTrainer(params, model, train_, validation_)
    history = trainer.train_and_evaluate()

    plt.title("MSE Loss - Plot")
    plt.plot(history["train_loss"], label="training loss")
    plt.plot(history["valid_loss"], label="validation loss")
    plt.legend()
    plt.show()

    plt.title("Accuracy - Plot")
    plt.plot(history["train_accuracy"], label="training accuracy")
    plt.plot(history["valid_accuracy"], label="validation accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()