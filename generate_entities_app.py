import os, requests
from datasets import load_dataset
from entity_factory import EntityFactory

def main():
    dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
    factory = EntityFactory()

    training_set = []
    for item in dataset['train']:
        training_set.append(factory.create(item))
    validation_set = []
    for item in dataset['validation']:
        validation_set.append(factory.create(item))

    pass

if __name__ == "__main__":
    main()