import os
import pandas as pd
from file_util import dump, load
from datasets import load_dataset
from wikidata.client import Client
from entity_factory import EntityFactory

TEST_SET_FILE_NAME = "test_unlabeled.csv"

TRAINING_FILE_NAME = "training.bin"
VALIDATION_FILE_NAME = "validation.bin"
TEST_FILE_NAME = "test.bin"

def create_set(dataset, factory, file_name):
    limit = len(dataset)
    result = []
    for index, item in enumerate(dataset):
        created = factory.create(item)
        if created is not None:
            result.append(created)
        if (index + 1) % 10 == 0:
            print("creating", file_name, index + 1, "/", limit)
    return result

def load_or_create_set(factory, dataset, file_name):
    if not (os.path.exists(file_name)):
        created = create_set(dataset, factory, file_name)
        dump(file_name, created)
        return created
    else:
        return load(file_name)

class NLPDataset:
    def __init__(self):
        if (not (os.path.exists(TRAINING_FILE_NAME)) or not (os.path.exists(VALIDATION_FILE_NAME))
                or not (os.path.exists(TEST_FILE_NAME))):
            # load the project dataset
            dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
            # a factory object is used to create our entities
            factory = EntityFactory(Client())

            self.training_set = load_or_create_set(factory, dataset['train'], TRAINING_FILE_NAME)
            self.validation_set = load_or_create_set(factory, dataset['validation'], VALIDATION_FILE_NAME)
            self.test_set = load_or_create_set(factory, pd.read_csv(TEST_SET_FILE_NAME).to_dict('records'), TEST_FILE_NAME)
        else:
            # by default load the dataset from a local dump
            self.training_set = load(TRAINING_FILE_NAME)
            self.validation_set = load(VALIDATION_FILE_NAME)
            self.test_set = load(TEST_FILE_NAME)

    def __str__(self):
        return ("training: " + str(len(self.training_set)) + ". validation: " + str(len(self.validation_set)) +
                ". test: " + str(len(self.test_set)))

if __name__ == "__main__":
    nlp = NLPDataset()
    print(nlp)
