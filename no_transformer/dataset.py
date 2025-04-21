import os
from itertools import islice
from file_util import dump, load
from datasets import load_dataset
from wikidata.client import Client
from entity_factory import EntityFactory

TRAINING_FILE_NAME = "training.bin"
VALIDATION_FILE_NAME = "validation.bin"

def create_set(dataset, factory, limit, file_name):
    # apply the limits
    if limit is None:
        limit = len(dataset)
    result = []
    for index, item in enumerate(islice(dataset, limit)):
        created = factory.create(item)
        if created is not None:
            result.append(created)
        if (index + 1) % 10 == 0:
            print("creating", file_name, index + 1, "/", limit)
    return result

class Dataset:
    def __init__(self, training_limit=None, validation_limit=None, force_reload=False):
        if not (os.path.exists(TRAINING_FILE_NAME)) or not (os.path.exists(VALIDATION_FILE_NAME)) or force_reload:
            # load the project dataset
            dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
            # a factory object is used to create our entities
            factory = EntityFactory(Client())

            self.training_set = create_set(dataset['train'], factory, training_limit, TRAINING_FILE_NAME)
            self.validation_set = create_set(dataset['validation'], factory, validation_limit, VALIDATION_FILE_NAME)
            dump(TRAINING_FILE_NAME, self.training_set)
            dump(VALIDATION_FILE_NAME, self.validation_set)
        else:
            # by default load the dataset from a local dump
            self.training_set = load(TRAINING_FILE_NAME)
            self.validation_set = load(VALIDATION_FILE_NAME)

    def __str__(self):
        return "training: " + str(len(self.training_set)) + ". validation: " + str(len(self.validation_set))
