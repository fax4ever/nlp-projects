import os, pickle
from itertools import islice

from datasets import load_dataset
from wikidata.client import Client
from entity_factory import EntityFactory

TRAINING_FILE_NAME = "training.bin"
VALIDATION_FILE_NAME = "validation.bin"

def create_set_and_dump(dataset, factory, limit, file_name):
    # remove dump files if present
    if os.path.exists(file_name):
        os.remove(file_name)
    # apply the limits
    if limit is None:
        limit = len(dataset)
    result = []
    for index, item in enumerate(islice(dataset, limit)):
        result.append(factory.create(item))
        if (index + 1) % 10 == 0:
            print("creating", file_name, index + 1, "/", limit)
    # save files for the next time!
    with open(file_name, 'wb') as file:
        print("dumping", file_name)
        # noinspection PyTypeChecker
        pickle.dump(result, file)
    return result

class Dataset:
    def __init__(self, training_limit=None, validation_limit=None, force_reload=False):
        if not (os.path.exists(TRAINING_FILE_NAME)) or not (os.path.exists(VALIDATION_FILE_NAME)) or force_reload:
            # load the project dataset
            dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
            # a factory object is used to create our entities
            factory = EntityFactory(Client())

            self.training_set = create_set_and_dump(dataset['train'], factory, training_limit, TRAINING_FILE_NAME)
            self.validation_set = create_set_and_dump(dataset['validation'], factory, validation_limit, VALIDATION_FILE_NAME)
        else:
            # by default load the dataset from a local dump
            with open(TRAINING_FILE_NAME, 'rb') as file:
                print("loading", TRAINING_FILE_NAME)
                # noinspection PyTypeChecker
                self.training_set = pickle.load(file)
            with open(VALIDATION_FILE_NAME, 'rb') as file:
                print("loading", VALIDATION_FILE_NAME)
                # noinspection PyTypeChecker
                self.validation_set = pickle.load(file)

    def __str__(self):
        return "training: " + str(len(self.training_set)) + ". validation: " + str(len(self.validation_set))
