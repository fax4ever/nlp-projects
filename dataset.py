import os, pickle
from itertools import islice

from datasets import load_dataset
from wikidata.client import Client
from entity_factory import EntityFactory

TRAINING_FILE_NAME = "training.bin"
VALIDATION_FILE_NAME = "validation.bin"

class Dataset:
    def __init__(self, training_limit=None, validation_limit=None, force_reload=False):
        if not(os.path.exists(TRAINING_FILE_NAME)) or not(os.path.exists(VALIDATION_FILE_NAME)) or force_reload:
            # remove dump files if present
            if os.path.exists(TRAINING_FILE_NAME):
                os.remove(TRAINING_FILE_NAME)
            if os.path.exists(VALIDATION_FILE_NAME):
                os.remove(VALIDATION_FILE_NAME)
            # load the project dataset
            dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
            # apply the limits
            if training_limit is None:
                training_limit = len(dataset['train'])
            if validation_limit is None:
                validation_limit = len(dataset['validation'])
            # a factory object is used to create our entities
            factory = EntityFactory(Client())
            self.training_set = []
            for item in islice(dataset['train'], training_limit):
                self.training_set.append(factory.create(item))
            self.validation_set = []
            for item in islice(dataset['validation'], validation_limit):
                self.validation_set.append(factory.create(item))
            # save files for the next time!
            with open(TRAINING_FILE_NAME, 'wb') as file:
                # noinspection PyTypeChecker
                pickle.dump(self.training_set, file)
            with open(VALIDATION_FILE_NAME, 'wb') as file:
                # noinspection PyTypeChecker
                pickle.dump(self.validation_set, file)
        else:
            # by default load the dataset from a local dump
            with open(TRAINING_FILE_NAME, 'rb') as file:
                # noinspection PyTypeChecker
                self.training_set = pickle.load(file)
            with open(VALIDATION_FILE_NAME, 'rb') as file:
                # noinspection PyTypeChecker
                self.validation_set = pickle.load(file)

    def __str__(self):
        return "training: " + str(len(self.training_set)) + ". validation: " + str(len(self.validation_set))