import os
from wikidata.client import Client
from datasets import load_dataset
from feature import Feature

def extract_entity_id(url):
    return url.strip().split("/")[-1]

class DataAccess:
    def __init__(self):
        self.dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
        self.client = Client()

    def train(self, index):
        return self.dataset['train'][index]

    def entity(self, entity_id):
        return self.client.get(entity_id, load=True)

    def feature(self, index):
        train = self.train(index)
        entity_id = extract_entity_id(train['item'])
        entity = self.entity(entity_id)
        return Feature(entity_id, train, entity)




