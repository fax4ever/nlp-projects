import sys, os
from datasets import load_dataset

sys.path.insert(1, sys.path[0].replace("transformer", "no_transformer"))
from nlp_dataset import NLPDataset
from entity_factory import extract_entity_id

def build_entity_dict():
    entity_dict = {}
    nlp_dataset = NLPDataset()
    for entity in nlp_dataset.training_set:
        entity_dict[entity.entity_id] = entity
    for entity in nlp_dataset.validation_set:
        entity_dict[entity.entity_id] = entity
    return entity_dict

def output_label(label):
    if label == 'cultural agnostic':
        return 0
    if label == 'cultural representative':
        return 1
    if label == 'cultural exclusive':
        return 2
    raise ValueError('label not suppoerted: ' + label)

class WikiDataset:
    def __init__(self):
        entity_dict = build_entity_dict()
        dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
        # enriching the entities with the wiki pages
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
        self.dataset = dataset.map(map_labels)

    def tokenize(self, tokenizer):
        def tokenize_function(items):
            return tokenizer(items["description"], items["wiki_text"], padding=True, truncation=True)
        return self.dataset.map(tokenize_function, batched=True)
