import torch

from entity import Entity

def type_boolean(base_type):
    if base_type == 'entity':
        return 1
    if base_type == 'concept':
        return 0
    raise ValueError('type not suppoerted: ' + base_type)

def output_label(label):
    if label == 'cultural agnostic':
        return 0
    if label == 'cultural representative':
        return 1
    if label == 'cultural exclusive':
        return 2
    raise ValueError('label not suppoerted: ' + label)

class ProcessedEntity:
    def __init__(self, base: Entity, desc_text, wiki_text):
        self.base_entity = str(base)
        # processed fields
        self.desc_text = desc_text
        self.wiki_text = wiki_text
        self.labels_text = base.labels
        self.descriptions_text = base.descriptions
        self.aliases_text = base.aliases
        self.pages_text = base.wikipedia_pages
        # Using map to denote a Python dictionary,
        # since the dictionary is already use for a word (term) dictionary
        self.claims_map = base.claims
        self.subcategory = base.subcategory
        self.category = base.category

        # build later (then the dictionaries are finalized)
        self.desc_vector = None
        self.wiki_vector = None
        self.labels_vector = None
        self.descriptions_vector = None
        self.aliases_vector = None
        self.pages_vector = None
        self.claims_vector = None
        # it includes implicitly the category
        # since the subcategory values have been ordered by category
        self.subcategory_scalar = None
        self.subcategory_scalar_len = None
        # in this case we can assume that we have only two types (entity vs concept)
        self.type_boolean = type_boolean(base.type)
        self.output_label = output_label(base.label)

    def __str__(self):
        return self.base_entity + " < " + str(len(self.desc_text)) + ", " + str(len(self.wiki_text)) + " >"

    def desc_tensor(self):
        return torch.tensor(self.desc_vector)
    def wiki_tensor(self):
        return torch.tensor(self.wiki_vector)
    def labels_tensor(self):
        return torch.tensor(self.labels_vector)
    def descriptions_tensor(self):
        return torch.tensor(self.descriptions_vector)
    def aliases_tensor(self):
        return torch.tensor(self.aliases_text)
    def pages_tensor(self):
        return torch.tensor(self.pages_vector)
    def claims_tensor(self):
        return torch.tensor(self.claims_vector)
    def category_tensor(self):
        return torch.tensor(self.subcategory_scalar)
    def type_tensor(self):
        return torch.tensor(self.type_boolean)