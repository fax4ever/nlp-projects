from entity import Entity
import numpy as np

def type_vector(base_type):
    vector = np.zeros(1, dtype=np.float32)
    if base_type == 'entity':
        vector[0] = vector[0] + 1
    return vector

def label_to_number(label):
    if label == 'cultural agnostic':
        return 0
    if label == 'cultural representative':
        return 1
    if label == 'cultural exclusive':
        return 2
    raise ValueError('label not suppoerted: ' + label)

class ProcessedEntity:
    def __init__(self, base: Entity, desc_text, wiki_text):
        self.base_entity = base.entity_id + ": " + base.name
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
        self.subcategory_vector = None
        # in this case we can assume that we have only two types (entity vs concept)
        self.type_vector = type_vector(base.type)
        self.desc_glove_vector = None
        if hasattr(base, 'label'):
            self.output_label = label_to_number(base.label)

    def __str__(self):
        return self.base_entity + " < " + str(len(self.desc_text)) + ", " + str(len(self.wiki_text)) + " >"

    def dataset_item(self):
        return {
            "desc" : self.desc_vector,
            "wiki" : self.wiki_vector,
            "labels" : self.labels_vector,
            "descriptions" : self.descriptions_vector,
            "aliases" : self.aliases_vector,
            "pages" : self.pages_vector,
            "claims" : self.claims_vector,
            "category" : self.subcategory_vector,
            "type" : self.type_vector,
            "desc_glove" : self.desc_glove_vector,
            "output_label" : self.output_label,
            "base" : self.base_entity
        }