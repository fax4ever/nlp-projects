from entity import Entity

class ProcessedEntity:
    def __init__(self, base: Entity, desc_text, wiki_text):
        self.base_entity = str(base)
        # processed fields
        self.desc_text = desc_text
        self.wiki_text = wiki_text
        self.label_text = base.label
        self.descriptions_text = base.descriptions
        self.aliases_text = base.aliases
        self.pages_text = base.wikipedia_pages
        # Using map to denote a Python dictionary,
        # since the dictionary is already use for a word (term) dictionary
        self.claims_map = base.claims

        # build later (then the dictionaries are finalized)
        self.desc_vector = None
        self.wiki_vector = None
        self.label_vector = None
        self.descriptions_vector = None
        self.aliases_vector = None
        self.pages_vector = None
        self.claims_vector = None

    def __str__(self):
        return self.base_entity + " < " + str(len(self.desc_text)) + ", " + str(len(self.wiki_text)) + " >"