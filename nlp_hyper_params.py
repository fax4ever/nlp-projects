from processed_entity import ProcessedEntity

class NLPHyperParams:
    def __init__(self):
        # processed fields
        self.desc_dim = None
        self.desc_scale = 64
        self.wiki_dim = None
        self.wiki_scale = 64
        self.labels_dim = None
        self.labels_scale = 8
        self.descriptions_dim = None
        self.descriptions_scale = 8
        self.aliases_dim = None
        self.aliases_scale = 8
        self.pages_dim = None
        self.pages_scale = 8
        self.claims_dim = None
        self.claims_scale = 16
        self.category_dim = 1 # scalar
        self.category_scale = 8
        self.type_dim = 1 # scalar
        self.type_scale = 8
        # common classifier
        self.hidden_layers = 128
        self.dropout = 0.2

    def compute(self, example: ProcessedEntity):
        self.desc_dim = example.desc_vector.shape[0]
        self.wiki_dim = example.wiki_vector.shape[0]
        self.labels_dim = example.labels_vector.shape[0]
        self.descriptions_dim = example.descriptions_vector.shape[0]
        self.aliases_dim = example.aliases_vector.shape[0]
        self.pages_dim = example.pages_vector.shape[0]
        self.claims_dim = example.claims_vector.shape[0]

    def desc(self):
        return self.desc_dim, self.desc_scale
    def wiki(self):
        return self.desc_dim, self.desc_scale
    def labels(self):
        return self.labels_dim, self.labels_scale
    def descriptions(self):
        return self.descriptions_dim, self.descriptions_scale
    def aliases(self):
        return self.aliases_dim, self.aliases_scale
    def pages(self):
        return self.pages_dim, self.pages_scale
    def claims(self):
        return self.claims_dim, self.claims_scale
    def total_scale(self):
        return self.desc_scale + self.wiki_scale + self.labels_scale + self.descriptions_scale + self.aliases_scale + \
            self.pages_scale + self.claims_scale + self.category_scale + self.type_scale
