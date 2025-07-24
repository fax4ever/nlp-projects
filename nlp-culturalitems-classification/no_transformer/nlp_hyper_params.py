from processed_entity import ProcessedEntity

class NLPHyperParams:
    def __init__(self, example: ProcessedEntity):
        # original vector dimensions
        self.desc_dim = example.desc_vector.shape[0]
        self.wiki_dim = example.wiki_vector.shape[0]
        self.labels_dim = example.labels_vector.shape[0]
        self.descriptions_dim = example.descriptions_vector.shape[0]
        self.aliases_dim = example.aliases_vector.shape[0]
        self.pages_dim = example.pages_vector.shape[0]
        self.claims_dim = example.claims_vector.shape[0]
        self.category_dim = 1 # scalar
        self.type_dim = 1 # scalar
        self.desc_glove_dim = example.desc_glove_vector.shape[0]

        # scaled dimensions
        self.desc_scale = 64*2
        self.wiki_scale = 64*2
        self.labels_scale = 8*2
        self.descriptions_scale = 8*2
        self.aliases_scale = 8*2
        self.pages_scale = 8*2
        self.claims_scale = 16*2
        self.category_scale = 8*2
        self.type_scale = 8*2
        self.desc_glove_scale = 64*2

        # common classifier
        self.hidden_layers = 128*2
        self.dropout = 0.15
        self.learning_rate = 1e-3
        self.weight_decay = 1e-2
        self.epochs = 20 # manual early stopping
        self.batch_size = 32

    def desc(self):
        return self.desc_dim, self.desc_scale
    def wiki(self):
        return self.wiki_dim, self.wiki_scale
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
    def desc_glove(self):
        return self.desc_glove_dim, self.desc_glove_scale
    def total_scale(self):
        return self.desc_scale + self.wiki_scale + self.labels_scale + self.descriptions_scale + self.aliases_scale + \
            self.pages_scale + self.claims_scale + self.category_scale + self.type_scale + self.desc_glove_scale

    def params(self):
        return {
            'desc': self.desc(),
            'wiki': self.wiki(),
            'labels': self.labels(),
            'descriptions': self.descriptions(),
            'aliases': self.aliases(),
            'pages': self.pages(),
            'claims': self.claims(),
            'desc_glove': self.desc_glove(),
            'total_scale': self.total_scale(),
            'category_dim': self.category_dim,
            'category_scale': self.category_scale,
            'type_dim': self.type_dim,
            'type_scale': self.type_scale,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout
        }
