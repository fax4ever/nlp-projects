class Feature:
    def __init__(self, entity_id, train, entity, wiki_text):
        self.entity_id = entity_id
        self.train = train
        self.entity = entity
        self.wiki_text = wiki_text

    def __str__(self):
        return self.entity_id + " " + str(self.train) + " " + str(self.entity) + " " + self.wiki_text[:1000]

