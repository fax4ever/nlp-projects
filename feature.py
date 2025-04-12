class Feature:
    def __init__(self, entity_id, train, entity):
        self.entity_id = entity_id
        self.train = train
        self.entity = entity

    def __str__(self):
        return self.entity_id + " " + str(self.train) + " " + str(self.entity)

