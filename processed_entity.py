class ProcessedEntity:
    def __init__(self, base_entity, description, wiki_text):
        # extension by composition
        self.base_entity = base_entity
        # processed fields
        self.description = description
        self.wiki_text = wiki_text

    def __str__(self):
        return str(self.base_entity) + " < " + str(len(self.description)) + ", " + str(len(self.wiki_text)) + " >"