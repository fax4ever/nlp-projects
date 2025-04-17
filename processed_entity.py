class ProcessedEntity:
    def __init__(self, base_entity, description, wiki_text):
        self.base_entity = str(base_entity)
        # processed fields
        self.description = description
        self.wiki_text = wiki_text
        # build later (then the dictionaries are finalized)
        self.desc_vector = None
        self.wiki_vector = None

    def __str__(self):
        return self.base_entity + " < " + str(len(self.description)) + ", " + str(len(self.wiki_text)) + " >"

    def text_to_vector(self, desc_dictionary, wiki_dictionary):
        self.desc_vector = desc_dictionary.words_to_vector(self.description)
        self.wiki_vector = wiki_dictionary.words_to_vector(self.wiki_text)