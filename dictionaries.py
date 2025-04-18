from dictonary import Dictionary
from processed_entity import ProcessedEntity

UNK = '<UNK>' # the token to be used for out of vocabulary words
DESC_VOCAB_SIZE = 4_000
WIKI_VOCAB_SIZE = 10_000

class Dictionaries:
    def __init__(self):
        # descriptions and wiki text words are in 2 different vector spaces
        self.desc = Dictionary()
        self.wiki = Dictionary()

    def include(self, processed_entity: ProcessedEntity):
        self.desc.include(processed_entity.description)
        self.wiki.include(processed_entity.wiki_text)

    def build(self):
        self.desc.build(DESC_VOCAB_SIZE, UNK)
        self.wiki.build(WIKI_VOCAB_SIZE, UNK)

    def finalize(self, processed_entity: ProcessedEntity):
        processed_entity.desc_vector = self.desc.words_to_vector(processed_entity.description)
        processed_entity.wiki_vector = self.wiki.words_to_vector(processed_entity.wiki_text)