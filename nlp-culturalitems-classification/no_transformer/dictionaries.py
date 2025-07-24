from dictonary import Dictionary
from glove_processing import GloveProcessing
from processed_entity import ProcessedEntity
from category_table import CategoryTable

UNK = '<UNK>' # the token to be used for out of vocabulary words
DESC_VOCAB_SIZE = 4_000
WIKI_VOCAB_SIZE = 10_000
GLOVE_EMBEDDING_SIZE = 20

class Dictionaries:
    def __init__(self):
        # descriptions and wiki text words are in 2 different vector spaces
        self.desc = Dictionary()
        self.wiki = Dictionary()
        # we use the same languages keys dictionaries for:
        # labels_text, descriptions_text, aliases_text, pages_text
        self.languages = Dictionary()
        self.claims = Dictionary()
        self.category_table = CategoryTable()
        # extra add glove embeddings
        self.glove_desc = GloveProcessing(GLOVE_EMBEDDING_SIZE)

    def include(self, processed_entity: ProcessedEntity):
        self.desc.include(processed_entity.desc_text)
        self.wiki.include(processed_entity.wiki_text)
        self.languages.include(processed_entity.labels_text)
        self.languages.include(processed_entity.descriptions_text)
        self.languages.include(processed_entity.aliases_text)
        self.languages.include(processed_entity.pages_text)
        self.claims.include(list(processed_entity.claims_map.keys()))
        self.category_table.include(processed_entity)

    def build(self):
        self.desc.build(DESC_VOCAB_SIZE, UNK)
        self.wiki.build(WIKI_VOCAB_SIZE, UNK)
        self.claims.build_no_limits()
        # those guys are not too large: so we can not limit them
        self.languages.build_no_limits()
        self.category_table.build()

    def finalize(self, processed_entity: ProcessedEntity):
        processed_entity.desc_vector = self.desc.words_to_vector(processed_entity.desc_text)
        processed_entity.wiki_vector = self.wiki.words_to_vector(processed_entity.wiki_text)
        processed_entity.labels_vector = self.languages.words_to_vector(processed_entity.labels_text)
        processed_entity.descriptions_vector = self.languages.words_to_vector(processed_entity.descriptions_text)
        processed_entity.aliases_vector = self.languages.words_to_vector(processed_entity.aliases_text)
        processed_entity.pages_vector = self.languages.words_to_vector(processed_entity.pages_text)
        processed_entity.claims_vector = self.claims.map_to_vector(processed_entity.claims_map)
        processed_entity.subcategory_vector = self.category_table.subcat_to_vector(processed_entity.subcategory)
        processed_entity.desc_glove_vector = self.glove_desc.words_to_vect(processed_entity.desc_text)