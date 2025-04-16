import nltk, string

from dataset import Dataset
from dictonary import Dictionary
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from processed_entity import ProcessedEntity

VOCAB_SIZE = 4_000
UNK = '<UNK>' # the token to be used for out of vocabulary words
nltk.download('stopwords')
stop = set(stopwords.words('english') + list(string.punctuation))

def tokenize(text):
    result = []
    if text is None:
        return result
    for sentence in nltk.sent_tokenize(text.lower()):
        result.extend([WordNetLemmatizer().lemmatize(i) for i in nltk.word_tokenize(sentence) if i not in stop])
    return result

class ProcessedDataset(Dataset):
    def __init__(self, training_limit=None, validation_limit=None, force_reload=False):
        super().__init__(training_limit, validation_limit, force_reload)
        # descriptions and wiki text words are in 2 different vector spaces
        self.desc_dictionary = Dictionary()
        self.wiki_dictionary = Dictionary()
        # from the base data, add a list of processed entities
        self.processed_training_set = [self.process(entity) for entity in self.training_set]
        self.processed_validation_set = [self.process(entity) for entity in self.validation_set]
        # when we've collected all the words for the two spaces, we can build them
        self.desc_dictionary.build(VOCAB_SIZE, UNK)
        self.wiki_dictionary.build(VOCAB_SIZE, UNK)

    def process(self, entity):
        description_tokenized = tokenize(entity.description)
        wiki_text_tokenized = tokenize(entity.wiki_text)
        self.desc_dictionary.include(description_tokenized)
        self.wiki_dictionary.include(wiki_text_tokenized)
        return ProcessedEntity(entity, description_tokenized, wiki_text_tokenized)
