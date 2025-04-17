import nltk, string, os, pickle

from file_util import dump, load
from dataset import Dataset
from dictonary import Dictionary
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from processed_entity import ProcessedEntity

TRAINING_PROC_FILE_NAME = "training-proc.bin"
VALIDATION_PROC_FILE_NAME = "validation-proc.bin"
DESC_VOCAB_SIZE = 4_000
WIKI_VOCAB_SIZE = 10_000
UNK = '<UNK>' # the token to be used for out of vocabulary words
nltk.download('stopwords')
nltk.download('wordnet')
stop = set(stopwords.words('english') + list(string.punctuation))

def text_process(text):
    result = []
    if text is None:
        return result
    for sentence in nltk.sent_tokenize(text.lower()):
        result.extend([WordNetLemmatizer().lemmatize(i) for i in nltk.word_tokenize(sentence) if i not in stop])
    return result

def create_processed(entity, desc_dictionary, wiki_dictionary):
    description_tokenized = text_process(entity.description)
    wiki_text_tokenized = text_process(entity.wiki_text)
    desc_dictionary.include(description_tokenized)
    wiki_dictionary.include(wiki_text_tokenized)
    return ProcessedEntity(entity, description_tokenized, wiki_text_tokenized)

class ProcessedDataset(Dataset):
    def __init__(self, training_limit=None, validation_limit=None, force_reload=False):
        super().__init__(training_limit, validation_limit, force_reload)
        if not (os.path.exists(TRAINING_PROC_FILE_NAME)) or not (os.path.exists(VALIDATION_PROC_FILE_NAME)) or force_reload:
            self.processed_training_set, self.processed_validation_set = self.processing()
            dump(TRAINING_PROC_FILE_NAME, self.processed_training_set)
            dump(VALIDATION_PROC_FILE_NAME, self.processed_validation_set)
        else:
            # by default load the dataset from a local dump
            self.processed_training_set = load(TRAINING_PROC_FILE_NAME)
            self.processed_validation_set = load(VALIDATION_PROC_FILE_NAME)

    def processing(self):
        print("processing the data")
        # descriptions and wiki text words are in 2 different vector spaces
        desc_dictionary = Dictionary()
        wiki_dictionary = Dictionary()
        # from the base data, add a list of processed entities
        print("training set text processing started")
        processed_training_set = [create_processed(entity, desc_dictionary, wiki_dictionary) for entity in self.training_set]
        print("training set text processing ended")
        print("validation set text processing started")
        processed_validation_set = [create_processed(entity, desc_dictionary, wiki_dictionary) for entity in self.validation_set]
        print("validation set text processing ended")
        print("building dictionaries")
        # when we've collected all the words for the two spaces, we can build them
        desc_dictionary.build(DESC_VOCAB_SIZE, UNK)
        wiki_dictionary.build(WIKI_VOCAB_SIZE, UNK)
        print("text to vector started")
        # build the vectors from the texts
        for entity in processed_training_set:
            entity.text_to_vector(desc_dictionary, wiki_dictionary)
        for entity in processed_validation_set:
            entity.text_to_vector(desc_dictionary, wiki_dictionary)
        print("text to vector finished")
        return processed_training_set, processed_validation_set
