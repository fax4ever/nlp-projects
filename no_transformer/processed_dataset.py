import nltk, string, os, pickle

from file_util import dump, load
from nlp_dataset import NLPDataset
from dictionaries import Dictionaries
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from iterable_entities import IterableEntities
from processed_entity import ProcessedEntity

TRAINING_PROC_FILE_NAME = "training-proc.bin"
VALIDATION_PROC_FILE_NAME = "validation-proc.bin"

def text_process(text, stop):
    result = []
    if text is None:
        return result
    for sentence in nltk.sent_tokenize(text.lower()):
        result.extend([WordNetLemmatizer().lemmatize(i) for i in nltk.word_tokenize(sentence) if i not in stop])
    return result

def create_processed(entity, dictionaries, stop):
    description_tokenized = text_process(entity.description, stop)
    wiki_text_tokenized = text_process(entity.wiki_text, stop)
    result = ProcessedEntity(entity, description_tokenized, wiki_text_tokenized)
    dictionaries.include(result)
    return result

class ProcessedDataset(NLPDataset):
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
        nltk.download('stopwords')
        nltk.download('wordnet')
        stop = set(stopwords.words('english') + list(string.punctuation) + ['==', "''", '``', "'s", '==='])
        print("processing the data")
        dictionaries = Dictionaries()
        # from the base data, add a list of processed entities
        print("training set text processing started")
        processed_training_set = []
        for index, entity in enumerate(self.training_set):
            processed_training_set.append(create_processed(entity, dictionaries, stop))
            if (index+1) % 100 == 0:
                print("training set processed", index+1, "entities")
        print("training set text processing ended")
        print("validation set text processing started")
        processed_validation_set = []
        for index, entity in enumerate(self.validation_set):
            processed_validation_set.append(create_processed(entity, dictionaries, stop))
            if (index+1) % 100 == 0:
                print("validation set processed", index+1, "entities")
        print("validation set text processing ended")
        print("building dictionaries")
        # when we've collected all the words for the two spaces, we can build them
        dictionaries.build()
        print("text to vector started")
        # build the vectors from the texts
        for entity in processed_training_set:
            dictionaries.finalize(entity)
        for entity in processed_validation_set:
            dictionaries.finalize(entity)
        print("text to vector finished")
        return processed_training_set, processed_validation_set

    def training(self):
        return IterableEntities(self.processed_training_set)

    def validation(self):
        return IterableEntities(self.processed_validation_set)
