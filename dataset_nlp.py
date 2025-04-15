import nltk, string

from dataset import Dataset
from dictonary import Dictionary
from nltk.corpus import stopwords

VOCAB_SIZE = 4_000
UNK = '<UNK>' # the token to be used for out of vocabulary words
nltk.download('stopwords')
stop = set(stopwords.words('english') + list(string.punctuation))

def tokenize(text):
    result = []
    if text is None:
        return []
    for sentence in nltk.sent_tokenize(text.lower()):
        result.extend([i for i in nltk.word_tokenize(sentence) if i not in stop])
    return result

class DatasetNLP(Dataset):
    def __init__(self, training_limit=None, validation_limit=None, force_reload=False):
        super().__init__(training_limit, validation_limit, force_reload)

        self.desc_dictionary = Dictionary()
        self.wiki_dictionary = Dictionary()
        self.training_descriptions = []
        self.training_wiki_text = []
        self.validation_descriptions = []
        self.validation_wiki_text = []
        for entity in self.training_set:
            description_tokenized = tokenize(entity.description)
            wiki_text_tokenized = tokenize(entity.wiki_text)
            self.training_descriptions.append(description_tokenized)
            self.training_wiki_text.extend(wiki_text_tokenized)
            self.desc_dictionary.include(description_tokenized)
            self.wiki_dictionary.include(wiki_text_tokenized)
        for entity in self.validation_set:
            description_tokenized = tokenize(entity.description)
            wiki_text_tokenized = tokenize(entity.wiki_text)
            self.validation_descriptions.append(description_tokenized)
            self.validation_wiki_text.extend(wiki_text_tokenized)
            self.desc_dictionary.include(description_tokenized)
            self.wiki_dictionary.include(wiki_text_tokenized)

        self.desc_dictionary.build(VOCAB_SIZE, UNK)
        self.wiki_dictionary.build(VOCAB_SIZE, UNK)
