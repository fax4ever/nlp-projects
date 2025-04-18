import collections
import numpy as np

class Dictionary:
    def __init__(self):
        self.occurrences = []
        self.unk_token = None
        self.word_to_id = None

    def include(self, tokenized_text):
        self.occurrences.extend(tokenized_text)

    def build(self, max_vocab_size, unk_token):
        self.unk_token = unk_token
        counter = collections.Counter(self.occurrences)
        self.word_to_id = {key: index for index, (key, _) in enumerate(counter.most_common(max_vocab_size - 1))}
        assert unk_token not in self.word_to_id
        self.word_to_id[unk_token] = max_vocab_size - 1
        self.occurrences = None # free memory space

    def length(self):
        return len(self.word_to_id)

    def words_to_vector(self, words):
        vector = np.zeros(self.length(), dtype=np.float32)
        for word in words:
            if word == self.unk_token:
                continue
            if word in self.word_to_id:
                vector[self.word_to_id[word]] = vector[self.word_to_id[word]] + 1
        return vector