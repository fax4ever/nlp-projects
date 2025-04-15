import collections

class Dictionary:
    def __init__(self):
        self.words = []
        self.word2id = None
        self.word2frequency = None
        self.tot_occurrences = None
        self.id2word = None

    def include(self, tokenized_text):
        self.words.extend(tokenized_text)

    def build(self, max_vocab_size, unk_token):
        counter = collections.Counter(self.words)
        self.word2id = {key: index for index, (key, _) in enumerate(counter.most_common(max_vocab_size - 1))}
        assert unk_token not in self.word2id
        self.word2id[unk_token] = max_vocab_size - 1
        # word -> frequency
        self.word2frequency = {x: counter[x] for x in self.word2id if x is not unk_token}
        self.tot_occurrences = sum(self.word2frequency[x] for x in self.word2frequency)
        # id -> word
        self.id2word = {value: key for key, value in self.word2id.items()}

    def length(self):
        return len(self.word2id)