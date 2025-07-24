from torchtext.vocab import GloVe

PAD_TOKEN = '<PAD>'

class GloveProcessing:
    def __init__(self, context_size):
        self.glove = GloVe(name='6B', dim=100)
        self.context_size = context_size

    def words_to_vect(self, words):
        return self.glove.get_vecs_by_tokens(self.tokens(words)).view(-1)

    def tokens(self, words):
        return words[:self.context_size] + [PAD_TOKEN]*(self.context_size-len(words))

if __name__ == "__main__":
    # example of usages
    glove_processing = GloveProcessing(3)
    bla = glove_processing.words_to_vect(['cat', 'logic', 'legion'])
    print(bla)
    bla = glove_processing.words_to_vect(['cat', 'logic'])
    print(bla)
    bla = glove_processing.words_to_vect(['cat', 'logic', 'legion', 'red'])
    print(bla)