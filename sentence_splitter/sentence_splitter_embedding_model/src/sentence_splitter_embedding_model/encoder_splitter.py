from sentence_splitter_embedding_model.encoder_hyperparams import EncoderHyperParams

class EncoderSplitter:

    def __init__(self, train, validation):
        self.train = train
        self.validation = validation
        self.params = EncoderHyperParams()

