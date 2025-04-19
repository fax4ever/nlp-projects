from torch import nn
from nlp_hyper_params import NLPHyperParams

def rescale_vector_layer(params):
    in_features, out_features = params
    # frequency vector fields rescaling (applying also a RuLU individually):
    return nn.Sequential(nn.Linear(in_features, out_features),nn.ReLU())

class MultiModalModel(nn.Module):
    def __init__(self, p : NLPHyperParams) -> None:
        super(MultiModalModel, self).__init__()

        # individual input layers for frequency vectors
        self.desc = rescale_vector_layer(p.desc())
        self.wiki = rescale_vector_layer(p.wiki())
        self.labels = rescale_vector_layer(p.labels())
        self.descriptions = rescale_vector_layer(p.descriptions())
        self.aliases = rescale_vector_layer(p.aliases())
        self.pages = rescale_vector_layer(p.claims())
        self.claims = rescale_vector_layer(p.claims())

        # In progress ...
        # self.category_embedding = nn.Embedding(category_vocab_size, 32)
        # self.category_proj = nn.Linear(32, common_dim)