import torch
from torch import nn
from nlp_hyper_params import NLPHyperParams
from processed_entity import ProcessedEntity


def rescale_vector_layer(params):
    in_features, out_features = params
    # frequency vector fields rescaling (applying also a RuLU individually):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

class MultiModalModel(nn.Module):
    def __init__(self, p : NLPHyperParams) -> None:
        super(MultiModalModel, self).__init__()
        # individual input layers for frequency vectors
        self.desc = rescale_vector_layer(p.desc())
        self.wiki = rescale_vector_layer(p.wiki())
        self.labels = rescale_vector_layer(p.labels())
        self.descriptions = rescale_vector_layer(p.descriptions())
        self.aliases = rescale_vector_layer(p.aliases())
        self.pages = rescale_vector_layer(p.pages())
        self.claims = rescale_vector_layer(p.claims())
        # individual input layers for scalar value
        self.category = nn.Linear(p.category_dim, p.category_scale)
        self.type = nn.Linear(p.type_dim, p.type_scale)
        # common classifier
        self.classifier = nn.Sequential(
            nn.Linear(p.total_scale(), p.hidden_layers),
            nn.ReLU(),
            nn.Dropout(p.dropout),
            nn.Linear(p.hidden_layers, 3)
        )

    def forward(self, processed_entity: ProcessedEntity):
        desc_feat = self.desc(processed_entity.desc_tensor())
        wiki_feat = self.wiki(processed_entity.wiki_tensor())
        labels_feat = self.labels(processed_entity.labels_tensor())
        descriptions_feat = self.descriptions(processed_entity.descriptions_tensor())
        aliases_feat = self.aliases(processed_entity.aliases_tensor())
        pages_feat = self.pages(processed_entity.pages_tensor())
        claims_feat = self.claims(processed_entity.claims_tensor())
        category_feat = self.category(processed_entity.category_tensor())
        type_feat = self.type(processed_entity.type_tensor())
        combined = torch.cat([desc_feat, wiki_feat, labels_feat, descriptions_feat, aliases_feat, pages_feat,
                              claims_feat, category_feat, type_feat], dim=1)
        return self.classifier(combined)