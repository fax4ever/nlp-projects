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
        self.type_proj = nn.Linear(p.type_dim, p.type_scale)
        # common classifier
        self.classifier = nn.Sequential(
            nn.Linear(p.total_scale(), p.hidden_layers),
            nn.ReLU(),
            nn.Dropout(p.dropout),
            nn.Linear(p.hidden_layers, 3)
        )

    def forward(self, dataset_items, device):
        desc_feat = self.desc(dataset_items['desc'].to(device))
        wiki_feat = self.wiki(dataset_items['wiki'].to(device))
        labels_feat = self.labels(dataset_items['labels'].to(device))
        descriptions_feat = self.descriptions(dataset_items['descriptions'].to(device))
        aliases_feat = self.aliases(dataset_items['aliases'].to(device))
        pages_feat = self.pages(dataset_items['pages'].to(device))
        claims_feat = self.claims(dataset_items['claims'].to(device))
        category_feat = self.category(dataset_items['category'].to(device))
        type_feat = self.type_proj(dataset_items['type'].to(device))
        combined = torch.cat([desc_feat, wiki_feat, labels_feat, descriptions_feat, aliases_feat, pages_feat,
                              claims_feat, category_feat, type_feat], dim=1)
        return self.classifier(combined)