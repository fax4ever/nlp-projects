import torch
from torch import nn
from nlp_hyper_params import NLPHyperParams
from huggingface_hub import PyTorchModelHubMixin

def rescale_vector_layer(params):
    in_features, out_features = params
    # frequency vector fields rescaling (applying also a RuLU individually):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

class MultiModalModel(nn.Module, PyTorchModelHubMixin,
                      repo_url="fax4ever/culturalitems-no-transformer",
                      pipeline_tag="text-classification",
                      license="apache-2.0"):
    def __init__(self, p : NLPHyperParams, device) -> None:
        super(MultiModalModel, self).__init__()
        self.device = device
        # individual input layers for frequency vectors
        self.desc = rescale_vector_layer(p.desc()).to(device)
        self.wiki = rescale_vector_layer(p.wiki()).to(device)
        self.labels = rescale_vector_layer(p.labels()).to(device)
        self.descriptions = rescale_vector_layer(p.descriptions()).to(device)
        self.aliases = rescale_vector_layer(p.aliases()).to(device)
        self.pages = rescale_vector_layer(p.pages()).to(device)
        self.claims = rescale_vector_layer(p.claims()).to(device)
        # individual input layers for scalar value
        self.category = nn.Linear(p.category_dim, p.category_scale).to(device)
        self.type_proj = nn.Linear(p.type_dim, p.type_scale).to(device)
        # common classifier
        self.classifier = nn.Sequential(
            nn.Linear(p.total_scale(), p.hidden_layers),
            nn.ReLU(),
            nn.Dropout(p.dropout),
            nn.Linear(p.hidden_layers, 3)
        ).to(device)

    def forward(self, dataset_items):
        desc_feat = self.desc(dataset_items['desc'].to(self.device))
        wiki_feat = self.wiki(dataset_items['wiki'].to(self.device))
        labels_feat = self.labels(dataset_items['labels'].to(self.device))
        descriptions_feat = self.descriptions(dataset_items['descriptions'].to(self.device))
        aliases_feat = self.aliases(dataset_items['aliases'].to(self.device))
        pages_feat = self.pages(dataset_items['pages'].to(self.device))
        claims_feat = self.claims(dataset_items['claims'].to(self.device))
        category_feat = self.category(dataset_items['category'].to(self.device))
        type_feat = self.type_proj(dataset_items['type'].to(self.device))
        combined = torch.cat([desc_feat, wiki_feat, labels_feat, descriptions_feat, aliases_feat, pages_feat,
                              claims_feat, category_feat, type_feat], dim=1)
        return self.classifier(combined)