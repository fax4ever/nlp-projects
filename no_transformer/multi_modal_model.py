import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin

def rescale_vector_layer(params):
    in_features, out_features = params
    # frequency vector fields rescaling (applying also a RuLU individually):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

class MultiModalModel(nn.Module, PyTorchModelHubMixin,
                      repo_url="fax4ever/culturalitems-no-transformer",
                      pipeline_tag="text-classification",
                      license="apache-2.0"):
    def __init__(self, params):
        super(MultiModalModel, self).__init__()
        # individual input layers for frequency vectors
        self.desc = rescale_vector_layer(params['desc'])
        self.wiki = rescale_vector_layer(params['wiki'])
        self.labels = rescale_vector_layer(params['labels'])
        self.descriptions = rescale_vector_layer(params['descriptions'])
        self.aliases = rescale_vector_layer(params['aliases'])
        self.pages = rescale_vector_layer(params['pages'])
        self.claims = rescale_vector_layer(params['claims'])
        # individual input layers for scalar value
        self.category = nn.Linear(params['category_dim'], params['category_scale'])
        self.type_proj = nn.Linear(params['type_dim'], params['type_scale'])
        # glove
        self.desc_glove = rescale_vector_layer(params['desc_glove'])
        # common classifier
        self.classifier = nn.Sequential(
            nn.Linear(params['total_scale'], params['hidden_layers']),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(params['hidden_layers'], 3)
        )

    def to_device(self, device):
        self.device = device
        self.desc = self.desc.to(device)
        self.wiki = self.wiki.to(device)
        self.labels = self.labels.to(device)
        self.descriptions = self.descriptions.to(device)
        self.aliases = self.aliases.to(device)
        self.pages = self.pages.to(device)
        self.claims = self.claims.to(device)
        self.category = self.category.to(device)
        self.type_proj = self.type_proj.to(device)
        self.desc_glove = self.desc_glove.to(device)
        self.classifier = self.classifier.to(device)

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
        desc_glove_feat = self.desc_glove(dataset_items['desc_glove'].to(self.device))
        combined = torch.cat([desc_feat, desc_glove_feat, wiki_feat, labels_feat, descriptions_feat, aliases_feat, pages_feat,
                              claims_feat, category_feat, type_feat], dim=1)
        return self.classifier(combined)