import torchcrf
import torch.nn as nn
from transformers import AutoModel

class NLUModel(nn.Module):
    def __init__(self, intent_classes, entity_classes):
        super(NLUModel, self).__init__()
        self.base_model = AutoModel.from_pretrained("bert-base-uncased")

        # Freeze BERT layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        base_model_output_size = self.base_model.config.hidden_size

        # Intent Classification Branch
        self.intent_branch = nn.Sequential(
            nn.Linear(base_model_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, intent_classes)
        )

        # Named Entity Recognition Branch
        self.entity_linear = nn.Sequential(
            nn.Linear(base_model_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, entity_classes)
        )
        self.crf_layer = torchcrf.CRF(entity_classes)  # Create a CRF layer

    def forward(self, x, attention_mask):
        base_model_output = self.base_model(x, attention_mask=attention_mask)

        # Intent Classification
        intent_output = self.intent_branch(base_model_output.last_hidden_state[:, 0, :])

        # Named Entity Recognition
        entity_scores = self.entity_linear(base_model_output.last_hidden_state)

        return intent_output, entity_scores  # Return intent output and entity scores


    
    

class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        super(CRFLayer, self).__init__()
        self.crf = torchcrf.CRF(num_tags)  # Create a CRF layer

    def forward(self, emissions, tags, mask=None):
        # Calculate the CRF loss
        loss = -self.crf(emissions, tags, mask=mask)

        return loss

