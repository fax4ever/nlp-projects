import torch
import torch.nn as nn
from typing import Union, Any, Optional
from transformers import Trainer
      

class WeightedTrainer(Trainer):
    def compute_loss(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], return_outputs: bool = False, num_items_in_batch: Optional[torch.Tensor] = None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 2 labels with different weights
        # Simple class weights: give sentence endings 30x more importance
        # Based on your data: 96.7% class 0, 3.3% class 1 → ~30:1 ratio
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 30.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        return (loss, outputs) if return_outputs else loss   
