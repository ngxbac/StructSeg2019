from functools import partial

import numpy as np
import torch
import torch.nn as nn
from catalyst.dl.utils import criterion
from catalyst.utils import get_activation_fn


class MultiDiceLoss(nn.Module):
    def __init__(
        self,
        activation: str = "Softmax",
        num_classes: int = 7,
        weight = None,
        dice_weight: float = 0.3,
    ):
        super().__init__()
        if weight is not None:
            weight = torch.from_numpy(np.asarray(weight).astype(np.float32))
        else:
            weight = None
        self.num_classes = num_classes
        self.activation = activation
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_loss = criterion.dice

    def forward(self, logits, targets):
        activation_fnc = get_activation_fn(self.activation)
        logits_softmax = activation_fnc(logits)

        ce_loss = self.ce_loss(logits, targets)

        dice_loss = 0
        for cls in range(self.num_classes):
            targets_cls = (targets == cls).float()
            outputs_cls = logits_softmax[:, cls]
            score = 1 - criterion.dice(outputs_cls, targets_cls, eps=1e-7, activation='none', threshold=None)
            dice_loss += score / self.num_classes

        loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
        return loss
