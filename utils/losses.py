# https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/joint_loss.py
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


__all__ = ["JointLoss", "WeightedLoss", "MSELoss", "SoftBCELoss"]


class WeightedLoss(_Loss):

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight

class JointLoss(_Loss):

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)

class MSELoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor = None) -> Tensor:
        loss = F.mse_loss(y_pred, y_true, reduction="none")

        if mask is not None:
            loss *= mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class RMSELoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean', eps=1e-6) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor = None) -> Tensor:
        loss = torch.sqrt(F.mse_loss(y_pred, y_true, reduction="none") + self.eps)

        if mask is not None:
            loss *= mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class SoftBCELoss(nn.Module):

    __constants__ = [
        "weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy(y_pred, soft_targets, weight=self.weight, reduction="none")

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss

