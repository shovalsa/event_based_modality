from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
import sys
import numpy as np

@Metric.register("mod_tagging_accuracy")
class TaggingAccuracy(Metric):
    """
    Sequence Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    """

    def __init__(self) -> None:
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        threshold: float=0.5
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, k, sequence_length).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, sequence_length).
        mask : `torch.BoolTensor`, optional (default = None).
            A masking tensor the same size as `gold_labels`.
        """
        for pred, gold in zip(predictions, gold_labels):
            for p, g in zip(pred, gold):
                if p != 'O' and p == g:
                    print('yay')
                    print(pred)
                    print(gold)
                    self.TP += 1
                elif p != 'O':
                    print('lol')
                    print(pred)
                    self.FP += 1
                elif g != 'O':
                    self.FN += 1


    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated accuracy.
        """
        if self.TP == 0:
            f1 = 0.
        else:
            precision = self.TP / (self.TP + self.FP)
            recall = self.TP / (self.TP + self.FN)
            f1 = 2.0 / ((1/recall) + (1/precision))
        return {"tagging_f1": f1}

    @overrides
    def reset(self):
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0