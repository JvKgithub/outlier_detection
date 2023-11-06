from typing import Dict

import torch
import torch.nn as nn


class CSIModel(nn.Module):
    r"""
    Adjust classification model for CSI multitask learning: This model adjusts a base classification model by removing its final classification layer
    and adding three distinct heads: one for similarity learning, one for shift classification, and one for out-of-distribution classification.

    Args:
        in_base_model (nn.Module)
            A classification model.
        in_dupcliation_factor (int)
            The number of classes for the shift classification task.
        in_num_classes (int, optional)
            The number of classes for the out-of-distribution classification task. Default is 10.
        in_simclr_dim (int, optional)
            The dimensionality of the similarity task's output. Default is 128.

    Attributes:
    feature_net (nn.Sequential)
        The base model without its classifier.
    sim_layer (nn.Sequential)
        The head for the similarity task.
    shift_cls_layer (nn.Linear)
        The head for the shift classification task.
    ood_cls_layer (nn.Linear)
        The head for the out-of-distribution classification task.
    """

    def __init__(self, in_base_model: nn.Module, in_dupcliation_factor: int, in_num_classes: int = 10, in_simclr_dim: int = 128):
        super().__init__()
        # Keep all layers except the last classifier
        self.feature_net = nn.Sequential(*list(in_base_model.children())[:-1])
        num_neurons = _get_last_layer(in_base_model).in_features

        # Define the three heads
        self.sim_layer = nn.Sequential(
            nn.Linear(num_neurons, out_features=num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, out_features=in_simclr_dim),
        )
        self.shift_cls_layer = nn.Linear(num_neurons, out_features=in_dupcliation_factor)
        self.ood_cls_layer = nn.Linear(num_neurons, out_features=in_num_classes)

    def forward(self, in_image: torch.tensor, in_sim: bool = True, in_shift: bool = True, in_cls: bool = True) \
            -> Dict:
        r"""
        Forward pass for the CSI model.

        Args:
            in_image (torch.Tensor)
                Input image tensor.
            in_sim (bool)
                Flag to indicate whether the similarity branch of the network should be activated.
            in_shift (bool)
                Flag to indicate whether the shift-classification branch of the network should be activated.
            in_cls (bool)
                Flag to indicate whether the general classification branch of the network should be activated.

    Returns:
        - Dict:
            - 'sim' corresponds to the similarity task output of shape (batch_size, in_simcrl_dim).
            - 'shift' corresponds to the shift classification task output of shape (batch_size, in_dupcliation_factor).
            - 'cls'corresponds to the out-of-distribution classification task output of shape (batch_size, in_num_classes).
        """
        features = self.feature_net(in_image)
        features = features.flatten(start_dim=1)  # Flatten the output for the dense layers

        out_predictions = {}
        if in_sim:
            out_predictions['sim'] = self.sim_layer(features)
        if in_shift:
            out_predictions['shift'] = self.shift_cls_layer(features)
        if in_cls:
            out_predictions['cls'] = self.ood_cls_layer(features)

        return out_predictions


def _get_last_layer(in_sequential) -> nn.Module:
    r"""
    Recursively retrieves the last layer of a potentially nested Sequential.

    Args:
        in_sequential (nn.Sequential)
            The input sequential module.

    Returns:
        nn.Module: The last non-Sequential layer.
    """
    out_last_module = list(in_sequential.children())[-1]

    if isinstance(out_last_module, nn.Sequential):
        return _get_last_layer(out_last_module)
    return out_last_module
