import torch.nn as nn
import torchvision.models as models

from .csi_model import CSIModel


def build_effnet(in_index: int, in_duplication_factor: int, in_pretrained: bool = True) -> nn.Module:
    r"""
    Builds the custom EfficientNetCSI model for CSI tasks.

    Args:
        in_index (int)
            Index of the EfficientNet version (0 for B0, 1 for B1, ... 7 for B7). Special index 8 for small input images such as CIFAR, for which B0
            is initialized and two downsampling operations are removed.
        in_duplication_factor (int)
            Number of times an image is expected to be duplicated for CSI.
        pretrained (bool, optional)
            If True, initializes the EfficientNet base with weights pre-trained on ImageNet. Default is True.

    Returns:
        nn.Module: The EfficientNetCSI model.
    """
    base_model = _get_efficientnet_base(in_index, in_pretrained)
    out_efficientnet_csi = CSIModel(base_model, in_duplication_factor)

    # For small data like CIFAR, even B0 has too much downsampling, reduce the first and last downsampling convs to stride (1,1)
    if in_index == 8:
        out_efficientnet_csi.feature_net[0][0][0].stride = (1, 1)
        out_efficientnet_csi.feature_net[0][6][0].block[1][0].stride = (1, 1)
    return out_efficientnet_csi


def _get_efficientnet_base(in_index: int, in_pretrained=True) -> nn.Module:
    r"""
    Get the scaled version of EfficientNet based on the provided index.

    Args:
        in_index (int)
            Index of the EfficientNet version (0 for B0, 1 for B1, ... 7 for B7). Special index 8 for small input images such as CIFAR, for which B0
            is initialized and two downsampling operations are removed.
        in_pretrained (bool)
            If True, returns a model pre-trained on ImageNet.

    Returns:
        model (nn.Module): EfficientNet model of the specified version.
    """
    if in_index < 0 or in_index > 8:
        raise ValueError("Index should be between 0 and 8 (inclusive) for EfficientNetB0 to EfficientNetB7 and EfficientNetB0_small_images.")

    # List of EfficientNet scales
    efficientnet_versions = [
        models.efficientnet_b0, models.efficientnet_b1, models.efficientnet_b2, models.efficientnet_b3,
        models.efficientnet_b4, models.efficientnet_b5, models.efficientnet_b6, models.efficientnet_b7,
        models.efficientnet_b0
    ]

    # List of EfficientNet weights
    efficientnet_weights = [
        models.efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1, models.efficientnet.EfficientNet_B1_Weights.IMAGENET1K_V1,
        models.efficientnet.EfficientNet_B2_Weights.IMAGENET1K_V1, models.efficientnet.EfficientNet_B3_Weights.IMAGENET1K_V1,
        models.efficientnet.EfficientNet_B4_Weights.IMAGENET1K_V1, models.efficientnet.EfficientNet_B5_Weights.IMAGENET1K_V1,
        models.efficientnet.EfficientNet_B6_Weights.IMAGENET1K_V1, models.efficientnet.EfficientNet_B7_Weights.IMAGENET1K_V1,
        models.efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ]

    weights = efficientnet_weights[in_index] if in_pretrained else None
    out_efficientnet_base = efficientnet_versions[in_index](weights=weights)
    return out_efficientnet_base
