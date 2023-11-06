import torch.nn as nn
import torchvision.models as models

from .csi_model import CSIModel


def build_resnet(in_index: int, in_duplication_factor: int, in_pretrained: bool = True) -> nn.Module:
    r"""
    Builds the custom ResNetCSI model for CSI tasks.

    Args:
        in_index (int)
            Index of the ResNet version (0 for ResNet-18, 1 for ResNet-50, 2 for ResNet-101, 3 for ResNet-152). Special index 4 for small input images
            such as CIFAR, for which ResNet-18 is initialized and two downsampling operations are removed.
        in_duplication_factor (int)
            Number of times an image is expected to be duplicated for CSI.
        in_pretrained (bool, optional)
            If True, initializes the ResNet base with weights pre-trained on ImageNet. Default is True.

    Returns:
        nn.Module: The ResNetCSI model.
    """
    base_model = _get_resnet_base(in_index, in_pretrained)
    out_resnet_csi = CSIModel(base_model, in_duplication_factor)

    # For small data like CIFAR, even ResNet-18 has too much downsampling, reduce the first conv stride to (1,1) and remove first max pool layer
    if in_index == 5:
        out_resnet_csi.feature_net[0].stride = (1, 1)
        layers = list(out_resnet_csi.feature_net.children())
        layers.pop(3)
        out_resnet_csi.feature_net = nn.Sequential(*layers)
    return out_resnet_csi


def _get_resnet_base(in_index: int, in_pretrained=True) -> nn.Module:
    r"""
    Get the scaled version of ResNet based on the provided index.

    Args:
        in_index (int)
            Index of the ResNet version (0 for ResNet-18, 1 for ResNet-18, 2 for ResNet-50, 3 for ResNet-101, 4 for ResNet-152). Special index 5 for
            small input images such as CIFAR, for which ResNet-18 is initialized and two downsampling operations are removed.
        in_pretrained (bool)
            If True, returns a model pre-trained on ImageNet.

    Returns:
        model (nn.Module): ResNet model of the specified version.
    """
    if in_index < 0 or in_index > 5:
        raise ValueError("Index should be between 0 and 5 (inclusive) for ResNet-18 to ResNet-152 and ResNet-18_small_images.")

    # List of ResNet scales
    resnet_versions = [
        models.resnet18, models.resnet34, models.resnet50, models.resnet101, models.resnet152,
        models.resnet18,
    ]
    # List of ResNet weights
    resnet_weight_versions = [
        models.resnet.ResNet18_Weights.IMAGENET1K_V1, models.resnet.ResNet34_Weights.IMAGENET1K_V1, models.resnet.ResNet50_Weights.IMAGENET1K_V1,
        models.resnet.ResNet101_Weights.IMAGENET1K_V1, models.resnet.ResNet152_Weights.IMAGENET1K_V1,
        models.resnet.ResNet18_Weights.IMAGENET1K_V1,
    ]

    weights = resnet_weight_versions[in_index] if in_pretrained else None
    out_resnet_base = resnet_versions[in_index](weights=weights)
    return out_resnet_base
