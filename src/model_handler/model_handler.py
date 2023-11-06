from typing import Optional, Union

import torch

from src.model_handler.models_od.efficientnet_od import build_effnet
from src.model_handler.models_od.resnet_od import build_resnet


class ModelHandler:
    r"""
    Handles the initialization of supported neural networks.

    Supported networks:
        - 'resnet'
        - 'efficientnet'

    Attributes:
        SUPPORTED_NETWORKS (set of str)
            A set containing names of supported networks.

    Args:
        in_model_name (str)
            Name of the dataset to use.
        in_device (int)
            Batch size for the DataLoader.
        in_checkpoint_path (Union[str, None])
            path to checkpoint or None if no pretrained model should be loaded
        in_duplication_factor (int)
            Number of times an image is expected to be duplicated for CSI.
    """

    SUPPORTED_NETWORKS = {'efficientnet', 'resnet'}

    def __init__(self, in_model_name: str, in_device: str, in_duplication_factor: int = 4, in_checkpoint_path: Optional[Union[str, None]] = False,
                 in_model_scale: int = 0, in_pretrained_imagenet: bool = False):
        if in_model_name not in ModelHandler.SUPPORTED_NETWORKS:
            raise NotImplementedError(f"Network {in_model_name} is not supported. "
                                      f"Supported networks are: {', '.join(ModelHandler.SUPPORTED_NETWORKS)}")

        self.model_name = in_model_name
        self.checkpoint_path = in_checkpoint_path
        self.device = in_device
        self.duplication_factor = in_duplication_factor
        self.model_scale = in_model_scale
        self.pretrained = in_pretrained_imagenet
        self.checkpoint = None
        self._init_model()

        if self.checkpoint_path:
            self._load_weights()

    def _load_weights(self):
        r"""
        Load model weights. Only used for evaluation. To continue training use Trainer() class and supply checkpoint.

        """
        self.checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])

    def _init_model(self):
        r"""
        Initialize the neural network based on the provided model name.

        Returns:
            nn.Module: Initialized neural network model.

        Raises:
            NotImplementedError: If the provided `self.model_name` is not in the SUPPORTED_NETWORKS set.
        """
        if self.model_name == 'efficientnet':
            self.model = build_effnet(self.model_scale, self.duplication_factor, self.pretrained).to(self.device)
        elif self.model_name == 'resnet':
            self.model = build_resnet(self.model_scale, self.duplication_factor, self.pretrained).to(self.device)
        else:
            raise NotImplementedError(f"Network {self.model_name} is not supported")
