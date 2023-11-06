from typing import Callable

import torchvision
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10

from .base_datasets import BaseDatasetCSI, BaseDatasetOD


class Cifar10SingleClassCSI(BaseDatasetCSI):
    r"""
    Dataset wrapper for CIFAR-10 to extract samples from a single target class and apply CSI specific transformations.

    Args:
        in_root (str)
            Root directory of the CIFAR-10 dataset where dataset exists or will be saved to.
        in_train (bool)
            If True, creates dataset from training set, otherwise from test set.
        in_target_class (int)
            The class label for the target class.
        in_duplication_factor (int)
            Number of times an image is duplicated.
        in_pre_shift_transform (Compose)
            Transformations applied before shifting.
        in_shift_transform (Callable)
            Shift transformations applied to the data.
        in_post_shift_transform (Compose)
            Transformations applied after shifting.
    """

    def __init__(self, in_root: str, in_train: bool, in_target_class: int, in_duplication_factor: int,
                 in_pre_shift_transform: Compose, in_shift_transform: Callable, in_post_shift_transform: Compose):
        super().__init__()
        self.dataset: CIFAR10 = torchvision.datasets.CIFAR10(root=in_root, train=in_train, download=True)
        self.target_class = in_target_class
        self.indices = [idx for idx, label in enumerate(self.dataset.targets) if label == self.target_class]
        self.pre_shift_transform = in_pre_shift_transform
        self.shift_transform = in_shift_transform
        self.post_shift_transform = in_post_shift_transform
        self.duplication_factor = in_duplication_factor


class Cifar10OD(BaseDatasetOD):
    r"""
    Dataset wrapper for CIFAR-10 for outlier detection tasks. Target class samples are assigned a label 0, while all other classes are labeled as 1.

    Args:
        in_root (str)
            Root directory of the CIFAR-10 dataset where dataset exists or will be saved to.
        in_train (bool)
            If True, creates dataset from training set, otherwise from test set.
        in_target_class (int)
            The class label for the target class.
        in_transform (Compose)
            Transformations applied to the images before returning.

    Attributes:
        dataset (Dataset)
            The original CIFAR-10 dataset.
        target_class (int)
            The target class.
        transform (Compose)
            Transformations applied to the images.
    """

    def __init__(self, in_root: str, in_train: bool, in_target_class: int, in_transform: Compose):
        self.dataset = torchvision.datasets.CIFAR10(root=in_root, train=in_train, download=True)
        self.target_class = in_target_class
        self.transform = in_transform


NORMALIZATION_STATS_CIFAR10 = {
    0: {
        'cls_name': 'airplane',
        'mean': [0.5256556272506714, 0.5603305697441101, 0.5889057517051697],
        'std': [0.2502190172672272, 0.24083183705806732, 0.2659754455089569]
    },
    1: {
        'cls_name': 'automobile',
        'mean': [0.47118282318115234, 0.4545295536518097, 0.44719868898391724],
        'std': [0.26806581020355225, 0.26582688093185425, 0.27494576573371887]
    },
    2: {
        'cls_name': 'bird',
        'mean': [0.4892503321170807, 0.4914778172969818, 0.42404502630233765],
        'std': [0.22705493867397308, 0.2209421992301941, 0.24337899684906006]
    },
    3: {
        'cls_name': 'cat',
        'mean': [0.4954816997051239, 0.4564116597175598, 0.4155380427837372],
        'std': [0.2568451464176178, 0.252272367477417, 0.25799524784088135]
    },
    4: {
        'cls_name': 'deer',
        'mean': [0.47159042954444885, 0.4652053713798523, 0.3782070577144623],
        'std': [0.21732863783836365, 0.20652803778648376, 0.21182379126548767]
    },
    5: {
        'cls_name': 'dog',
        'mean': [0.499925434589386, 0.4646367132663727, 0.4165467917919159],
        'std': [0.2504265308380127, 0.2437489628791809, 0.2489451766014099]
    },
    6: {
        'cls_name': 'frog',
        'mean': [0.4700562059879303, 0.43839356303215027, 0.3452188968658447],
        'std': [0.2288840264081955, 0.21856245398521423, 0.22042015194892883]
    },
    7: {
        'cls_name': 'horse',
        'mean': [0.5019589066505432, 0.4798647463321686, 0.416886568069458],
        'std': [0.2430475503206253, 0.24397236108779907, 0.2517147362232208]
    },
    8: {
        'cls_name': 'ship',
        'mean': [0.49022579193115234, 0.5253955125808716, 0.5546857714653015],
        'std': [0.2496257871389389, 0.2406878024339676, 0.251496821641922]
    },
    9: {
        'cls_name': 'truck',
        'mean': [0.4986669421195984, 0.4853416979312897, 0.47807592153549194],
        'std': [0.26805269718170166, 0.26910752058029175, 0.28101733326911926]
    }
}
