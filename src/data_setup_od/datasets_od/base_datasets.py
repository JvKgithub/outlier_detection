from typing import Tuple, List, Callable, Union

from abc import ABC
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.datasets import VisionDataset, ImageFolder


class BaseDatasetCSI(Dataset, ABC):
    # Variables to be defined in subclass
    dataset: Dataset
    indices: List[int]
    pre_shift_transform: Compose
    shift_transform: Callable
    post_shift_transform: Compose
    duplication_factor: int

    def __len__(self) -> int:
        r"""
        Returns the number of samples of the target class in the dataset.

        Returns:
            int: Number of samples of the target class.
        """
        return len(self.indices)

    def __getitem__(self, in_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Retrieves image from dataset based on the provided index, applies the CSI transformations. For detailed explanation check
        https://github.com/alinlab/CSI and https://arxiv.org/pdf/2007.08176.pdf.
        Process in short, assuming duplication factor of 4:
        - image is copied once and pre_shift_transforms are applied (making this augmentation consistence across later shifting transforms),
        e.g. horizontal flip
        - both copies are duplicated three times (resulting in 8 images) and the shifting transform is applied to the three duplicates. The labels
          represent which version of the shift transform was applied.
          e.g. 90, 180, 270 degree rotataion, at this point the two sets of four images are identical
        - independent post_shift_transformation are applied to all images, resulting in four image pairs with different transformations
        e.g. RandomCrop, ColorJitter

        Args:
            in_idx (int): Index of the target class sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed image and its label.
        """
        image, _ = self.dataset[self.indices[in_idx]]

        # Apply pre shift transform
        image_tensor = self.pre_shift_transform(image)
        images_1, images_2 = image_tensor.unsqueeze(0).repeat(2, 1, 1, 1).chunk(2)
        # Duplicate and apply shift transformation
        images_1, labels_1 = self.shift_transform(images_1, self.duplication_factor)
        images_2, labels_2 = self.shift_transform(images_2, self.duplication_factor)
        out_images = torch.cat((images_1, images_2), dim=0)
        out_labels = torch.cat((labels_1, labels_2), dim=0)

        # Augment all duplicates independently with post shift transformations
        if self.post_shift_transform:
            for i in range(out_images.shape[0]):
                out_images[i] = self.post_shift_transform(out_images[i])

        return out_images, out_labels


class BaseDatasetOD(Dataset, ABC):
    # Variables to be defined in subclass
    dataset: Union[VisionDataset, ImageFolder]
    transform: Compose
    target_class: int

    def __len__(self) -> int:
        r"""
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, in_idx: int) -> Tuple[torch.Tensor, int]:
        r"""
        Retrieves image from the dataset based on the provided index, applies the specified transformations,
        and returns the transformed image and its outlier label: 0 for target class, 1 for other classes.

        Args:
            in_idx (int)
                Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the transformed image and its outlier label.
        """
        out_image, label = self.dataset[in_idx]
        out_image = self.transform(out_image)
        return out_image, 0 if label == self.target_class else 1
