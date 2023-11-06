import json
from typing import Optional, Union, Dict

import torch
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from tqdm import tqdm


def calc_dataset_statistics(in_dataset_type: str, in_path_root: Optional[str] = None) -> None:
    """
    Calculate class-wise statistics for the specified dataset type and save them as a JSON file.
    Supports CIFAR10 and custom datasets in ImageFolder format.

    Args:
        in_dataset_type (str, optional):
            Type of the dataset. Either 'cifar10' or 'imagefolder'.
        in_path_root (str, optional):
            Root path for the ImageFolder dataset. Required if in_dataset_type is 'imagefolder'.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    if in_dataset_type == 'cifar10':
        trainset = CIFAR10(root='../../data', train=True, download=True, transform=transform)
    elif in_dataset_type == 'imagefolder':
        trainset = ImageFolder(root=in_path_root, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {in_dataset_type}")
    stats = calc_classwise_statistics(trainset)
    with open(f"{in_dataset_type}_normalization_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)


def calc_classwise_statistics(in_dataset: Union[CIFAR10, ImageFolder]) -> Dict:
    """
    Calculate class-wise mean and standard deviation for the provided dataset. Assumes 3 channels (RGB) images.

    Args:
        in_dataset (Union[CIFAR10, ImageFolder]):
            Dataset either in CIFAR10 or ImageFolder format.

    Returns:
        stats (dict):
            Dictionary containing class names and their corresponding mean and std values.
    """
    num_classes = len(in_dataset.class_to_idx)
    classwise_pixel_count = {i: 0 for i in range(num_classes)}  # Total number of pixels for each class
    classwise_sum = {i: torch.zeros(3) for i in range(num_classes)}  # Assuming 3 channels (RGB), adjust if not
    classwise_sum_of_squares = {i: torch.zeros(3) for i in range(num_classes)}
    class_names = {v: k for k, v in in_dataset.class_to_idx.items()}  # reverse mapping of class_to_idx

    for img, label in tqdm(in_dataset, desc="Processing images"):
        num_pixels = img.size(1) * img.size(2)

        # Update the running totals
        classwise_pixel_count[label] += num_pixels
        classwise_sum[label] += torch.sum(img, dim=[1, 2])
        classwise_sum_of_squares[label] += torch.sum(img ** 2, dim=[1, 2])

    stats = {}
    for label in classwise_pixel_count.keys():
        mean = classwise_sum[label] / classwise_pixel_count[label]
        var = classwise_sum_of_squares[label] / classwise_pixel_count[label] - mean ** 2
        std = torch.sqrt(var)

        stats[label] = {
            'cls_name': class_names[label],
            'mean': mean.tolist(),
            'std': std.tolist()
        }
    return stats


if __name__ == "__main__":
    DATASET_TYPE = 'cifar10'
    PATH_ROOT = '../../../data/'
    calc_dataset_statistics(DATASET_TYPE, PATH_ROOT)
