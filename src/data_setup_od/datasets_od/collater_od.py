from typing import List, Tuple

import torch


def collater_rearrange_batch_shifting_transform(in_data: List[Tuple[torch.Tensor, torch.Tensor]], in_duplication_factor: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Rearranges a batch of data by interleaving duplicated images and their labels. Batch follows this format:
        - A_no_shift_1
        - A_no_shift_2
        -  ...
        - A_shift_1_1
        - A_shift_1_2
        - ...
        - A_shift_2_1
        - A_shift_2_2
        - ...
        - B_no_shift_1
        - B_no_shift_2
        - ...
        - B_shift_1_1
        - B_shift_1_2
        - ...
        - B_shift_2_1
        - B_shift_2_2
    A and B are the original image pairs created before the shift transformation step. Duplication factor is three in this example.

    Args:
        in_data (List[Tuple[torch.Tensor, torch.Tensor]])
            Input data consisting of a list of tuples where each tuple contains an image tensor and its label tensor.
        in_duplication_factor (int)
            Number of times an image is expected to be duplicated.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the rearranged image tensor and its label tensor.

    Raises:
        ValueError: If the duplication factor provided is inconsistent with the number of duplicated images in the input data. If main structure is
        followed, this should never happen.
    """
    if in_duplication_factor != in_data[0][0].shape[0] / 2:
        raise ValueError(f"Duplication factor inconsistency. Collater expects {in_duplication_factor}, "
                         f"but image is duplicated {in_data[0][0].shape[0] / 2} times.")

    # Unzip the input data into separate lists of images and labels
    images_list, labels_list = zip(*in_data)

    # Stack lists to get a new dimension representing each minibatch, then reshape to interleave
    out_images = torch.stack(images_list, dim=1).reshape(-1, *images_list[0].shape[1:])
    out_labels = torch.stack(labels_list, dim=1).reshape(-1)

    return out_images, out_labels
