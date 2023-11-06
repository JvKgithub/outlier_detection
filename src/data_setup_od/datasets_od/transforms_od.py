from typing import Tuple

import torch
from torchaudio_augmentations import Noise


def shifting_transform_rotation(in_images: torch.Tensor, in_duplication_factor: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Applies a shifting transform in the form of a rotation to the input images and duplicates them based on the specified factor.

    Args:
        in_images (torch.Tensor)
            The input tensor of images of shape (batch_size, channels, height, width).
        in_duplication_factor (int)
            Factor by which data is duplicated using the shift transform. Must be between 0 and 4.

    Returns:
        - out_images (torch.Tensor): The tensor of rotated images based on the duplication factor.
        - out_labels (torch.Tensor): The tensor containing the rotation labels (0, 1, 2, 3) for each rotated image.

    Raises:
        ValueError: If in_duplication_factor is not between 1 and 4.
    """
    if not 1 <= in_duplication_factor <= 4:
        raise ValueError(f"For rotation shift transform in_duplication_factor must be between 1 and 4 but is {in_duplication_factor}")

        # Generate the rotations based on in_duplication_factor, excluding 0
    rotations = list(range(1, in_duplication_factor))

    # Rotate the images based on the calculated rotations
    rotated_images = [torch.rot90(in_images, k=r, dims=[2, 3]) for r in rotations]
    # Prepend with original images that received no rotation
    rotated_images.insert(0, in_images)

    out_images = torch.cat(rotated_images, dim=0)

    # Create corresponding label tensors, repeating the sequence [0, 1, 2, 3]
    out_labels = torch.tensor(list(range(in_duplication_factor)) * in_images.size(0), dtype=torch.long)

    return out_images, out_labels


def shifting_transform_gaussian_noise(in_audio: torch.Tensor, in_duplication_factor: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Applies Gaussian noise with increasing intensity based on the duplication factor.

    Args:
        in_audio (torch.Tensor)
            The input tensor of audio arrays of shape (batch_size, channels, height, width).
        in_duplication_factor (int)
            Factor by which data is duplicated using the noise transform. Must be between 0 and 4.

    Returns:
        - out_images (torch.Tensor): The tensor of images with added noise based on the duplication factor.
        - out_labels (torch.Tensor): The tensor containing the noise intensity labels (0, 1, 2, 3) for each noisy image.

    Raises:
        ValueError: If in_duplication_factor is not between 1 and 4.
    """
    if not 1 <= in_duplication_factor <= 4:
        raise ValueError(f"For Gaussian noise transform, in_duplication_factor must be between 1 and 4 but is {in_duplication_factor}")

    if in_duplication_factor == 2:
        noise_snr_range = [[0.001, 0.1]]
    else:
        noise_snr_range = [[0.0001, 0.001], [0.001, 0.01], [0.01, 0.1]][:in_duplication_factor - 1]

    noisy_signal_list = [in_audio]  # Prepend with the original image

    for intensity in noise_snr_range:
        noisy_signal = Noise(min_snr=intensity[0], max_snr=intensity[1])(in_audio)
        noisy_signal_list.append(noisy_signal)

    # Concatenate the original and noisy images
    out_audios = torch.cat(noisy_signal_list, dim=0)

    # Create corresponding label tensors
    out_labels = torch.tensor(list(range(in_duplication_factor)) * in_audio.size(0), dtype=torch.long)

    return out_audios, out_labels
