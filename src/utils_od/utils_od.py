import random
from typing import Tuple, Dict, Optional
import warnings

import numpy as np
import torch
from torchvision import transforms
import torchaudio_augmentations as aud_aug
from torchaudio.transforms import MelSpectrogram

from src.data_setup_od.datasets_od.transforms_od import shifting_transform_rotation, shifting_transform_gaussian_noise
from src.data_setup_od.datasets_od.cifar10_od import NORMALIZATION_STATS_CIFAR10


def set_seed(in_seed: int = 42):
    r"""
    Sets the random seed for Python's random module, NumPy's random module, and all relevant torch modules to ensure
    reproducibility of results. It also makes the operations on GPU deterministic.

    Args:
        in_seed (int, optional)
            The seed value to be set. Defaults to 42.
    """
    random.seed(in_seed)
    np.random.seed(in_seed)
    torch.manual_seed(in_seed)
    torch.cuda.manual_seed(in_seed)
    torch.cuda.manual_seed_all(in_seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_transforms(in_transform_type: str, in_image_size: Tuple, in_dataset_mean: Optional[Tuple] = None, in_dataset_std: Optional[Tuple] = None,
                   in_sample_rate: Optional[int] = None, in_sample_len: Optional[int] = None) -> Dict:
    r"""
    Retrieve a dictionary of data transformations based on the specified transform type. This function supports
    the creation of transformations for 'image' and 'wav' data types. For 'image' type, it processes natural images
    with RGB channels, and for 'wav' type, it handles waveform audio data.

    Args:
        in_transform_type (str):
            The type of data transformation to retrieve. Supported types are 'image' and 'wav'.
        in_image_size (Tuple):
            Desired output size of the image, only applicable for 'image' transform type.
        in_dataset_mean (Optional[Tuple]):
            The mean value of the dataset used for normalization, applicable for 'image' transform type.
        in_dataset_std (Optional[Tuple]):
            The standard deviation of the dataset used for normalization, applicable for 'image' transform type.
        in_sample_rate (Optional[int]):
            The sample rate of the audio data, only applicable for 'wav' transform type.
        in_sample_len (Optional[int]):
            The length of the audio samples, only applicable for 'wav' transform type.

    Returns:
        Dict:
            A dictionary containing transformation all train and test transforms.
            Raises NotImplementedError if an unsupported transform type is specified.

    Raises:
        NotImplementedError: If `in_transform_type` is neither 'image' nor 'wav'.
    """
    if in_transform_type == 'image':
        out_transforms_dict = _get_transforms_natural_image(in_image_size, in_dataset_mean, in_dataset_std)
    elif in_transform_type == 'wav':
        out_transforms_dict = _get_transforms_wav(in_image_size, in_sample_rate, in_sample_len)
    else:
        raise NotImplementedError("Other transforms than image not yet implemented.")
    return out_transforms_dict


def _get_transforms_natural_image(in_image_size: Tuple, in_normalization_mean: Tuple, in_normalization_std: Tuple):
    """
    Get data transformations for RGB natural images

    Args:
        in_image_size (tuple)
            Desired input image size.
        in_normalization_mean (tuple)
            Color channel-wise data mean
        in_normalization_std (tuple)
            Color channel-wise data standard deviation.

    Returns:
        dict: Dictionary containing relevant transformations for training and testing.

    """

    pre_shift_transforms = transforms.Compose([
        transforms.Resize(in_image_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    post_shift_transforms = transforms.Compose([
        transforms.RandomApply(transforms=[
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomApply(transforms=[
            transforms.Grayscale(num_output_channels=3)
        ], p=0.2),
        transforms.RandomApply(transforms=[
            transforms.RandomResizedCrop(size=in_image_size, scale=(0.08, 1.0),
                                         ratio=(3. / 4., 4. / 3.), antialias=False)
        ], p=0.8),
        transforms.Normalize(mean=in_normalization_mean, std=in_normalization_std)
    ])

    shift_transforms = shifting_transform_rotation

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=in_normalization_mean, std=in_normalization_std)
    ])

    return {
        "pre_shift_transforms": pre_shift_transforms,
        "post_shift_transforms": post_shift_transforms,
        "shift_transform": shift_transforms,
        "test_transforms": test_transforms
    }


def _get_transforms_wav(in_image_size: Tuple, in_sample_rate: int, in_sample_len: int):
    r"""
    Get data transformations for .wav data. Some audio augmentations are performed in time-domain, then the sample is converted into a Mel spectogram.

    Args:
        in_image_size (tuple)
            Desired input image size, in_image_size[0] is the number of mels, in_image_size[1] is time dimension in pixels.
        in_sample_rate (int)
            Sampling rate of the audio files. Have to be identical for all files of the dataset.
        in_sample_len (int)
            Length of the audio sample. Since Complete audio file is too large for a single training sample, it will be cut into a samples.

    Returns:
        dict: Dictionary containing relevant transformations for training and testing.
    """

    pre_shift_transforms = transforms.Compose([])
    shift_transforms = shifting_transform_gaussian_noise

    # Mel spectogram properties
    n_fft = 2048  # Common value for music processing
    desired_time_frames = in_image_size[1]
    hop_length = (in_sample_len - 1) // desired_time_frames + 1

    # Augmentations probabilities, some with a higher, some with a lower probability
    prob_small = 0.4
    prob_large = 0.8
    post_shift_transforms = transforms.Compose([
        aud_aug.RandomApply(transforms=[aud_aug.PolarityInversion()], p=prob_large),
        aud_aug.RandomApply(transforms=[aud_aug.Gain(min_gain=-6, max_gain=0)], p=prob_small),
        aud_aug.RandomApply(transforms=[aud_aug.HighLowPass(sample_rate=in_sample_rate)], p=prob_large),
        aud_aug.RandomApply(transforms=[
            aud_aug.PitchShift(n_samples=in_sample_len, sample_rate=in_sample_rate, pitch_shift_min=-5,
                               pitch_shift_max=5)], p=prob_small),
        aud_aug.RandomApply(transforms=[aud_aug.Reverb(sample_rate=in_sample_rate)], p=prob_small),
        MelSpectrogram(n_mels=in_image_size[0], hop_length=hop_length, n_fft=n_fft),
        transforms.Lambda(normalized_log_spectrum),  # Convert to log and normalize the Mel spectrogram
        transforms.Resize(in_image_size)  # To ensure input size consistency due to rounding in MelSpectogram
    ])
    test_transforms = transforms.Compose([
        MelSpectrogram(n_mels=in_image_size[0], hop_length=hop_length, n_fft=n_fft),
        transforms.Lambda(normalized_log_spectrum),  # Convert to log and normalize the Mel spectrogram
        transforms.Resize(in_image_size)
    ])

    return {
        "pre_shift_transforms": pre_shift_transforms,
        "post_shift_transforms": post_shift_transforms,
        "shift_transform": shift_transforms,
        "test_transforms": test_transforms
    }


def normalized_log_spectrum(in_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply log conversion and normalization to a tensor representing a Mel spectrogram.
    The function uses log1p for log conversion which computes log(1+x) for numerical stability,
    especially for values close to zero. After log conversion, it normalizes the values
    to be between 0 and 1. If the maximum and minimum values are the same, the tensor is left
    unchanged to avoid division by zero while handling uniform data.

    Args:
        in_tensor (torch.Tensor)
            The input tensor to apply the log conversion and normalization.

    Returns:
        torch.Tensor
            The tensor after applying log conversion and normalization, with values scaled
            between 0 and 1. If the input tensor has uniform values, the returned tensor
            will be the same as the input tensor.
    """
    log_scaled_tensor = torch.log1p(in_tensor)
    min_val, max_val = log_scaled_tensor.min(), log_scaled_tensor.max()

    if min_val == max_val:
        # Leave the tensor unchanged if all the values are the same
        normalized_tensor = in_tensor
    else:
        normalized_tensor = (log_scaled_tensor - min_val) / (max_val - min_val)

    return normalized_tensor


NORMALIZATION_STATS = {
    'cifar10': NORMALIZATION_STATS_CIFAR10
}


def get_normalization_stats(in_config: Dict) -> Tuple[Tuple, Tuple]:
    """
    Get mean and standard deviation for data processing. If none are specified and known config[dataset_name] is in (cifar10, ), precalculated
    statistics are used. If not, zero mean and unit std are used as defaults.

    Args:
        in_config (tuple)
            Configuration dict

    Returns:
        out_dataset_mean (tuple)
            Color channel-wise mean.
        out_dataset_std (tuple)
            Color channel-wise std.
    """
    # Default values
    default_mean = (0.0, 0.0, 0.0)
    default_std = (1.0, 1.0, 1.0)

    # If data_mean and data_std are specified in the config, use them.
    out_dataset_mean = in_config['dataset_mean'] if in_config['dataset_mean'] else None
    out_dataset_std = in_config['dataset_std'] if in_config['dataset_std'] else None

    # If not specified, try to infer from NORMALIZATION_STATS.
    if not out_dataset_mean and in_config['dataset_name'] in NORMALIZATION_STATS:
        out_dataset_mean = tuple(NORMALIZATION_STATS[in_config['dataset_name']][in_config['target_class']]['mean'])

    if not out_dataset_std and in_config['dataset_name'] in NORMALIZATION_STATS:
        out_dataset_std = tuple(NORMALIZATION_STATS[in_config['dataset_name']][in_config['target_class']]['std'])

    # If still not set, use default values and print a warning.
    if not out_dataset_mean:
        out_dataset_mean = default_mean
        warnings.warn('Unknown dataset mean statistics, using zero-mean as default.')

    if not out_dataset_std:
        out_dataset_std = default_std
        warnings.warn('Unknown dataset standard deviation statistics, using unit-std as default.')

    return out_dataset_mean, out_dataset_std
