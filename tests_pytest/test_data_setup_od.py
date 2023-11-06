from typing import Tuple, List

import pytest
import torch

from src.data_setup_od import data_setup
from src.data_setup_od.datasets_od.transforms_od import shifting_transform_rotation
from src.data_setup_od.datasets_od.collater_od import collater_rearrange_batch_shifting_transform


# ---- Tests for transforms_od ----
def test_shifting_transform_rotation_errors():
    """Tests if not allowed duplication factors raises an error."""
    dummy_images = torch.rand(1, 3, 32, 32)

    # Check for ValueError for 0 and 5
    with pytest.raises(ValueError, match=r"must be between 1 and 4 but is 0"):
        shifting_transform_rotation(dummy_images, 0)

    with pytest.raises(ValueError, match=r"must be between 1 and 4 but is 5"):
        shifting_transform_rotation(dummy_images, 5)


def test_shifting_transform_rotation_dims():
    """Tests if allowed duplication factors produce correctly shaped data."""
    dummy_images = torch.rand(5, 3, 32, 32)

    for factor in range(1, 5):
        transformed_images, transformed_labels = shifting_transform_rotation(dummy_images, factor)

        # Check the number of resulting images
        assert transformed_images.shape[0] == dummy_images.shape[0] * factor
        assert transformed_labels.shape[0] == dummy_images.shape[0] * factor

        # Check other dimensions are unchanged
        assert transformed_images.shape[1:] == dummy_images.shape[1:]


# ---- Tests for collater_od ----
def generate_mock_data(batch_size: int, img_shape: Tuple[int, int, int], duplication_factor: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Generates mock data in the form of a list of tuples, where each tuple contains an image tensor and its label tensor.

    This utility function creates mock data for testing purposes. Each image tensor in the list will have a shape determined by the
    `batch_size`, `img_shape`, and `duplication_factor`. The labels are random integers between 0 and 9.

    Args:
        batch_size (int): The number of distinct batches of image-label pairs.
        img_shape (Tuple[int, int, int]): The shape of an individual image in the format (channels, height, width).
        duplication_factor (int): Number of times an image is expected to be duplicated in each batch.

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]: List containing `batch_size` tuples, where each tuple holds an image tensor of shape
        `(duplication_factor * 2, *img_shape)` and a corresponding label tensor of shape `(duplication_factor * 2)`.
    """
    imgs = torch.randn(batch_size, duplication_factor * 2, *img_shape)
    labels = torch.randint(0, 10, (batch_size, duplication_factor * 2))
    data_list = [(imgs[i], labels[i]) for i in range(batch_size)]
    return data_list


def test_collater_rearrange_batch_shifting_transform_valid_input():
    """Tests if the collater produces correctly shaped data."""
    batch_size = 8
    duplication_factor = 4
    mock_data = generate_mock_data(batch_size=batch_size, img_shape=(3, 32, 32), duplication_factor=duplication_factor)
    out_images, out_labels = collater_rearrange_batch_shifting_transform(mock_data, in_duplication_factor=duplication_factor)

    # Verify that the output shapes are correct
    assert out_images.shape == (batch_size * duplication_factor * 2, 3, 32, 32)
    assert out_labels.shape == (batch_size * duplication_factor * 2,)


def test_collater_rearrange_batch_shifting_transform_incorrect_duplication_factor():
    """Tests if the collater recognizes duplication_factor missmatches."""
    batch_size = 8
    duplication_factor_mock = 3
    duplication_factor_collater = 4
    mock_data = generate_mock_data(batch_size=batch_size, img_shape=(3, 32, 32), duplication_factor=duplication_factor_mock)

    # Expect a ValueError due to incorrect duplication factor
    with pytest.raises(ValueError, match="Duplication factor inconsistency"):
        collater_rearrange_batch_shifting_transform(mock_data, in_duplication_factor=duplication_factor_collater)


# ---- Tests for data_setup_od ----
def test_supported_dataset():
    """Tests if an existing dataset does not raise an error."""
    data_setup.DataSetupOD(in_dataset='cifar10',
                           in_path_download='path_dummy',
                           in_batchsize=1,
                           in_target_class=0,
                           in_sample_len=None)


def test_unsupported_dataset():
    """Tests if a not existing dataset does raise an error."""
    with pytest.raises(NotImplementedError, match=r"Dataset crazy_data is not supported"):
        data_setup.DataSetupOD(in_dataset='crazy_data',
                               in_path_download='path_dummy',
                               in_batchsize=1,
                               in_target_class=0,
                               in_sample_len=None)
