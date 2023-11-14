from typing import Callable
import os

import torch
import torchaudio
from torchaudio_augmentations import RandomResizedCrop
from torchvision.transforms import Compose
from torchvision.datasets import DatasetFolder


class MTTSingleClassCSIwav(DatasetFolder):
    r"""
    Dataset wrapper for custom subset of MagnaTagATune for outlier detection tasks.
    Extract samples from a single target class and apply CSI specific transformations.

    Args:
        in_root (str)
            Root directory of the MTT dataset. Folder structure is assumed to be:
            ./root_dir
            >train
            >>cls_1
            >>cls_2
            >test
            >>cls_1
            >>cls_2
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
        in_sample_len (int)
            The length of the audio sample, which is fed into the network. Only necessary when processing music not images.
    """

    def __init__(self, in_root: str, in_train: bool, in_target_class: int, in_duplication_factor: int, in_sample_len: int,
                 in_pre_shift_transform: Compose, in_shift_transform: Callable, in_post_shift_transform: Compose):
        path_data = os.path.join(in_root, 'train') if in_train else os.path.join(in_root, 'test')
        super().__init__(path_data, loader=load_wav_file, extensions=('.wav',))
        self.class_to_idx = self.class_to_idx  # Dictionary mapping classes to indices.
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}  # Reverse mapping
        self.classes = sorted(self.class_to_idx.keys())  # Alphabetically sorted class names
        self.target_class = in_target_class
        self.indices = [idx for idx, (path, label) in enumerate(self.samples) if label == self.target_class]

        self.sample_len = in_sample_len
        self.pre_shift_transform = in_pre_shift_transform
        self.shift_transform = in_shift_transform
        self.post_shift_transform = in_post_shift_transform
        self.duplication_factor = in_duplication_factor

    def __len__(self):
        r"""
         Returns the total number of samples in the dataset.

         Returns:
             int: Total number of samples.
         """
        return len(self.indices)

    def __getitem__(self, in_idx):
        r"""
        Processing pipeline:
        - retrieve file
        - cut out two random samples (can be overlapping)
        - duplicate according to duplication_factor
        - apply transformations

        Args:
            in_idx (int)
                Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the transformed image and its outlier label.
        """
        path_wav, _ = self.samples[in_idx]
        audio_sample = self.loader(path_wav)

        # Apply pre shift transform
        audio_sample = self.pre_shift_transform(audio_sample)

        # Create image pairs
        audio_1 = RandomResizedCrop(self.sample_len)(audio_sample)
        audio_2 = torch.clone(audio_1)

        # Apply shift
        audio_1, labels_1 = self.shift_transform(audio_1, self.duplication_factor)
        audio_2, labels_2 = self.shift_transform(audio_2, self.duplication_factor)
        out_audio_minibatch = torch.cat((audio_1, audio_2), dim=0).unsqueeze(1)  # Extra dimension needed for pitch change augmentation
        out_labels = torch.cat((labels_1, labels_2), dim=0)

        # Augment all duplicates independently with post shift transformations
        if self.post_shift_transform:
            # Initialize a list to hold the transformed tensors
            transformed_samples = []
            # Apply the post_shift_transform to each sample
            for i in range(out_audio_minibatch.shape[0]):
                transformed_sample = self.post_shift_transform(out_audio_minibatch[i])
                transformed_samples.append(transformed_sample)
            out_audio_minibatch = torch.stack(transformed_samples, dim=0)

        # Duplicate along channel axis to simulate RGB
        out_audio_minibatch = out_audio_minibatch.repeat(1, 3, 1, 1)
        return out_audio_minibatch, out_labels


class MTTODwav(DatasetFolder):

    def __init__(self, in_root: str, in_train: bool, in_target_class: int, in_transform: Compose):
        path_data = os.path.join(in_root, 'train') if in_train else os.path.join(in_root, 'test')
        super().__init__(path_data, loader=load_wav_file, extensions=('.wav',))
        self.class_to_idx = self.class_to_idx   # This will give you a dictionary mapping classes to indices.
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}   # Reverse mapping
        self.classes = sorted(self.class_to_idx.keys())  # Alphabetically sorted class names
        self.target_class = in_target_class
        self.transform = in_transform
        self.sample_len = 43740

    def __len__(self):
        r"""
         Returns the total number of samples in the dataset.

         Returns:
             int: Total number of samples.
         """
        return len(self.samples)

    def __getitem__(self, in_idx):
        r"""
        Retrieves image from the dataset based on the provided index, applies the specified transformations,
        and returns the transformed image and its outlier label: 0 for target class, 1 for other classes.

        Args:
            in_idx (int)
                Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the transformed image and its outlier label.
        """
        path_wav, out_label = self.samples[in_idx]
        out_label = 0 if out_label == self.target_class else 1
        audio_sample = self.loader(path_wav)

        # Calculate the number of full sub-samples that fit in the audio sample
        num_full_samples = audio_sample.shape[1] // self.sample_len

        # Trim the audio sample to only include full sub-samples
        trimmed_audio_length = num_full_samples * self.sample_len
        trimmed_audio_sample = audio_sample[:, :trimmed_audio_length]

        # Reshape the trimmed audio sample into a minibatch of sub-samples
        out_minibatch = trimmed_audio_sample.view(-1, self.sample_len)

        # Apply transform to each sub-sample if necessary
        if self.transform:
            out_minibatch = self.transform(out_minibatch)

        out_minibatch = out_minibatch.unsqueeze(1).repeat(1, 3, 1, 1)
        out_label = torch.full((out_minibatch.size(0),), out_label, dtype=torch.long)
        return out_minibatch, out_label


def load_wav_file(in_path: str) -> torch.Tensor:
    r"""
    Helper function to load audio file.

    Args:
        in_path (str)
            Index of the sample to retrieve.

    Returns:
        torch.Tensor: Tensor containing the waveform of the audiofile.
    """
    out_waveform, _ = torchaudio.load(in_path)
    return out_waveform
