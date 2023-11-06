from typing import Optional, Dict, Tuple, Union
from functools import partial

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Compose
from torchvision.datasets import DatasetFolder

from .datasets_od.transforms_od import shifting_transform_rotation
from .datasets_od.cifar10_od import Cifar10SingleClassCSI, Cifar10OD
from .datasets_od.mtt_od import MTTSingleClassCSIwav, MTTODwav
from .datasets_od.collater_od import collater_rearrange_batch_shifting_transform
from .datasets_od.base_datasets import BaseDatasetCSI


class DataSetupOD:
    r"""
    Data setup for outlier detection tasks on supported datasets. This class provides functionality to create DataLoader objects for training and
    testing.

    Supported datasets:
        - 'cifar10'
        - 'mtt_wav'

    Attributes:
        SUPPORTED_DATASETS (set)
            Set of strings representing the supported datasets.

    Args:
        in_dataset (str)
            Name of the dataset to use.
        in_path_download (str)
            Path where the dataset is or will be downloaded/stored.
        in_batchsize (int)
            Batch size for the DataLoader.
        in_val_size (float)
            Proportion of the training set to use as validation.
        in_target_class (int)
            The class label for the target class.
        in_sample_len (int)
            The length of the audio sample, which is fed into the network. Only necessary when processing music not images.
        in_num_workers (int, optional)
            Number of subprocesses to use for data loading. Defaults to 0.
        in_duplication_factor (int, optional)
            Number of times an image is expected to be duplicated. Defaults to 4.
        in_transforms (Dict, optional)
            A dictionary containing transformations. It can have keys:
                - 'pre_shift_transforms': Transformations to be applied before shifting. Defaults to Compose([transforms.ToTensor()])
                - 'shift_transform': Shifting transformation function. Defaults to rotation.
                - 'post_shift_transforms': Transformations to be applied after shifting. Defaults to Compose([])
                - 'test_transform': Transformations to be applied on test data. Defaults to Compose([transforms.ToTensor()])

    Raises:
        NotImplementedError: If the chosen dataset is not supported.
    """

    SUPPORTED_DATASETS = {'cifar10', 'mtt_wav'}

    def __init__(self, in_dataset: str, in_path_download: str, in_batchsize: int, in_target_class: int, in_sample_len: Optional[int],
                 in_val_size: float = 0, in_num_workers: int = 0, in_duplication_factor: int = 4, in_transforms: Optional[Dict] = None):
        if in_dataset not in DataSetupOD.SUPPORTED_DATASETS:
            raise NotImplementedError(
                f"Dataset {in_dataset} is not supported. Supported datasets are: {', '.join(DataSetupOD.SUPPORTED_DATASETS)}")

        self.dataset = in_dataset
        self.path_data = in_path_download
        self.batchsize = in_batchsize
        self.val_size = in_val_size
        self.target_class = in_target_class
        self.num_workers = in_num_workers
        self.duplication_factor = in_duplication_factor
        self.train_transforms, self.test_transform = self._check_transforms(in_transforms)
        self.sample_len = in_sample_len

    def get_trainloader(self) -> Tuple[DataLoader, DataLoader]:
        r"""
        Gets DataLoaders for training and validation based on the specified dataset.

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple containing DataLoaders for training and validation.
        """
        full_trainset = self._get_train_dataset()

        # Split the  data into training and validation sets
        num_train = int((1 - self.val_size) * len(full_trainset))
        num_val = len(full_trainset) - num_train
        trainset, valset = random_split(full_trainset, [num_train, num_val])

        collate_fn_sort_batch = partial(collater_rearrange_batch_shifting_transform, in_duplication_factor=self.duplication_factor)
        out_train_loader = DataLoader(trainset, batch_size=self.batchsize, shuffle=True, num_workers=self.num_workers,
                                      collate_fn=collate_fn_sort_batch)
        out_val_loader = DataLoader(valset, batch_size=self.batchsize, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_sort_batch)
        return out_train_loader, out_val_loader

    def _get_train_dataset(self) -> Union[BaseDatasetCSI, DatasetFolder]:
        r"""
        Gets training data of specified dataset.

        Returns:
            Union[BaseDatasetCSI, DatasetFolder]: Specified training dataset
        """
        if self.dataset == 'cifar10':
            out_full_trainset = Cifar10SingleClassCSI(in_root=self.path_data, in_train=True, in_target_class=self.target_class,
                                                      in_duplication_factor=self.duplication_factor,
                                                      in_pre_shift_transform=self.train_transforms['pre_shift_transforms'],
                                                      in_shift_transform=self.train_transforms['shift_transform'],
                                                      in_post_shift_transform=self.train_transforms['post_shift_transforms'])

            return out_full_trainset
        if self.dataset == 'mtt_wav':
            out_full_trainset = MTTSingleClassCSIwav(in_root=self.path_data, in_train=True, in_target_class=self.target_class,
                                                     in_duplication_factor=self.duplication_factor,
                                                     in_pre_shift_transform=self.train_transforms['pre_shift_transforms'],
                                                     in_shift_transform=self.train_transforms['shift_transform'],
                                                     in_post_shift_transform=self.train_transforms['post_shift_transforms'],
                                                     in_sample_len=self.sample_len)
            return out_full_trainset
        raise NotImplementedError(f"Dataset {self.dataset} is not supported. Supported datasets are: {', '.join(DataSetupOD.SUPPORTED_DATASETS)}")

    def get_testloader(self) -> DataLoader:
        r"""
        Gets DataLoader for testing based on the specified dataset.

        Returns:
            DataLoader: DataLoader for testing data.
        """
        testset = self._get_test_dataset()
        # On music files, samples of the same songs will be colected in a minibatch
        batch_size_test = 1 if self.dataset == 'mtt_wav' else self.batchsize
        out_testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=self.num_workers)
        return out_testloader

    def _get_test_dataset(self) -> Dataset:
        r"""
        Gets test Dataset of specified data.

        Returns:
            Dataset: Specified test dataset
        """
        if self.dataset == 'cifar10':
            out_testset = Cifar10OD(in_root=self.path_data, in_train=False, in_target_class=self.target_class, in_transform=self.test_transform)
            return out_testset
        if self.dataset == 'mtt_wav':
            out_testset = MTTODwav(in_root=self.path_data, in_train=False, in_target_class=self.target_class, in_transform=self.test_transform)
            return out_testset
        raise NotImplementedError(f"Dataset {self.dataset} is not supported. Supported datasets are: {', '.join(DataSetupOD.SUPPORTED_DATASETS)}")

    @staticmethod
    def _check_transforms(in_transforms: Optional[Dict]) -> Tuple[Dict, Compose]:
        r"""
        Checks the supplied transformations and provides default transformations if any are missing.

        Args:
            in_transforms (Dict, optional)
                A dictionary containing transformations. It can have keys:
                    - 'pre_shift_transforms': Transformations to be applied before shifting. Defaults to Compose([transforms.ToTensor()])
                    - 'shift_transform': Shifting transformation function. Defaults to rotation.
                    - 'post_shift_transforms': Transformations to be applied after shifting. Defaults to Compose([])
                    - 'test_transform': Transformations to be applied on test data. Defaults to Compose([transforms.ToTensor()])

        Returns:
            Dict: A dictionary with complete transformations. If a transformation is not provided in the input, a default transformation will be set.
        """
        if in_transforms is None:
            return ({'pre_shift_transforms': Compose([ToTensor(), ]),
                     'shift_transform': shifting_transform_rotation,
                     'post_shift_transforms': Compose([])
                     },
                    Compose([ToTensor(), ]))

        out_train_transforms = {'pre_shift_transforms': None,
                                'shift_transform': None,
                                'post_shift_transforms': None
                                }

        if 'pre_shift_transforms' in in_transforms:
            out_train_transforms['pre_shift_transforms'] = in_transforms['pre_shift_transforms']
        else:
            out_train_transforms['pre_shift_transforms'] = Compose([ToTensor(), ])

        if 'shift_transform' in in_transforms:
            out_train_transforms['shift_transform'] = in_transforms['shift_transform']
        else:
            out_train_transforms['shift_transform'] = shifting_transform_rotation

        if 'post_shift_transforms' in in_transforms:
            out_train_transforms['post_shift_transforms'] = in_transforms['post_shift_transforms']
        else:
            out_train_transforms['post_shift_transforms'] = Compose([])

        if 'test_transforms' in in_transforms:
            out_test_transform = in_transforms['test_transforms']
        else:
            out_test_transform = Compose([ToTensor(), ])

        return out_train_transforms, out_test_transform
