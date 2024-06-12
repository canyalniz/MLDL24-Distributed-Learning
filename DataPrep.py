from typing import Any
import torch
import torch.utils
import torch.utils.data
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import torch.multiprocessing


# Trick to assign different transforms to train/val subsets
# by @ptrblck https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4
class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class DataPrepper:
    def __init__(
        self,
        root="Datasets",
        download=False,
        val_ratio=0,
        uint8_augmentations_list=[],
        normalize=True,
        num_folds=1,
    ) -> None:

        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError("Validation ratio has to be in [0,1).")
        if num_folds < 0:
            raise ValueError("Number of folds must be grater than 0.")

        self.root = root
        self.download = download
        self.val_ratio = val_ratio
        self.uint8_augmentations_list = uint8_augmentations_list
        self.normalize = normalize
        self.num_folds = num_folds

        self.cifar100_training = datasets.CIFAR100(
            root=self.root, train=True, download=self.download, transform=None
        )

        self.cifar100_testing = datasets.CIFAR100(
            root=self.root, train=False, download=self.download, transform=None
        )

        self.train_transforms_base = (
            [v2.ToImage()]
            + self.uint8_augmentations_list
            + [v2.ToDtype(dtype=torch.float32)]
        )
        self.test_transforms_base = [v2.ToImage(), v2.ToDtype(dtype=torch.float32)]

    def get_dataloaders(self, **kwargs):
        validation_subset = None

        train_transforms = self.train_transforms_base
        test_transforms = self.test_transforms_base

        if self.normalize:
            training_channel_means = (
                self.cifar100_training.data.mean(axis=(0, 1, 2), dtype=np.float32) / 255
            )
            training_channel_stds = (
                self.cifar100_training.data.std(axis=(0, 1, 2)) / 255
            )

            test_transforms.append(
                v2.Normalize(training_channel_means, training_channel_stds)
            )

        if self.val_ratio:
            val_transforms = self.test_transforms_base

            training_subset, validation_subset = random_split(
                self.cifar100_training, [1 - self.val_ratio, self.val_ratio]
            )

            if self.normalize:
                train_idx = training_subset.indices

                training_subset_channel_means = (
                    self.cifar100_training.data[train_idx].mean(
                        axis=(0, 1, 2), dtype=np.float32
                    )
                    / 255
                )
                training_subset_channel_stds = (
                    self.cifar100_training.data[train_idx].std(axis=(0, 1, 2)) / 255
                )

                train_transforms.append(
                    v2.Normalize(
                        training_subset_channel_means, training_subset_channel_stds
                    )
                )
                val_transforms.append(
                    v2.Normalize(
                        training_subset_channel_means, training_subset_channel_stds
                    )
                )

            training_subset = SubsetDataset(training_subset)
            validation_subset = SubsetDataset(validation_subset)

            validation_subset.transform = v2.Compose(val_transforms)

        else:
            training_subset = self.cifar100_training

            if self.normalize:
                train_transforms.append(
                    v2.Normalize(training_channel_means, training_channel_stds)
                )

        training_subset.transform = v2.Compose(train_transforms)
        self.cifar100_testing.transform = v2.Compose(test_transforms)

        if validation_subset:
            return (
                DataLoader(training_subset, **kwargs),
                DataLoader(validation_subset, **kwargs),
                DataLoader(self.cifar100_testing, **kwargs),
            )

        return (
            DataLoader(training_subset, **kwargs),
            DataLoader(self.cifar100_testing, **kwargs),
        )

    def get_k_fold_dataloaders(self, **kwargs):
        if self.num_folds < 2:
            raise ValueError(
                "KFold cross validation can only be used with num_folds > 1"
            )

        kfolder = KFold(n_splits=self.num_folds)
        folds = kfolder.split(self.cifar100_training)

        train_transforms = self.train_transforms_base
        test_transforms = self.test_transforms_base

        for train_idx, val_idx in folds:
            if self.normalize:
                fold_training_channel_means = (
                    self.cifar100_training.data[train_idx].mean(
                        axis=(0, 1, 2), dtype=np.float32
                    )
                    / 255
                )
                fold_training_channel_stds = (
                    self.cifar100_training.data[train_idx].std(axis=(0, 1, 2)) / 255
                )
                train_transforms.append(
                    v2.Normalize(
                        fold_training_channel_means,
                        fold_training_channel_stds,
                    )
                )
                test_transforms.append(
                    v2.Normalize(
                        fold_training_channel_means,
                        fold_training_channel_stds,
                    )
                )

            train_subset = SubsetDataset(Subset(self.cifar100_training, train_idx))
            val_subset = SubsetDataset(Subset(self.cifar100_training, val_idx))

            train_subset.transform = v2.Compose(train_transforms)
            val_subset.transform = v2.Compose(test_transforms)

            yield DataLoader(train_subset, **kwargs), DataLoader(val_subset, **kwargs)


class TensorDataset(Dataset):
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets

    def __getitem__(self, index) -> Any:
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class DataPrepperCuda:
    def __init__(
        self,
        root="Datasets",
        download=False,
        val_ratio=0,
        num_folds=1,
    ) -> None:
        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError("Validation ratio has to be in [0,1).")
        if num_folds < 0:
            raise ValueError("Number of folds must be grater than 0.")

        self.root = root
        self.download = download
        self.val_ratio = val_ratio
        self.training_channel_means = None
        self.training_channel_stds = None
        self.num_folds = num_folds
        self.kfold_training_channel_means = None
        self.kfold_training_channel_stds = None

        self.cifar100_training = datasets.CIFAR100(
            root=self.root,
            train=True,
            download=self.download,
        )

        self.cifar100_testing = datasets.CIFAR100(
            root=self.root,
            train=False,
            download=self.download,
        )

        self.cifar100_training.data = torch.tensor(
            self.cifar100_training.data.transpose((0, 3, 1, 2)), device="cuda"
        )
        self.cifar100_training.targets = torch.tensor(
            self.cifar100_training.targets, device="cuda"
        )

        self.cifar100_testing.data = torch.tensor(
            self.cifar100_testing.data.transpose((0, 3, 1, 2)), device="cuda"
        )
        self.cifar100_testing.targets = torch.tensor(
            self.cifar100_testing.targets, device="cuda"
        )

    def get_dataloaders(self, **kwargs):
        training_subset = TensorDataset(
            self.cifar100_training.data,
            self.cifar100_training.targets,
        )
        testing_subset = TensorDataset(
            self.cifar100_testing.data,
            self.cifar100_testing.targets,
        )

        if self.val_ratio:
            train_idx, val_idx = train_test_split(
                np.arange(self.cifar100_training.__len__()), test_size=self.val_ratio
            )
            # training_subset, validation_subset = random_split(
            #     self.cifar100_training, [1 - self.val_ratio, self.val_ratio]
            # )

            # train_idx = training_subset.indices
            # val_idx = validation_subset.indices

            self.training_channel_means = (
                self.cifar100_training.data[train_idx].mean(
                    dim=(0, 2, 3), dtype=torch.float32
                )
                / 255
            )
            self.training_channel_stds = (
                self.cifar100_training.data[train_idx]
                .to(dtype=torch.float32)
                .std(dim=(0, 2, 3))
                / 255
            )

            training_subset = TensorDataset(
                self.cifar100_training.data[train_idx],
                self.cifar100_training.targets[train_idx],
            )
            validation_subset = TensorDataset(
                self.cifar100_training.data[val_idx],
                self.cifar100_training.targets[val_idx],
            )

            return (
                DataLoader(training_subset, **kwargs),
                DataLoader(validation_subset, **kwargs),
                DataLoader(testing_subset, **kwargs),
            )

        self.training_channel_means = (
            self.cifar100_training.data.mean(dim=(0, 2, 3), dtype=torch.float32) / 255
        )
        self.training_channel_stds = (
            self.cifar100_training.data.to(dtype=torch.float32).std(dim=(0, 2, 3)) / 255
        )

        return DataLoader(training_subset, **kwargs)

    def get_k_fold_dataloaders(self, **kwargs):
        if self.num_folds < 2:
            raise ValueError(
                "KFold cross validation can only be used with num_folds > 1"
            )

        kfolder = KFold(n_splits=self.num_folds)
        folds = kfolder.split(self.cifar100_training)

        self.kfold_training_channel_means = torch.empty((self.num_folds, 3))
        self.kfold_training_channel_stds = torch.empty((self.num_folds, 3))

        for i, (train_idx, val_idx) in enumerate(folds):
            self.kfold_training_channel_means[i] = (
                self.cifar100_training.data[train_idx].mean(
                    dim=(0, 2, 3), dtype=torch.float32
                )
                / 255
            )
            self.kfold_training_channel_stds[i] = (
                self.cifar100_training.data[train_idx]
                .to(dtype=torch.float32)
                .std(dim=(0, 2, 3))
                / 255
            )

            training_subset = TensorDataset(
                self.cifar100_training.data[train_idx],
                self.cifar100_training.targets[train_idx],
            )
            validation_subset = TensorDataset(
                self.cifar100_training.data[val_idx],
                self.cifar100_training.targets[val_idx],
            )

            yield (
                DataLoader(training_subset, **kwargs),
                DataLoader(validation_subset, **kwargs),
            )
