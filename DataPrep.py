from typing import Any
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from numpy import float32


class TensorDataset(Dataset):
    def __init__(self, data, targets, transform=None) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index) -> Any:
        if self.transform:
            return self.transform(self.data[index]), self.targets[index]
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class DataPrepper:
    def __init__(
        self,
        root="Datasets",
        download=False,
        val_ratio=0,
        uint8_augmentations_list=[],
        normalize=True,
    ) -> None:
        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError("Validation ratio has to be in [0,1)")

        self.root = root
        self.download = download
        self.val_ratio = val_ratio
        self.uint8_augmentations_list = uint8_augmentations_list
        self.normalize = normalize

    def get_datasets(self, cuda=True):

        cifar100_training = datasets.CIFAR100(
            root=self.root,
            train=True,
            download=self.download,
        )
        cifar100_testing = datasets.CIFAR100(
            root=self.root,
            train=False,
            download=self.download,
        )
        cifar100_validation = None

        # cifar100 ndarrays are channels last by default
        # pytorch image processing expects channels first by default
        cifar100_training.data = cifar100_training.data.transpose((0, 3, 1, 2))
        cifar100_testing.data = cifar100_testing.data.transpose((0, 3, 1, 2))

        if self.val_ratio:
            train_data, val_data, train_targets, val_targets = train_test_split(
                cifar100_training.data,
                cifar100_training.targets,
                test_size=self.val_ratio,
            )
            cifar100_training = TensorDataset(train_data, train_targets)
            cifar100_validation = TensorDataset(val_data, val_targets)

        if cuda:
            cifar100_training.data = torch.tensor(cifar100_training.data, device="cuda")
            cifar100_training.targets = torch.tensor(
                cifar100_training.targets, device="cuda"
            )

            cifar100_testing.data = torch.tensor(cifar100_testing.data, device="cuda")
            cifar100_testing.targets = torch.tensor(
                cifar100_testing.targets, device="cuda"
            )

            if cifar100_validation:
                cifar100_validation.data = torch.tensor(
                    cifar100_validation.data, device="cuda"
                )
                cifar100_validation.targets = torch.tensor(
                    cifar100_validation.targets, device="cuda"
                )

            # the original ndarrays are already converted into uint8 Tensors
            # ToImage() unnecessary
            test_transforms_list = [v2.ToDtype(torch.float32, scale=True)]
            train_transforms_list = self.uint8_augmentations_list + [
                v2.ToDtype(torch.float32, scale=True)
            ]
        else:
            # the original ndarrays are untouched
            # ToImage() needed
            test_transforms_list = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
            train_transforms_list = (
                [v2.ToImage()]
                + self.uint8_augmentations_list
                + [v2.ToDtype(torch.float32, scale=True)]
            )

        if self.normalize:
            # data is a uint8 Tensor(if cuda) or ndarray(if not cuda) w/ elements in [0,255]
            # normalization will be applied to float elements in [0,1]
            # so we divide the means and the stds by 255 to account for scaling
            # the logic is there to follow the right syntax (torch or np) according to cuda

            # calculate means for each channel on the training set
            training_channel_means = (
                (cifar100_training.data.mean(dim=(0, 2, 3), dtype=torch.float32) / 255)
                if cuda
                else (cifar100_training.data.mean(axis=(0, 2, 3), dtype=float32) / 255)
            )
            # calculate stds for each channel on the training set
            training_channel_stds = (
                (
                    cifar100_training.data.to(dtype=torch.float32).std(dim=(0, 2, 3))
                    / 255
                )
                if cuda
                else (cifar100_training.data.std(axis=(0, 2, 3)) / 255)
            )
            # add normalization step to transform lists
            test_transforms_list = test_transforms_list + [
                v2.Normalize(training_channel_means, training_channel_stds)
            ]
            train_transforms_list = train_transforms_list + [
                v2.Normalize(training_channel_means, training_channel_stds)
            ]

        cifar100_training.transform = v2.Compose(train_transforms_list)
        cifar100_testing.transform = v2.Compose(test_transforms_list)
        if cifar100_validation:
            # validation set gets the test transforms
            cifar100_validation.transform = v2.Compose(test_transforms_list)

        return (cifar100_training, cifar100_testing, cifar100_validation)

    def get_dataloaders(
        self, cuda=True, batch_size=32, val_full_batchsize=False, **kwargs
    ):
        train_set, test_set, val_set = self.get_datasets(cuda)
        if val_set:
            if val_full_batchsize:
                return (
                    DataLoader(train_set, batch_size=batch_size, **kwargs),
                    DataLoader(test_set, batch_size=batch_size, **kwargs),
                    DataLoader(
                        val_set, batch_size=int(val_set.__len__() / 2), **kwargs
                    ),
                )
            else:
                return (
                    DataLoader(train_set, batch_size=batch_size, **kwargs),
                    DataLoader(test_set, batch_size=batch_size, **kwargs),
                    DataLoader(val_set, batch_size=batch_size, **kwargs),
                )

        return DataLoader(train_set, batch_size=batch_size, **kwargs), DataLoader(
            test_set, batch_size=batch_size, **kwargs
        )
