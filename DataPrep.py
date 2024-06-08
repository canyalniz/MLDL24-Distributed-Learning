from typing import Any
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from numpy import float32
import torch.multiprocessing


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
        self.training_channel_means = None
        self.training_channel_stds = None

    def _get_cifar100_datasets(self):
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

        if self.val_ratio:
            train_data, val_data, train_targets, val_targets = train_test_split(
                cifar100_training.data,
                cifar100_training.targets,
                test_size=self.val_ratio,
            )
            cifar100_training = TensorDataset(train_data, train_targets)
            cifar100_validation = TensorDataset(val_data, val_targets)

        # data is a uint8 Tensor(if cuda) or ndarray(if not cuda) w/ elements in [0,255]
        # normalization will be applied to float elements in [0,1]
        # so we divide the means and the stds by 255 to account for scaling
        # the logic is there to follow the right syntax (torch or np) according to cuda

        # calculate means for each channel on the training set
        self.training_channel_means = (
            cifar100_training.data.mean(axis=(0, 1, 2), dtype=float32) / 255
        )

        # calculate stds for each channel on the training set
        self.training_channel_stds = cifar100_training.data.std(axis=(0, 1, 2)) / 255

        return cifar100_training, cifar100_testing, cifar100_validation

    def get_datasets(self):
        cifar100_training, cifar100_testing, cifar100_validation = (
            self._get_cifar100_datasets()
        )

        test_transforms_list = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        train_transforms_list = (
            [v2.ToImage()]
            + self.uint8_augmentations_list
            + [v2.ToDtype(torch.float32, scale=True)]
        )
        if self.normalize:
            # add normalization step to transform lists
            test_transforms_list = test_transforms_list + [
                v2.Normalize(self.training_channel_means, self.training_channel_stds)
            ]
            train_transforms_list = train_transforms_list + [
                v2.Normalize(self.training_channel_means, self.training_channel_stds)
            ]

        cifar100_training.transform = v2.Compose(train_transforms_list)
        cifar100_testing.transform = v2.Compose(test_transforms_list)
        if cifar100_validation:
            # validation set gets the test transforms
            cifar100_validation.transform = v2.Compose(test_transforms_list)

        return cifar100_training, cifar100_testing, cifar100_validation

    def get_cuda_preloaded_datasets(self):
        cifar100_training, cifar100_testing, cifar100_validation = (
            self._get_cifar100_datasets()
        )

        # cifar100 ndarrays are channels last by default
        # pytorch image processing expects channels first by default
        # when preloading to cuda ToImage() will not be applied
        # need to manually put into channels first
        # by calling transpose on the data ndarrays
        cifar100_training.data = torch.tensor(
            cifar100_training.data.transpose((0, 3, 1, 2)), device="cuda"
        )
        cifar100_training.targets = torch.tensor(
            cifar100_training.targets, device="cuda"
        )

        cifar100_testing.data = torch.tensor(
            cifar100_testing.data.transpose((0, 3, 1, 2)), device="cuda"
        )
        cifar100_testing.targets = torch.tensor(cifar100_testing.targets, device="cuda")

        if cifar100_validation:
            cifar100_validation.data = torch.tensor(
                cifar100_validation.data.transpose((0, 3, 1, 2)), device="cuda"
            )
            cifar100_validation.targets = torch.tensor(
                cifar100_validation.targets, device="cuda"
            )

        return cifar100_training, cifar100_testing, cifar100_validation

    def get_dataloaders(
        self, preload_cuda=True, batch_size=32, num_workers=0, **kwargs
    ):
        if preload_cuda:
            num_workers = 0
            print(
                "preloaded cuda tensors don't support num_workers > 0, setting num_workers=0"
            )

            train_set, test_set, val_set = self.get_cuda_preloaded_datasets()
        else:
            train_set, test_set, val_set = self.get_datasets()

        if val_set:
            return (
                DataLoader(
                    train_set, batch_size=batch_size, num_workers=num_workers, **kwargs
                ),
                DataLoader(
                    test_set, batch_size=batch_size, num_workers=num_workers, **kwargs
                ),
                DataLoader(
                    val_set, batch_size=batch_size, num_workers=num_workers, **kwargs
                ),
            )

        return DataLoader(
            train_set, batch_size=batch_size, num_workers=num_workers, **kwargs
        ), DataLoader(
            test_set, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
