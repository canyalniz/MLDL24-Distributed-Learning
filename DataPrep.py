import torch
from torchvision import datasets
from torch.utils.data import random_split
from torchvision.transforms import ToTensor


class DataPrepper:
    def __init__(
        self, root="Datasets", download=False, val_ratio=0, manual_seed=42
    ) -> None:
        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError("Validation ratio has to be in [0,1)")

        self.root = root
        self.download = download
        self.val_ratio = val_ratio
        self.manual_seed = manual_seed

    def get_datasets(self):
        cifar100_training = datasets.CIFAR100(
            root=self.root,
            train=True,
            download=self.download,
            transform=ToTensor(),
            target_transform=(
                lambda t: torch.nn.functional.one_hot(torch.tensor(t), num_classes=100)
            ),
        )
        cifar100_testing = datasets.CIFAR100(
            root=self.root,
            train=False,
            download=self.download,
            transform=ToTensor(),
            target_transform=(
                lambda t: torch.nn.functional.one_hot(torch.tensor(t), num_classes=100)
            ),
        )

        if self.val_ratio == 0:
            return (cifar100_training, cifar100_testing, None)

        training_set, val_set = random_split(
            cifar100_training,
            [1 - self.val_ratio, self.val_ratio],
            generator=torch.Generator().manual_seed(self.manual_seed),
        )
        return (training_set, cifar100_testing, val_set)

    def get_dataloaders(self):
        train_set, test_set, val_set = self.get_datasets()
        # wip
        pass
