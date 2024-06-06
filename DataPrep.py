import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.transforms import v2


class Cifar100CudaDataset(Dataset):
    def __init__(
        self,
        root="Datasets",
        train=True,
        download=False,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    ) -> None:
        cifar100 = datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )

        self.len = cifar100.__len__()
        batch_loader = DataLoader(
            dataset=cifar100, batch_size=self.len, pin_memory=True
        )

        data, target = next(iter(batch_loader))

        self.data = data.to("cuda")
        self.target = target.to("cuda")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class DataPrepper:
    def __init__(
        self,
        root="Datasets",
        download=False,
        val_ratio=0,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    ) -> None:
        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError("Validation ratio has to be in [0,1)")

        self.root = root
        self.download = download
        self.val_ratio = val_ratio
        self.transform = transform

    def get_datasets(self, cuda=True):
        if cuda:
            cifar100_training = Cifar100CudaDataset(
                root=self.root,
                train=True,
                download=self.download,
                transform=self.transform,
            )
            cifar100_testing = Cifar100CudaDataset(
                root=self.root,
                train=False,
                download=self.download,
                transform=self.transform,
            )
        else:
            cifar100_training = datasets.CIFAR100(
                root=self.root,
                train=True,
                download=self.download,
                transform=self.transform,
            )
            cifar100_testing = datasets.CIFAR100(
                root=self.root,
                train=False,
                download=self.download,
                transform=self.transform,
            )

        if self.val_ratio == 0:
            return (cifar100_training, cifar100_testing, None)

        training_set, val_set = random_split(
            cifar100_training,
            [1 - self.val_ratio, self.val_ratio],
            generator=torch.Generator(),
        )
        return (training_set, cifar100_testing, val_set)

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
