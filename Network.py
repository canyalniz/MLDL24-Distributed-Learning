import torch
from torchvision.transforms import v2


class Network(torch.nn.Module):
    def __init__(self, uint8_augmentations_list=[], normalization_transform=None):
        super().__init__()
        self.network_stack = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1600, 384),
            torch.nn.Linear(384, 192),
            torch.nn.Linear(192, 100),
        )

        input_transforms_list = [v2.ToDtype(torch.float32, scale=True)]

        if normalization_transform:
            input_transforms_list.append(normalization_transform)

        self.cuda_train_forward = torch.nn.Sequential(
            v2.Compose(uint8_augmentations_list + input_transforms_list),
            torch.nn.Conv2d(3, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1600, 384),
            torch.nn.Linear(384, 192),
            torch.nn.Linear(192, 100),
        )
        self.cuda_val_forward = torch.nn.Sequential(
            v2.Compose(input_transforms_list),
            torch.nn.Conv2d(3, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1600, 384),
            torch.nn.Linear(384, 192),
            torch.nn.Linear(192, 100),
        )

    def forward(self, input):
        return self.network_stack(input)
