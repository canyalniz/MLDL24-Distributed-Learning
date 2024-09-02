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

        self.uint8_augmentations_list = uint8_augmentations_list

        input_transforms_list = [v2.ToDtype(torch.float32, scale=True)]

        if normalization_transform:
            input_transforms_list.append(normalization_transform)

        self.cpl_train_transform = v2.Compose(
            self.uint8_augmentations_list + input_transforms_list
        )
        self.cpl_val_transform = v2.Compose(input_transforms_list)
    
    def reset(self):
        for layer in self.network_stack:
            if callable(getattr(layer, "reset_parameters", None)):
                layer.reset_parameters()

    def udpate_normalization_transform(self, normalization_transform):
        self.cpl_train_transform = v2.Compose(
            self.uint8_augmentations_list
            + [v2.ToDtype(torch.float32, scale=True), normalization_transform]
        )
        self.cpl_val_transform = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), normalization_transform]
        )

    def forward(self, input):
        return self.network_stack(input)

    def cpl_train_forward(self, input):
        transformed_input = self.cpl_train_transform(input)
        return self.network_stack(transformed_input)

    def cpl_val_forward(self, input):
        transformed_input = self.cpl_val_transform(input)
        return self.network_stack(transformed_input)
