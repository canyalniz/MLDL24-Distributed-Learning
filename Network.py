import torch


class Network(torch.nn.Module):
    def __init__(self):
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

    def forward(self, input):
        return self.network_stack(input)
