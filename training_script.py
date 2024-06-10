import torch
from DataPrep import DataPrepper
from lars import LARSSGD
from ModelActions import train_for_epochs, train_for_epochs_preloaded_cuda
from Network import Network
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

# learningRates = [0.1, 0.01, 0.001]
# batchSizes = [2048]
# weightDecay = [0.0001, 0.001, 0.0004]

epochs = 100
preload_cuda = True
uint8_augmentations_list = [
    v2.RandomResizedCrop(size=(32, 32), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
]
normalize = True


if preload_cuda:
    dp = DataPrepper(
        val_ratio=0.2,
    )
    train_loader, test_loader, val_loader = dp.get_dataloaders(
        batch_size=2048, num_workers=0, preload_cuda=True
    )

    model = Network(
        uint8_augmentations_list=uint8_augmentations_list,
        normalization_transform=(
            v2.Normalize(dp.training_channel_means, dp.training_channel_stds)
            if normalize
            else None
        ),
    )
else:
    dp = DataPrepper(
        val_ratio=0.2,
        normalize=normalize,
        uint8_augmentations_list=uint8_augmentations_list,
    )
    train_loader, test_loader, val_loader = dp.get_dataloaders(
        batch_size=2048,
        num_workers=2,
        pin_memory=True,
        preload_cuda=False,
    )

    model = Network()

model.to("cuda")

loss_fn = torch.nn.CrossEntropyLoss()

writer = SummaryWriter()

optimizer = torch.optim.SGD(model.parameters())
# optimizer = LARSSGD(model.parameters(), lr=0.01)
lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

if preload_cuda:
    train_for_epochs_preloaded_cuda(
        model,
        train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        val_loader=val_loader,
        epochs=epochs,
        epoch_lr_scheduler=lr_scheduler,
        tb_writer=writer,
        save_model_epochs_period=50,
    )
else:
    train_for_epochs(
        model,
        train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        val_loader=val_loader,
        epochs=epochs,
        epoch_lr_scheduler=lr_scheduler,
        tb_writer=writer,
        save_model_epochs_period=50,
    )
