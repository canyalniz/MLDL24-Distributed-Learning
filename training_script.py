import torch
from DataPrep import DataPrepper
from Optimizers import LARSSGD
from ModelActions import train_for_epochs
from Network import Network
from torch.utils.tensorboard import SummaryWriter


model = Network()
model.to("cuda")
dp = DataPrepper(val_ratio=0.2)
train_loader, test_loader, val_loader = dp.get_dataloaders(batch_size=4)
writer = SummaryWriter()
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters())
optimizer = LARSSGD(model.parameters())
train_for_epochs(
    model,
    train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    val_loader=val_loader,
    epochs=5,
    tb_writer=writer,
)
