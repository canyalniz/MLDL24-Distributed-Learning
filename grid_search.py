import itertools
import torch
import argparse
from DataPrep import DataPrepper, DataPrepperCuda
from lars import LARSSGD
from lamb import LAMBAdamW
from ModelActions import train_for_epochs, train_for_epochs_preloaded_cuda
from Network import Network
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument("--optimizer", dest="optimizer")
parser.add_argument("--no-preload", action="store_false", dest="preload_cuda")
parser.add_argument("--num-folds", "--k", type=int, dest="num_folds")
parser.add_argument("--epochs", type=int, dest="epochs")
parser.add_argument("--download-data", action="store_true", dest="download_data")
args = parser.parse_args()

epochs = args.epochs
if epochs is None or epochs < 0:
    raise ValueError("Please provide a positive epochs option.")

num_folds = args.num_folds
if num_folds is None:
    num_folds = 1
elif num_folds < 1:
    raise ValueError("num_folds must be greater than 0.")

optimizers_dict = {
    "sgd":torch.optim.SGD,
    "lars":LARSSGD,
    "lamb":LAMBAdamW
}

optimizer_option = args.optimizer.lower()
try:
    optimizer_class = optimizers_dict[optimizer_option]
except KeyError:
    raise ValueError(f"Unrecognized optimizer option. Available options are {list(optimizers_dict.keys())}")

preload_cuda = args.preload_cuda

uint8_augmentations_list = [
    v2.RandomResizedCrop(size=(32, 32), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
]
normalize = True

lr_space = [0.1, 0.01, 0.001]
batch_size_space = [2048]
wd_space = [0.0001, 0.001, 0.0004]

dp = (
    DataPrepperCuda(val_ratio=0.2, num_folds=num_folds, download=args.download_data) if preload_cuda
    else DataPrepper(
            val_ratio=0.2,
            normalize=normalize,
            uint8_augmentations_list=uint8_augmentations_list,
            num_folds=num_folds,
            download=args.download_data
        )
)

model = (
    Network(uint8_augmentations_list=uint8_augmentations_list) if preload_cuda
    else Network()
)

model.to("cuda")
loss_fn = torch.nn.CrossEntropyLoss()

for lr, batch_size, wd in itertools.product(lr_space, batch_size_space, wd_space):
    optimizer = optimizer_class(model.parameters(), weight_decay=wd, lr=lr)
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    
    model.reset()

    run_id = f"lr{lr}_bs{batch_size}_wd{wd}"
    writer = SummaryWriter("runs/"+run_id)

    if preload_cuda:
        if num_folds == 1:
            train_loader, val_loader, test_loader = dp.dataloaders(batch_size=batch_size)
            
            if normalize:
                model.udpate_normalization_transform(v2.Normalize(dp.training_channel_means, dp.training_channel_stds))
            
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
                run_id=run_id
            )
        else:
            for i, (train_loader, val_loader) in enumerate(
                dp.k_fold_dataloaders(batch_size=batch_size)
            ):
                if normalize:
                    model.udpate_normalization_transform(
                        v2.Normalize(
                            dp.kfold_training_channel_means[i],
                            dp.kfold_training_channel_stds[i],
                        )
                    )

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
                    run_id=run_id
                )
    else:
        if num_folds == 1:
            train_loader, val_loader, test_loader = dp.dataloaders(batch_size=batch_size, num_workers=5, pin_memory=True)
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
                run_id=run_id
            )
        else:
            for i, (train_loader, val_loader) in enumerate(
                dp.k_fold_dataloaders(batch_size=batch_size, num_workers=7, pin_memory=True)
            ):
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
                    run_id=run_id
                )