import torch
from DataPrep import DataPrepper, DataPrepperCuda
from lars import LARSSGD
from lamb import LAMBAdamW
from DistributedModelActions import train_for_epochs_localsgd, train_for_epochs_slowmo
from Network import Network
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

# learningRates = [0.1, 0.01, 0.001]
# batchSizes = [2048]
# weightDecay = [0.0001, 0.001, 0.0004]

# preload_cuda = True

batch_size = 2048
epochs = 5
normalize = True

uint8_augmentations_list = [
    v2.RandomResizedCrop(size=(32, 32), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
]

num_folds = 1

partitions = 8
local_steps = 40


# if preload_cuda:
dp = DataPrepperCuda(val_ratio=0.2, num_folds=num_folds)

if num_folds == 1:
    train_loaders, val_loader, test_loader = dp.n_partitions_dataloaders(
        n=partitions, batch_size=batch_size
    )

model = Network(
    uint8_augmentations_list=uint8_augmentations_list,
    normalization_transform=(
        v2.Normalize(dp.training_channel_means, dp.training_channel_stds)
        if normalize and num_folds == 1
        else None
    ),
)
# else:
#     dp = DataPrepper(
#         val_ratio=0.2,
#         normalize=normalize,
#         uint8_augmentations_list=uint8_augmentations_list,
#         num_folds=num_folds,
#     )
#     if num_folds == 1:
#         train_loader, val_loader, test_loader = dp.dataloaders(
#             batch_size=batch_size,
#             num_workers=5,
#             pin_memory=True,
#         )

#     model = Network()

model.to("cuda")

loss_fn = torch.nn.CrossEntropyLoss()

writer = SummaryWriter()

optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.001, lr=0.1)
# optimizer = LARSSGD(model.parameters(), lr=0.01)
# optimizer = LAMBAdamW(model.parameters(), lr=0.01)
lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

# if preload_cuda:
if num_folds == 1:
    train_for_epochs_slowmo(
        model,
        train_loaders,
        inner_optimizer=optimizer,
        loss_fn=loss_fn,
        val_loader=val_loader,
        epochs=epochs,
        cosine_annealing_scheduler=True,
        tb_writer=writer,
        save_model_epochs_period=50,
        inner_loop_steps=local_steps,
        outer_lr=1,
        outer_momentum_factor=0.5,
    )
    # train_for_epochs_localsgd(
    #     model,
    #     train_loaders,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    #     val_loader=val_loader,
    #     epochs=epochs,
    #     cosine_annealing_scheduler=True,
    #     tb_writer=writer,
    #     save_model_epochs_period=50,
    #     local_steps=local_steps,
    # )
else:
    for i, (train_loaders, val_loader) in enumerate(
        dp.k_fold_n_partition_dataloaders(n=partitions, batch_size=batch_size)
    ):
        if normalize:
            model.udpate_normalization_transform(
                v2.Normalize(
                    dp.kfold_training_channel_means[i],
                    dp.kfold_training_channel_stds[i],
                )
            )

        train_for_epochs_localsgd(
            model,
            train_loaders,
            optimizer=optimizer,
            loss_fn=loss_fn,
            val_loader=val_loader,
            epochs=epochs,
            cosine_annealing_scheduler=True,
            tb_writer=writer,
            save_model_epochs_period=50,
            local_steps=local_steps,
        )
# else:
#     if num_folds == 1:
#         train_for_epochs(
#             model,
#             train_loader,
#             optimizer=optimizer,
#             loss_fn=loss_fn,
#             val_loader=val_loader,
#             epochs=epochs,
#             epoch_lr_scheduler=lr_scheduler,
#             tb_writer=writer,
#             save_model_epochs_period=50,
#         )
#     else:
#         for i, (train_loader, val_loader) in enumerate(
#             dp.k_fold_dataloaders(batch_size=batch_size, num_workers=7, pin_memory=True)
#         ):
#             train_for_epochs(
#                 model,
#                 train_loader,
#                 optimizer=optimizer,
#                 loss_fn=loss_fn,
#                 val_loader=val_loader,
#                 epochs=epochs,
#                 epoch_lr_scheduler=lr_scheduler,
#                 tb_writer=writer,
#                 save_model_epochs_period=50,
#             )
