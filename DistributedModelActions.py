import datetime
import torch
import os
from dataclasses import dataclass
from Network import Network
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy


def zero_module(module):
    """
    Zero out the parameters of the module in-place.
    """
    for p in module.parameters():
        p.detach().zero_()


def zero_param_list(param_list):
    """
    Zero out the parameters in the list in-place.
    """
    for p in param_list:
        p.detach().zero_()


@dataclass
class Worker:
    model: Network
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler


def train_for_epochs_localsgd(
    model,
    train_loaders,
    optimizer,
    loss_fn,
    val_loader=None,
    epochs=5,
    cosine_annealing_scheduler=True,
    tb_writer=None,
    save_model_epochs_period=None,
    skip_epochs_before_saving=0,
    model_save_dir="model_checkpoints",
    run_id=None,
    local_steps=2,
):
    local_copies = []

    optimizer_class = type(optimizer)

    num_workers = len(train_loaders)

    for _ in range(num_workers):
        local_model = copy.deepcopy(model)
        local_model.to("cuda")
        local_model.train(True)
        local_optimizer = optimizer_class(
            local_model.parameters(), **optimizer.defaults
        )
        local_epoch_lr_scheduler = (
            CosineAnnealingLR(local_optimizer, epochs)
            if cosine_annealing_scheduler
            else None
        )
        worker = Worker(local_model, local_optimizer, local_epoch_lr_scheduler)
        local_copies.append(worker)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_path = os.path.join(model_save_dir, run_id)
    os.makedirs(run_path)

    steps = 0

    for ep in range(epochs):
        print(ep)
        epoch_loss_value = 0
        success_train_pred = 0
        num_samples = 0

        train_iterators = [iter(t_l) for t_l in train_loaders]

        continuing = len(train_iterators) * [True]

        while any(continuing):
            for i, (t_i, local_copy) in enumerate(zip(train_iterators, local_copies)):
                try:
                    data, target = next(t_i)
                except StopIteration:
                    continuing[i] = False
                    pass

                local_copy.optimizer.zero_grad()

                logits = local_copy.model.cpl_train_forward(data)
                loss = loss_fn(logits, target)
                loss.backward()

                local_copy.optimizer.step()

                epoch_loss_value += loss
                success_train_pred += logits.argmax(1).eq(target).sum()
                num_samples += target.shape[0]

            steps = steps + 1

            if steps % local_steps:
                # prepare main model to receive sync by zeroing out parameters
                zero_module(model)

                # sync changes, average over local copies
                for local_copy in local_copies:
                    for local_param, main_param in zip(
                        local_copy.model.parameters(), model.parameters()
                    ):
                        with torch.no_grad():
                            main_param.add_(local_param, alpha=1 / num_workers)

                # update local copies to main parameters
                for local_copy in local_copies:
                    for local_param, main_param in zip(
                        local_copy.model.parameters(), model.parameters()
                    ):
                        with torch.no_grad():
                            local_param.copy_(main_param)

        tb_writer.add_scalar(
            "Train Loss per sample (epochs)", epoch_loss_value / num_samples, ep + 1
        )
        tb_writer.add_scalar(
            "Train Accuracy (epochs)", success_train_pred / num_samples, ep + 1
        )

        if cosine_annealing_scheduler:
            for local_copy in local_copies:
                local_copy.lr_scheduler.step()

        if save_model_epochs_period:
            if ep > skip_epochs_before_saving and ep % save_model_epochs_period == 0:
                identifier = f"{run_id}_ep{ep}"
                path = os.path.join(run_path, identifier)
                torch.save(model, path)

        if val_loader:
            val_loss = 0
            success_val_pred = 0
            num_samples = 0

            for data, target in val_loader:

                with torch.no_grad():
                    logits = model.cpl_val_forward(data)

                val_loss += loss_fn(logits, target)
                success_val_pred += logits.argmax(1).eq(target).sum()
                num_samples += target.shape[0]

            tb_writer.add_scalar(
                "Validation Loss per sample (epochs)",
                val_loss / num_samples,
                ep + 1,
            )
            tb_writer.add_scalar(
                "Validation Accuracy (epochs)",
                success_val_pred / num_samples,
                ep + 1,
            )


def train_for_epochs_slowmo(
    model,
    train_loaders,
    inner_optimizer,
    loss_fn,
    val_loader=None,
    epochs=5,
    cosine_annealing_scheduler=True,
    tb_writer=None,
    save_model_epochs_period=None,
    skip_epochs_before_saving=0,
    model_save_dir="model_checkpoints",
    run_id=None,
    inner_loop_steps=2,
    outer_lr=1,
    outer_momentum_factor=0.5,
):
    local_copies = []

    optimizer_class = type(inner_optimizer)

    num_workers = len(train_loaders)

    for _ in range(num_workers):
        local_model = copy.deepcopy(model)
        local_model.to("cuda")
        local_model.train(True)
        local_optimizer = optimizer_class(
            local_model.parameters(), **inner_optimizer.defaults
        )
        local_lr_scheduler = (
            CosineAnnealingLR(local_optimizer, epochs)
            if cosine_annealing_scheduler
            else None
        )
        worker = Worker(local_model, local_optimizer, local_lr_scheduler)
        local_copies.append(worker)

    # will represent x_{t, \tao}
    params_buffer = []
    for param in model.parameters():
        params_buffer.append(param.detach().clone())

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_path = os.path.join(model_save_dir, run_id)
    os.makedirs(run_path)

    outer_optimizer = torch.optim.SGD(
        model.parameters(), lr=outer_lr, momentum=outer_momentum_factor
    )

    steps = 0

    for ep in range(epochs):
        print(ep)
        epoch_loss_value = 0
        success_train_pred = 0
        num_samples = 0

        train_iterators = [iter(t_l) for t_l in train_loaders]

        continuing = len(train_iterators) * [True]

        while any(continuing):
            for i, (t_i, local_copy) in enumerate(zip(train_iterators, local_copies)):
                try:
                    data, target = next(t_i)
                except StopIteration:
                    continuing[i] = False
                    pass

                local_copy.optimizer.zero_grad()

                logits = local_copy.model.cpl_train_forward(data)
                loss = loss_fn(logits, target)
                loss.backward()

                local_copy.optimizer.step()

                epoch_loss_value += loss
                success_train_pred += logits.argmax(1).eq(target).sum()
                num_samples += target.shape[0]

            steps = steps + 1

            if steps % inner_loop_steps:
                # prepare params buffer to receive sync by zeroing out parameters
                zero_param_list(params_buffer)

                # Exact Average
                for local_copy in local_copies:
                    for local_param, buff in zip(
                        local_copy.model.parameters(), params_buffer
                    ):
                        with torch.no_grad():
                            buff.add_(local_param, alpha=1 / num_workers)

                current_inner_lr = local_copies[0].lr_scheduler.get_last_lr()[0]

                # prepare delta for optimizer step
                for exact_average, main_param in zip(params_buffer, model.parameters()):
                    # no_grad to avoid grad grad buildup
                    with torch.no_grad():
                        # assigning to grad directly, no need to call zero_grad on outer optimizer
                        main_param.grad = main_param.sub(exact_average)
                        # divide by the current inner lr to make the momentum buffer invariant
                        main_param.div_(current_inner_lr)

                for g in outer_optimizer.param_groups:
                    # this assumes outer_lr is constant
                    # this would need to be modified if a lr scheduler were to be applied to the outer optimizer
                    g["lr"] = outer_lr * current_inner_lr

                outer_optimizer.step()

                # TODO: Figure out local momentum buffer situation

                # update local copies to main parameters
                for local_copy in local_copies:
                    for local_param, main_params in zip(
                        local_copy.model.parameters(), model.parameters()
                    ):
                        with torch.no_grad():
                            local_param.copy_(main_params)

                if cosine_annealing_scheduler:
                    for local_copy in local_copies:
                        local_copy.lr_scheduler.step()

        tb_writer.add_scalar(
            "Train Loss per sample (epochs)", epoch_loss_value / num_samples, ep + 1
        )
        tb_writer.add_scalar(
            "Train Accuracy (epochs)", success_train_pred / num_samples, ep + 1
        )

        if save_model_epochs_period:
            if ep > skip_epochs_before_saving and ep % save_model_epochs_period == 0:
                identifier = f"{run_id}_ep{ep}"
                path = os.path.join(run_path, identifier)
                torch.save(model, path)

        if val_loader:
            val_loss = 0
            success_val_pred = 0
            num_samples = 0

            for data, target in val_loader:

                with torch.no_grad():
                    logits = model.cpl_val_forward(data)

                val_loss += loss_fn(logits, target)
                success_val_pred += logits.argmax(1).eq(target).sum()
                num_samples += target.shape[0]

            tb_writer.add_scalar(
                "Validation Loss per sample (epochs)",
                val_loss / num_samples,
                ep + 1,
            )
            tb_writer.add_scalar(
                "Validation Accuracy (epochs)",
                success_val_pred / num_samples,
                ep + 1,
            )
