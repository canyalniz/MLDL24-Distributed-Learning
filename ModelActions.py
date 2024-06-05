import datetime
import torch
import os


def train_for_epochs(
    model,
    train_loader,
    optimizer,
    loss_fn,
    val_loader=None,
    epochs=5,
    epoch_lr_scheduler=None,
    step_lr_scheduler=None,
    tb_writer=None,
    save_model_epochs_period=None,
    skip_epochs_before_saving=0,
    model_save_dir="model_checkpoints",
    run_id=None,
):
    global_steps = 0
    model.train(True)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_path = os.path.join(model_save_dir, run_id)
    os.makedirs(run_path)

    for ep in range(epochs):
        epoch_loss_value = 0
        success_train_pred = 0
        num_samples = 0
        for data, target in train_loader:
            data = data.to("cuda")
            target = target.to("cuda")

            optimizer.zero_grad()

            logits = model(data)
            loss = loss_fn(logits, target)
            loss.backward()

            optimizer.step()

            global_steps = global_steps + 1
            tb_writer.add_scalar("Train Loss per sample (steps)", loss, global_steps)

            if step_lr_scheduler:
                step_lr_scheduler.step()

            epoch_loss_value += loss
            success_train_pred += logits.argmax(1).eq(target).sum()
            num_samples += target.shape[0]

        tb_writer.add_scalar(
            "Train Loss per sample (epochs)", epoch_loss_value / num_samples, ep + 1
        )
        tb_writer.add_scalar(
            "Train Accuracy (epochs)", success_train_pred / num_samples, ep + 1
        )

        if epoch_lr_scheduler:
            epoch_lr_scheduler.step()

        if save_model_epochs_period:
            if ep > skip_epochs_before_saving and ep % save_model_epochs_period == 0:
                identifier = f"{run_id}_ep{ep}"
                path = os.path.join(run_path, identifier)
                torch.save(model, path)

        if val_loader:
            val_loss = 0
            success_val_pred = 0
            num_samples = 0

            model.eval()

            for data, target in val_loader:
                data = data.to("cuda")
                target = target.to("cuda")

                with torch.no_grad():
                    logits = model(data)

                val_loss += loss_fn(logits, target)
                success_val_pred += logits.argmax(1).eq(target).sum()
                num_samples += target.shape[0]

            tb_writer.add_scalar(
                "Validation Loss per sample (epochs)", val_loss / num_samples, ep + 1
            )
            tb_writer.add_scalar(
                "Validation Accuracy (epochs)", success_val_pred / num_samples, ep + 1
            )

            model.train(True)


def single_optimizer_step(
    model,
    batch,
    optimizer,
    loss_fn,
):
    data, target = batch
    data = data.to("cuda")
    target = target.to("cuda")

    optimizer.zero_grad()

    logits = model(data)
    loss = loss_fn(logits, target)
    loss.backward()

    optimizer.step()
