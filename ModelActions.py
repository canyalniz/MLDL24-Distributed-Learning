import torch


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
):
    global_steps = 0
    model.train(True)

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
