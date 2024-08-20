def log_train_step(batch_idx, loss, tepoch, config, epoch, writer, train_len):
    if batch_idx % config.train.log_interval == 0 and batch_idx > 0:
        niter = epoch * train_len + batch_idx

        loss = loss / config.train.log_interval

        writer.add_scalar("Loss/train", loss, niter)

        tepoch.set_postfix(loss=loss)

        return 0.0
    else:
        return loss


def log_val_step(loss, niter, epoch, writer):
    writer.add_scalar("Loss/val", loss, niter * epoch)

    print(f"Val Epoch {epoch}:        Loss: {loss}")
