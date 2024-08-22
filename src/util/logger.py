def log_train_step(batch_idx, loss, pbar, train_len, config, epoch, writer):
    if batch_idx % config.train.log_interval == 0 and batch_idx > 0:
        niter = epoch * train_len + batch_idx

        loss = loss / config.train.log_interval

        writer.add_scalar("Loss/train", loss, niter)

        pbar.set_postfix(loss=loss)

        return 0.0
    else:
        return loss


def log_val_step(batch_idx, loss, pbar, val_len, config, epoch, writer):
    if batch_idx % config.train.log_interval == 0 and batch_idx > 0:
        niter = epoch * val_len + batch_idx
        loss = loss / batch_idx
        writer.add_scalar("Loss/val", loss, niter)
        pbar.set_postfix(loss=loss)
