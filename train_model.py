from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig
import hydra
import torch
from model import AudioMNIST
from dataset import ClassificationDataModule


@hydra.main(config_name='train_cfg')
def main(cfg: DictConfig) -> None:

    dm = ClassificationDataModule(cfg)
    model = AudioMNIST(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename='bs_{0}_lr_{1}'.format(cfg.batch_size, cfg.lr),
        verbose=cfg.verbose
    )

    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=70,
        verbose=cfg.verbose,
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback, stop_callback],
        # precision=16,
        accelerator='auto',
        max_epochs=2000
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint(
        cfg.checkpoint_dir+'/audio_mnist.ckpt')
    torch.save(model.state_dict(), '/root/audio_classification/audio_mnist/model_v1.pth')


if __name__ == "__main__":
    main()
