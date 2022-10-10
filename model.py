from locale import normalize
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram
import torch
import torchaudio
from torch import nn, optim
from torch.nn import functional as F
import torch.nn as nn
import pytorch_lightning as pl


class AudioMNIST(pl.LightningModule):
    def __init__(self, cfg):
        super(AudioMNIST, self).__init__()
        self.in_channels = 1
        self.out_channels = cfg.n_classes
        self.lr = cfg.lr
        self.optim = cfg.optim
        self.sr = cfg.sr
        self.batch_size = cfg.batch_size
        self.transform = MelSpectrogram(sample_rate=self.sr, n_fft=512, win_length=400,
                                        hop_length=160, n_mels=128, window_fn=torch.hann_window, norm='slaney', normalized=True, power=1)
        self.criterion = torch.nn.CrossEntropyLoss()

        # layers
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.maxpool = nn.MaxPool2d(2, 2, 1)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

        self.save_hyperparameters()

    def forward(self, x):
        x1 = self.transform(x)  # [32,1,128,101]
        x2 = self.bn(x1)
        x3 = self.maxpool(nn.ReLU()(self.conv1(x2)))
        x4 = self.maxpool(nn.ReLU()(self.conv2(x3)))
        x5 = self.maxpool(nn.ReLU()(self.conv3(x4)))
        x6 = nn.Dropout2d(p=0.2)(self.maxpool(nn.ReLU()(self.conv4(x5))))
        x7 = nn.Flatten()(x6)
        x8 = nn.ReLU()(self.fc1(x7))
        x9 = nn.ReLU()(self.fc2(x8))
        x10 = self.fc3(x9)

        return x10

    def training_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.squeeze(), y.squeeze()
        x = x.unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        tensorboard_logs = {'train_loss': loss}
        # return loss
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.squeeze(), y.squeeze()
        x = x.unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.optim == 'adam':
            return optim.Adam(self.parameters(), lr=self.lr)  # eps=1e-07



