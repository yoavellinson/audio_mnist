import os
from random import shuffle
import sys
import gc
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import glob


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class Dataset(Dataset):
    def __init__(self, cfg):
        self.wav_paths = glob.glob(cfg.wav_paths+'/*.wav', recursive=True)
        shuffle(self.wav_paths)
        self.sr = cfg.sr
        self.classes = cfg.classes
        self.n_classes = cfg.n_classes
        self.batch_size = cfg.batch_size
        self.indexes = np.arange(len(self.wav_paths))

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        label = int(wav_path.split('/')[-1][0])
        # generate a batch of time data
        X = np.empty((1, int(self.sr), 1), dtype=np.float32)
        Y = np.empty((1, self.n_classes), dtype=np.float32)
        wav, rate = torchaudio.load(wav_path)
        X[0, ] = wav.reshape(-1, 1)
        Y[0, ] = to_categorical(label, self.n_classes)

        return X, Y


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        Dataset(self.cfg)

    def setup(self, stage=None):
        dataset = Dataset(self.cfg)
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        self.train, self.val = random_split(dataset, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, drop_last=True)
