import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample
from tqdm import tqdm
import librosa
import torchaudio
import torch


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def resamp(wav_path, args):
    wav, rate = torchaudio.load(wav_path)
    re_wav = resampler(wav)
    return re_wav


def clean_wavs(args):
    src_root = args.src_root

    dst_root = args.dst_root
    wav_paths = glob('{}/*.wav'.format(src_root), recursive=True)
  
    for wav in tqdm(wav_paths):
        re_wav = resamp(wav, args)
        mask, _ = envelope(re_wav.numpy().squeeze(), args.sr, args.threshold)
        en_wav = re_wav.squeeze()[mask]
        new_name = dst_root+'/'+wav.split('/')[-1]
        z = torch.zeros((args.sr-en_wav.shape[0]))
        torchaudio.save(new_name, torch.cat((en_wav, z)).unsqueeze(0), args.sr)
  


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='/Users/yoavellinson/code/Audio-Classification/audio_mnist/shiraz',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='/Users/yoavellinson/code/Audio-Classification/audio_mnist/clean_shiraz',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')
    parser.add_argument('--threshold', type=str, default=0.0006,
                        help='threshold magnitude for np.int16 dtype')
    parser.add_argument('--data_sr', type=int, default=44100,
                        help='the sampling rate of the dataset')
    args, _ = parser.parse_known_args()

    resampler = torchaudio.transforms.Resample(args.data_sr, args.sr)
    clean_wavs(args)
