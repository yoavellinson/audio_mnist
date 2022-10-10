import torchaudio
from model import AudioMNIST
import torch
import glob
model = AudioMNIST.load_from_checkpoint(
    '/root/audio_classification/audio_mnist/checkpoints/audio_mnist.ckpt')
model.eval()
print('model loaded')

classes = range(10)
t = 0
f = 0
acc = 0
for wav_path in glob.glob('/root/audio_classification/audio_mnist/yoav_clean/*.wav', recursive=True):
    wav, rate = torchaudio.load(wav_path)
    wav = wav.unsqueeze(0)
    y_hat = model(wav)
    pred = int(torch.argmax(y_hat, dim=-1))
    print(f'Target is:{wav_path.split(".")[-2]}, Prediction is:{pred}')
