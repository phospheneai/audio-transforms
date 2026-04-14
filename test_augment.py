import torch
import torchaudio

from tansforms import AddGaussianNoise, RandomPitchShift
from pipeline import Compose

# Load audio
audio, sr = torchaudio.load("sample.wav")

# Make mono (optional)
audio = audio.mean(dim=0, keepdim=True)

# Test individual transform
noise_aug = AddGaussianNoise(0.001, 0.01)
noisy_audio = noise_aug(audio)

torchaudio.save("noisy.wav", noisy_audio, sr)

# Test pipeline
pipeline = Compose([
    RandomPitchShift(sr),
    AddGaussianNoise(0.001, 0.01),
])

aug_audio = pipeline(audio)
torchaudio.save("pipeline.wav", aug_audio, sr)

print("Testing complete. Check output .wav files.")