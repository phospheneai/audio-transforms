"""
Demo script / notebook cell showing the full augmentation pipeline.

Run in a Jupyter notebook for interactive audio players and inline plots.
Run as a plain script for waveform plots saved to disk and OS audio playback.

Usage
-----
    python demo.py path/to/audio.wav [path/to/noises/dir]
"""

import sys

import matplotlib.pyplot as plt
import torch
import torchaudio

from authtransforms.pipeline import Compose, OneOf, SomeOf
from authtransforms.transforms import (
    AddGaussianNoise,
    Normalize,
    RandomApply,
    RandomClip,
    RandomGain,
    RandomPitchShift,
    RandomSpeedChange,
    TimeShift,
    ToMono,
)
from authtransforms.utils import audio_info, compare_audio, compare_play, plot_audio


# ---------------------------------------------------------------------------
# 1. Load audio
# ---------------------------------------------------------------------------

audio_path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
noise_dir = sys.argv[2] if len(sys.argv) > 2 else None

audio, sample_rate = torchaudio.load(audio_path)
print(f"Loaded: {audio_path}")
audio_info(audio, sample_rate, label="Original")

# ---------------------------------------------------------------------------
# 2. Build the augmentation pipeline
# ---------------------------------------------------------------------------

# Base transforms always applied
base_transforms = [
    ToMono(),
    RandomClip(sample_rate=sample_rate, clip_length=sample_rate * 5),  # 5-second clip
    Normalize(target_db=-3.0),
]

# One augmentation chosen randomly from a pool
augmentation_pool = [
    RandomSpeedChange(sample_rate=sample_rate, speed_factors=[0.9, 1.0, 1.1]),
    RandomPitchShift(sample_rate=sample_rate, semitones=(-2, 2)),
    TimeShift(max_shift=0.1),
]

# Extra noise — applied with 50% probability
noise_transforms = [
    RandomApply(AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.01), p=0.5),
]

# Add background noise augmentation if a noise directory was provided
if noise_dir:
    from authtransforms.transforms import RandomBackgroundNoise
    noise_transforms.append(
        RandomApply(
            RandomBackgroundNoise(sample_rate=sample_rate, noise_dir=noise_dir, min_snr_db=5, max_snr_db=20),
            p=0.7,
        )
    )

pipeline = Compose(
    base_transforms
    + [OneOf(augmentation_pool)]
    + noise_transforms
    + [Normalize(target_db=-3.0)]  # re-normalize after mixing
)

print("\nPipeline:")
print(pipeline)

# ---------------------------------------------------------------------------
# 3. Run augmentation
# ---------------------------------------------------------------------------

augmented = pipeline(audio)
audio_info(augmented, sample_rate, label="Augmented")

# ---------------------------------------------------------------------------
# 4. Visualize
# ---------------------------------------------------------------------------

fig = compare_audio(
    audio,
    augmented,
    sample_rate,
    title_before="Original",
    title_after="Augmented",
)
plt.show()

# Also show individual plots
fig_before = plot_audio(audio, sample_rate, title="Original Audio")
fig_after = plot_audio(augmented, sample_rate, title="Augmented Audio")
plt.show()

# ---------------------------------------------------------------------------
# 5. Play audio (interactive in Jupyter, OS player in scripts)
# ---------------------------------------------------------------------------

compare_play(audio, augmented, sample_rate, label_before="Original", label_after="Augmented")
