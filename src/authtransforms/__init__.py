"""
authtransforms — Audio augmentation pipeline for PyTorch.

Inspired by: https://jonathanbgn.com/2021/08/30/audio-augmentation.html
"""

from .pipeline import Compose, Identity, Lambda, OneOf, RandomOrder, SomeOf
from .transforms import (
    AddGaussianNoise,
    Normalize,
    RandomApply,
    RandomBackgroundNoise,
    RandomClip,
    RandomGain,
    RandomPitchShift,
    RandomSpeedChange,
    SpecAugment,
    TimeShift,
    ToMono,
)
from .utils import (
    audio_info,
    compare_audio,
    compare_play,
    play_audio,
    plot_audio,
    plot_spectrogram,
    plot_waveform,
)

__all__ = [
    # Pipeline combinators
    "Compose",
    "OneOf",
    "SomeOf",
    "RandomOrder",
    "Identity",
    "Lambda",
    # Transforms
    "RandomClip",
    "RandomSpeedChange",
    "RandomBackgroundNoise",
    "RandomPitchShift",
    "RandomGain",
    "AddGaussianNoise",
    "TimeShift",
    "SpecAugment",
    "RandomApply",
    "Normalize",
    "ToMono",
    # Utilities
    "plot_waveform",
    "plot_spectrogram",
    "plot_audio",
    "compare_audio",
    "play_audio",
    "compare_play",
    "audio_info",
]
