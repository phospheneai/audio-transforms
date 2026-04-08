"""
Audio augmentation transforms for PyTorch, inspired by:
https://jonathanbgn.com/2021/08/30/audio-augmentation.html

Transforms follow the torchvision convention: callable objects with __call__
that accept a (channels, samples) tensor and return a transformed tensor.
"""

import math
import os
import pathlib
import random

import torch
import torchaudio
import torchaudio.transforms as T


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------


class RandomClip:
    """Extract a random fixed-length clip from the audio.

    Args:
        sample_rate: Sample rate of the audio.
        clip_length: Desired clip length in samples.
        vad: If True, apply Voice Activity Detection to trim leading/trailing
             silence after clipping.
        vad_trigger_level: VAD sensitivity (higher = more aggressive trimming).
    """

    def __init__(self, sample_rate: int, clip_length: int, vad: bool = True, vad_trigger_level: float = 7.0):
        self.clip_length = clip_length
        self.vad = T.Vad(sample_rate=sample_rate, trigger_level=vad_trigger_level) if vad else None

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        audio_length = audio.shape[-1]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length - self.clip_length)
            audio = audio[..., offset : offset + self.clip_length]
        if self.vad is not None:
            audio = self.vad(audio)
        return audio

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(clip_length={self.clip_length})"


class RandomSpeedChange:
    """Randomly perturb audio playback speed.

    Resamples back to the original sample rate so the output length stays
    approximately the same.  Uses torchaudio's SoX effects backend.

    Args:
        sample_rate: Sample rate of the audio.
        speed_factors: Sequence of speed multipliers to choose from.
                       Default is (0.9, 1.0, 1.1) as used in the literature.
    """

    def __init__(self, sample_rate: int, speed_factors=(0.9, 1.0, 1.1)):
        self.sample_rate = sample_rate
        self.speed_factors = list(speed_factors)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        speed_factor = random.choice(self.speed_factors)
        if speed_factor == 1.0:
            return audio

        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio, self.sample_rate, sox_effects
        )
        return transformed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(speed_factors={self.speed_factors})"


class RandomBackgroundNoise:
    """Mix the audio with a random noise file at a random SNR level.

    Args:
        sample_rate: Target sample rate.
        noise_dir: Directory (searched recursively) containing .wav noise files.
        min_snr_db: Minimum signal-to-noise ratio in dB.
        max_snr_db: Maximum signal-to-noise ratio in dB.
    """

    def __init__(self, sample_rate: int, noise_dir: str, min_snr_db: float = 0, max_snr_db: float = 15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        noise_path = pathlib.Path(noise_dir)
        if not noise_path.exists():
            raise IOError(f"Noise directory `{noise_dir}` does not exist")
        self.noise_files = list(noise_path.glob("**/*.wav"))
        if not self.noise_files:
            raise IOError(f"No .wav files found in `{noise_dir}`")

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        noise_file = random.choice(self.noise_files)
        effects = [
            ["remix", "1"],              # convert to mono
            ["rate", str(self.sample_rate)],  # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(noise_file, effects, normalize=True)

        audio_length = audio.shape[-1]
        noise_length = noise.shape[-1]

        if noise_length > audio_length:
            offset = random.randint(0, noise_length - audio_length)
            noise = noise[..., offset : offset + audio_length]
        elif noise_length < audio_length:
            # tile noise to cover the audio
            repeats = math.ceil(audio_length / noise_length)
            noise = noise.repeat(1, repeats)[..., :audio_length]

        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / (audio_power + 1e-9)

        return (scale * audio + noise) / 2

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"snr=[{self.min_snr_db}, {self.max_snr_db}] dB, "
            f"n_files={len(self.noise_files)})"
        )


class RandomPitchShift:
    """Shift audio pitch by a random number of semitones.

    Args:
        sample_rate: Sample rate of the audio.
        semitones: Range tuple (min, max) of semitone shifts.
    """

    def __init__(self, sample_rate: int, semitones: tuple = (-2, 2)):
        self.sample_rate = sample_rate
        self.semitones = semitones

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        n = random.uniform(*self.semitones)
        if n == 0:
            return audio
        sox_effects = [["pitch", str(int(n * 100))], ["rate", str(self.sample_rate)]]
        transformed, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio, self.sample_rate, sox_effects
        )
        return transformed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(semitones={self.semitones})"


class RandomGain:
    """Scale audio amplitude by a random gain factor (in dB).

    Args:
        min_gain_db: Minimum gain in dB.
        max_gain_db: Maximum gain in dB.
    """

    def __init__(self, min_gain_db: float = -6, max_gain_db: float = 6):
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        gain = 10 ** (gain_db / 20)
        return audio * gain

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gain_db=[{self.min_gain_db}, {self.max_gain_db}])"


class AddGaussianNoise:
    """Add Gaussian white noise at a random amplitude.

    Args:
        min_amplitude: Minimum noise standard deviation.
        max_amplitude: Maximum noise standard deviation.
    """

    def __init__(self, min_amplitude: float = 0.001, max_amplitude: float = 0.015):
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        noise = torch.randn_like(audio) * amplitude
        return audio + noise

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"amplitude=[{self.min_amplitude}, {self.max_amplitude}])"
        )


class TimeShift:
    """Shift the audio waveform in time, wrapping or zero-padding.

    Args:
        max_shift: Maximum shift as a fraction of the total length (0–1).
        roll: If True, wrap the audio (circular shift). If False, zero-pad.
    """

    def __init__(self, max_shift: float = 0.2, roll: bool = False):
        self.max_shift = max_shift
        self.roll = roll

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        length = audio.shape[-1]
        shift = random.randint(-int(length * self.max_shift), int(length * self.max_shift))
        if shift == 0:
            return audio
        if self.roll:
            return torch.roll(audio, shift, dims=-1)
        # zero-pad version
        result = torch.zeros_like(audio)
        if shift > 0:
            result[..., shift:] = audio[..., : length - shift]
        else:
            result[..., : length + shift] = audio[..., -shift:]
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_shift={self.max_shift}, roll={self.roll})"


class SpecAugment:
    """Apply SpecAugment on the spectrogram (time & frequency masking).

    This transform expects a raw audio tensor and internally computes a
    spectrogram, applies masking, then returns the masked spectrogram.
    Use it as the final step in a pipeline when your model consumes spectrograms.

    Args:
        sample_rate: Sample rate.
        n_fft: FFT size.
        n_mels: Number of mel filterbanks.
        freq_mask_param: Maximum frequency mask width.
        time_mask_param: Maximum time mask width.
        n_freq_masks: Number of frequency masks.
        n_time_masks: Number of time masks.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 400,
        n_mels: int = 80,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        self.mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        spec = self.mel_spec(audio)
        for _ in range(self.n_freq_masks):
            spec = self.freq_masking(spec)
        for _ in range(self.n_time_masks):
            spec = self.time_masking(spec)
        return spec

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_freq={self.n_freq_masks}, n_time={self.n_time_masks})"


class RandomApply:
    """Apply a transform with probability p (mirrors torchvision.transforms.RandomApply).

    Args:
        transform: A callable transform.
        p: Probability of applying the transform (0–1).
    """

    def __init__(self, transform, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.transform(audio)
        return audio

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, transform={self.transform})"


class Normalize:
    """Normalize audio to peak amplitude of 1.0 (or a target level).

    Args:
        target_db: Target peak level in dBFS. Default -3 dBFS. Use None for 0 dBFS.
    """

    def __init__(self, target_db: float = -3.0):
        self.target_level = 10 ** (target_db / 20)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        peak = audio.abs().max()
        if peak < 1e-9:
            return audio
        return audio / peak * self.target_level

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_db={20 * math.log10(self.target_level):.1f})"


class ToMono:
    """Convert multi-channel audio to mono by averaging channels."""

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[0] > 1:
            return audio.mean(dim=0, keepdim=True)
        return audio

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class DummyTransform:
    """A dummy transform that does nothing. Useful for testing and debugging."""

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return audio

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()" 