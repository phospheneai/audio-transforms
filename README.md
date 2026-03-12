# authtransforms

Audio augmentation pipeline for PyTorch — a `torchvision.transforms`-style API for audio tensors.

Inspired by [Simple Audio Augmentation with PyTorch](https://jonathanbgn.com/2021/08/30/audio-augmentation.html).

---

## Getting Started

### What is Poetry?

Poetry is a tool that manages your Python dependencies and virtual environment for you — think of it like a smarter `pip` that also keeps track of exactly which package versions you have installed, and creates an isolated environment so nothing clashes with the rest of your system.

If you don't have it yet:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then restart your terminal and check it works:

```bash
poetry --version
```

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/phospheneai/audio-transforms.git
cd authtransforms
```

### Step 2 — Install dependencies

This command reads `pyproject.toml` and installs everything into an isolated virtual environment automatically. You don't need to create one yourself.

```bash
poetry install
```

Want JupyterLab too? Add the `notebook` group:

```bash
poetry install --with notebook
```

### Step 3 — Activate the environment

```bash
poetry shell
```

You're now inside the virtual environment. Your terminal prompt will change. From here you can run Python normally — all the installed packages are available.

To leave the environment later, just type `exit`.

### Step 4 — Open JupyterLab

```bash
jupyter lab
```

The `notebooks/demo.ipynb` notebook is the best place to start. Open it and select the **Python (authtransforms)** kernel from the kernel picker in the top-right corner.

> If the kernel doesn't appear, run this once and then refresh JupyterLab:
> ```bash
> python -m ipykernel install --user --name authtransforms --display-name "Python (authtransforms)"
> ```

---

### Project structure

```
authtransforms/
├── src/
│   └── authtransforms/
│       ├── __init__.py     — public API (imports everything)
│       ├── transforms.py   — all the audio transforms
│       ├── pipeline.py     — Compose, OneOf, SomeOf, …
│       └── utils.py        — plotting and playback helpers
├── notebooks/
│   └── demo.ipynb          — interactive demo (start here)
├── demo.py                 — same demo as a plain Python script
├── pyproject.toml          — project config and dependency list
├── .gitignore
└── README.md
```

All the source code lives in `src/authtransforms/`. If you want to add a new transform, add it to `transforms.py` and export it from `__init__.py`.

---

### Troubleshooting

#### `libsox.dylib` not found (macOS)

Some torchaudio transforms use SoX under the hood. If you see an error about `libsox.dylib`, install it via Homebrew and symlink it into the base miniconda lib directory (that's where torchaudio's rpath looks, regardless of which virtualenv is active):

```bash
brew install sox
ln -sf /opt/homebrew/lib/libsox.dylib ~/miniconda3/lib/libsox.dylib
```

This is a one-time fix per machine.

#### `ModuleNotFoundError: No module named 'authtransforms'`

The package wasn't installed into the environment yet. Run:

```bash
poetry install
```

#### Audio file not found from the notebook

The notebook runs from the `notebooks/` folder, so relative paths like `./audio.wav` look inside `notebooks/`. Use `Path.cwd().parent` to get the project root:

```python
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent
audio_path = PROJECT_ROOT / "sample-audio-multilingual/english/real/my_file.wav"
```

---

### With pip (no Poetry)

If you just want to use the package without Poetry:

```bash
pip install torch torchaudio matplotlib numpy soundfile ipython jupyterlab
pip install -e .   # install authtransforms itself in editable mode
```

---

## Quick Start

```python
import torchaudio
from authtransforms.pipeline import Compose, OneOf
from authtransforms.transforms import (
    ToMono, RandomClip, RandomSpeedChange,
    RandomPitchShift, AddGaussianNoise, Normalize, RandomApply,
)
from authtransforms.utils import compare_audio, compare_play
import matplotlib.pyplot as plt

audio, sr = torchaudio.load("speech.wav")

pipeline = Compose([
    ToMono(),
    RandomClip(sample_rate=sr, clip_length=sr * 5),
    OneOf([
        RandomSpeedChange(sample_rate=sr),
        RandomPitchShift(sample_rate=sr, semitones=(-2, 2)),
    ]),
    RandomApply(AddGaussianNoise(), p=0.5),
    Normalize(target_db=-3.0),
])

augmented = pipeline(audio)

compare_audio(audio, augmented, sr)
plt.show()

compare_play(audio, augmented, sr)
```

---

## Transforms

All transforms are callable objects that accept a `(channels, samples)` `torch.Tensor` and return a transformed tensor of the same shape (unless noted).

### `RandomClip(sample_rate, clip_length, vad=True, vad_trigger_level=7.0)`

Extract a random fixed-length segment from the audio. Optionally applies torchaudio's Voice Activity Detector to trim leading/trailing silence after clipping.

```python
clip = RandomClip(sample_rate=16000, clip_length=16000 * 4)  # 4-second clip
```

### `RandomSpeedChange(sample_rate, speed_factors=(0.9, 1.0, 1.1))`

Randomly perturb playback speed via SoX, then resample back to the original rate. Using 0.9× and 1.1× alongside the original has been shown to significantly improve speech recognition models.

```python
speed = RandomSpeedChange(sample_rate=16000)
```

### `RandomBackgroundNoise(sample_rate, noise_dir, min_snr_db=0, max_snr_db=15)`

Mix in a randomly chosen `.wav` file from `noise_dir` (searched recursively) at a random signal-to-noise ratio. Loops or crops the noise to match the audio length.

```python
noise = RandomBackgroundNoise(sample_rate=16000, noise_dir="./musan/noise")
```

> Recommended noise source: [MUSAN](http://www.openslr.org/17/) — an 11 GB collection of music, speech, and noise recordings.

### `RandomPitchShift(sample_rate, semitones=(-2, 2))`

Shift pitch by a random number of semitones within the given range, via SoX.

```python
pitch = RandomPitchShift(sample_rate=16000, semitones=(-3, 3))
```

### `RandomGain(min_gain_db=-6, max_gain_db=6)`

Scale amplitude by a random gain factor in dB.

```python
gain = RandomGain(min_gain_db=-6, max_gain_db=6)
```

### `AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015)`

Add Gaussian white noise at a random standard deviation.

```python
noise = AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.01)
```

### `TimeShift(max_shift=0.2, roll=False)`

Shift the waveform in time by up to `max_shift` fraction of the total length. Wraps circularly if `roll=True`, otherwise zero-pads.

```python
shift = TimeShift(max_shift=0.15)
```

### `SpecAugment(sample_rate, n_fft, n_mels, freq_mask_param, time_mask_param, n_freq_masks, n_time_masks)`

Apply SpecAugment — frequency and time masking directly on a mel-spectrogram. Returns a spectrogram tensor rather than a waveform; use as the final step when your model consumes spectrograms.

```python
spec_aug = SpecAugment(sample_rate=16000, freq_mask_param=27, time_mask_param=100)
```

### `RandomApply(transform, p=0.5)`

Apply any transform with probability `p`. Mirrors `torchvision.transforms.RandomApply`.

```python
maybe_noise = RandomApply(AddGaussianNoise(), p=0.3)
```

### `Normalize(target_db=-3.0)`

Peak-normalize the audio to a target dBFS level.

```python
norm = Normalize(target_db=-3.0)
```

### `ToMono()`

Average all channels to a single mono channel.

---

## Pipeline Combinators

### `Compose(transforms)`

Apply a list of transforms sequentially — the core building block, identical in spirit to `torchvision.transforms.Compose`.

```python
pipeline = Compose([
    ToMono(),
    RandomClip(sr, sr * 5),
    RandomSpeedChange(sr),
    Normalize(),
])
augmented = pipeline(audio)
```

### `OneOf(transforms, weights=None)`

Pick exactly one transform at random each call. Optionally provide `weights` for non-uniform sampling.

```python
augment = OneOf([
    RandomSpeedChange(sr),
    RandomPitchShift(sr),
    TimeShift(),
])
```

### `SomeOf(transforms, n=2, shuffle=True)`

Pick `n` transforms at random (without replacement) and apply them.

```python
augment = SomeOf([
    RandomGain(),
    AddGaussianNoise(),
    TimeShift(),
    RandomPitchShift(sr),
], n=2)
```

### `RandomOrder(transforms)`

Apply all transforms but in a random order each call.

```python
augment = RandomOrder([
    RandomGain(),
    AddGaussianNoise(),
    TimeShift(),
])
```

### `Lambda(func, name="Lambda")`

Wrap any callable as a named transform.

```python
double = Lambda(lambda x: x * 2, name="Double")
```

---

## Utilities

### Plotting

All plotting functions return a `matplotlib.Figure` and work in notebooks and scripts alike.

```python
from authtransforms.utils import plot_waveform, plot_spectrogram, plot_audio, compare_audio
import matplotlib.pyplot as plt

# Single panel
plot_waveform(audio, sr, title="Waveform")
plot_spectrogram(audio, sr, title="Mel-Spectrogram")

# Stacked waveform + spectrogram
plot_audio(audio, sr, title="Original")

# 2×2 side-by-side comparison
compare_audio(audio, augmented, sr, title_before="Original", title_after="Augmented")
plt.show()
```


### Playback

```python
from authtransforms.utils import play_audio, compare_play

# Play a single clip
play_audio(audio, sr, label="Original")

# Play before and after back-to-back
compare_play(audio, augmented, sr)
```

In a **Jupyter notebook** this renders an interactive HTML5 audio widget.
In a **plain script** it writes a temporary WAV and opens it with the OS default player (`afplay` on macOS, `aplay` on Linux, `winsound` on Windows).

### Info

```python
from authtransforms.utils import audio_info

audio_info(audio, sr, label="Original")
# [Original]
#   Shape      : (1, 128000)
#   Channels   : 1
#   Sample rate: 16000 Hz
#   Duration   : 8.000 s  (128000 samples)
#   Peak       : 0.7231
#   RMS        : 0.0842  (-21.5 dBFS)
#   dtype      : torch.float32
```

---

## Demo

```bash
 check notebooks/demo.ipynb to get to know how to use 
```

The demo runs the full pipeline, prints audio info before and after, displays a `compare_audio` figure, and plays both clips.

---

## References

- [Simple Audio Augmentation with PyTorch — Jonathan Boigne](https://jonathanbgn.com/2021/08/30/audio-augmentation.html)
- [torchaudio documentation](https://pytorch.org/audio/)
- [SpecAugment: A Simple Data Augmentation Method for ASR](https://arxiv.org/abs/1904.08779)
- [MUSAN noise dataset](http://www.openslr.org/17/)
- [audiomentations](https://github.com/iver56/audiomentations)
- [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
