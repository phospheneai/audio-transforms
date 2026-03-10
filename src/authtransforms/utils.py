"""
Audio visualization and playback utilities.

Works in Jupyter notebooks (IPython display) and plain scripts (matplotlib).

Example
-------
>>> from utils import plot_audio, play_audio, compare_audio
>>>
>>> audio, sr = torchaudio.load('speech.wav')
>>> augmented = pipeline(audio)
>>>
>>> compare_audio(audio, augmented, sr, title_before='Original', title_after='Augmented')
>>> play_audio(audio, sr, label='Original')
>>> play_audio(augmented, sr, label='Augmented')
"""

import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(audio: torch.Tensor) -> np.ndarray:
    """Return a 1-D numpy waveform (mono mix if multi-channel)."""
    wav = audio.detach().cpu()
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    return wav.squeeze().numpy()


def _in_notebook() -> bool:
    """Detect whether we're running inside a Jupyter / IPython kernel."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _tensor_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a tensor to WAV bytes for IPython playback."""
    buf = io.BytesIO()
    torchaudio.save(buf, audio.cpu(), sample_rate, format="wav")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_waveform(
    audio: torch.Tensor,
    sample_rate: int,
    title: str = "Waveform",
    ax: Optional[plt.Axes] = None,
    color: str = "#1f77b4",
) -> plt.Axes:
    """Plot the raw waveform of an audio tensor.

    Args:
        audio: Audio tensor of shape (channels, samples) or (samples,).
        sample_rate: Sample rate in Hz.
        title: Plot title.
        ax: Optional existing matplotlib Axes to draw on.
        color: Line color.

    Returns:
        The Axes object.
    """
    wav = _to_numpy(audio)
    times = np.arange(len(wav)) / sample_rate

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))

    ax.plot(times, wav, color=color, linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(times[0], times[-1])
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.grid(True, alpha=0.3)
    return ax


def plot_spectrogram(
    audio: torch.Tensor,
    sample_rate: int,
    title: str = "Spectrogram",
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: Optional[int] = 80,
    ax: Optional[plt.Axes] = None,
    cmap: str = "inferno",
) -> plt.Axes:
    """Plot a mel-spectrogram (or linear spectrogram) of an audio tensor.

    Args:
        audio: Audio tensor of shape (channels, samples) or (samples,).
        sample_rate: Sample rate in Hz.
        title: Plot title.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        n_mels: Number of mel filterbanks. Set to None for a linear spectrogram.
        ax: Optional existing matplotlib Axes.
        cmap: Matplotlib colormap.

    Returns:
        The Axes object.
    """
    if n_mels:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    else:
        transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)

    mono = audio.mean(dim=0, keepdim=True) if audio.shape[0] > 1 else audio
    spec = transform(mono).squeeze(0)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec).numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    duration = audio.shape[-1] / sample_rate
    img = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, sample_rate / 2 / 1000],
        cmap=cmap,
    )
    plt.colorbar(img, ax=ax, format="%+2.0f dB", label="dB")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    return ax


def plot_audio(
    audio: torch.Tensor,
    sample_rate: int,
    title: str = "Audio",
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: Optional[int] = 80,
) -> plt.Figure:
    """Plot waveform + spectrogram stacked vertically.

    Args:
        audio: Audio tensor of shape (channels, samples) or (samples,).
        sample_rate: Sample rate in Hz.
        title: Figure suptitle.
        n_fft: FFT window size for spectrogram.
        hop_length: STFT hop length.
        n_mels: Mel filterbanks (None for linear).

    Returns:
        The Figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), constrained_layout=True)
    plot_waveform(audio, sample_rate, title="Waveform", ax=axes[0])
    plot_spectrogram(audio, sample_rate, title="Mel-Spectrogram", ax=axes[1],
                     n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def compare_audio(
    before: torch.Tensor,
    after: torch.Tensor,
    sample_rate: int,
    title_before: str = "Before",
    title_after: str = "After",
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: Optional[int] = 80,
) -> plt.Figure:
    """Side-by-side (before / after) waveform + spectrogram comparison.

    Plots a 2×2 grid:
      - Row 1: Waveforms (before | after)
      - Row 2: Spectrograms (before | after)

    Args:
        before: Original audio tensor.
        after: Augmented audio tensor.
        sample_rate: Sample rate in Hz.
        title_before: Column title for the original.
        title_after: Column title for the augmented.
        n_fft: FFT size.
        hop_length: STFT hop length.
        n_mels: Mel filterbanks (None for linear).

    Returns:
        The Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 6), constrained_layout=True)

    plot_waveform(before, sample_rate, title=f"{title_before} — Waveform", ax=axes[0, 0], color="#1f77b4")
    plot_waveform(after, sample_rate, title=f"{title_after} — Waveform", ax=axes[0, 1], color="#d62728")
    plot_spectrogram(before, sample_rate, title=f"{title_before} — Spectrogram",
                     ax=axes[1, 0], n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    plot_spectrogram(after, sample_rate, title=f"{title_after} — Spectrogram",
                     ax=axes[1, 1], n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    fig.suptitle("Augmentation Comparison", fontsize=14, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------


def play_audio(
    audio: torch.Tensor,
    sample_rate: int,
    label: str = "Audio",
    autoplay: bool = False,
) -> None:
    """Play audio inline in a Jupyter notebook, or save a temp file and open it.

    In a Jupyter environment this renders an interactive HTML audio widget.
    In a plain Python script it writes a temporary WAV file and opens the
    system's default media player.

    Args:
        audio: Audio tensor of shape (channels, samples) or (samples,).
        sample_rate: Sample rate in Hz.
        label: Human-readable label shown above the player (notebook only).
        autoplay: If True, start playback automatically (notebook only).
    """
    if _in_notebook():
        _play_notebook(audio, sample_rate, label=label, autoplay=autoplay)
    else:
        _play_script(audio, sample_rate)


def _play_notebook(
    audio: torch.Tensor,
    sample_rate: int,
    label: str = "Audio",
    autoplay: bool = False,
) -> None:
    """Render an HTML5 audio widget in a Jupyter notebook cell."""
    try:
        from IPython.display import Audio, display, HTML
    except ImportError:
        raise ImportError("IPython is required for notebook playback: pip install ipython")

    wav_bytes = _tensor_to_wav_bytes(audio, sample_rate)
    display(HTML(f"<b>{label}</b>"))
    display(Audio(data=wav_bytes, rate=sample_rate, autoplay=autoplay))


def _play_script(audio: torch.Tensor, sample_rate: int) -> None:
    """Write a temp WAV and open it with the OS default player."""
    import platform
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    torchaudio.save(tmp_path, audio.cpu(), sample_rate)
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["afplay", tmp_path])
    elif system == "Linux":
        subprocess.Popen(["aplay", tmp_path])
    elif system == "Windows":
        import winsound
        winsound.PlaySound(tmp_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        print(f"[play_audio] Saved to {tmp_path} — open it manually.")


def compare_play(
    before: torch.Tensor,
    after: torch.Tensor,
    sample_rate: int,
    label_before: str = "Before",
    label_after: str = "After",
) -> None:
    """Play both before and after audio widgets (notebook) or sequentially (script).

    Args:
        before: Original audio tensor.
        after: Augmented audio tensor.
        sample_rate: Sample rate.
        label_before: Label for the original.
        label_after: Label for the augmented.
    """
    play_audio(before, sample_rate, label=label_before)
    play_audio(after, sample_rate, label=label_after)


# ---------------------------------------------------------------------------
# Quick info helper
# ---------------------------------------------------------------------------


def audio_info(audio: torch.Tensor, sample_rate: int, label: str = "Audio") -> None:
    """Print a summary of an audio tensor's properties.

    Args:
        audio: Audio tensor.
        sample_rate: Sample rate in Hz.
        label: Human-readable name shown in the output.
    """
    channels = audio.shape[0] if audio.dim() == 2 else 1
    samples = audio.shape[-1]
    duration = samples / sample_rate
    peak = audio.abs().max().item()
    rms = audio.pow(2).mean().sqrt().item()
    rms_db = 20 * np.log10(rms + 1e-9)

    print(f"[{label}]")
    print(f"  Shape      : {tuple(audio.shape)}")
    print(f"  Channels   : {channels}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration   : {duration:.3f} s  ({samples} samples)")
    print(f"  Peak       : {peak:.4f}")
    print(f"  RMS        : {rms:.4f}  ({rms_db:.1f} dBFS)")
    print(f"  dtype      : {audio.dtype}")
