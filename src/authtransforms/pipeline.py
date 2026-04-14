"""
Audio augmentation pipeline — a torchvision-style Compose for audio tensors.

Example
-------
>>> from transforms import RandomClip, RandomSpeedChange, AddGaussianNoise
>>> from pipeline import Compose, OneOf
>>>
>>> pipeline = Compose([
...     RandomClip(sample_rate=16000, clip_length=16000 * 4),
...     OneOf([
...         RandomSpeedChange(sample_rate=16000),
...         AddGaussianNoise(),
...     ]),
... ])
>>> augmented = pipeline(audio_tensor)
"""

import random
from typing import Callable, List, Optional, Sequence
import torch


class Compose:
    """Apply a sequence of transforms in order — mirrors torchvision.transforms.Compose.

    Args:
        transforms: List of callables that accept and return a Tensor.

    Example::

        pipeline = Compose([
            RandomClip(sample_rate, clip_length=64000),
            RandomSpeedChange(sample_rate),
            RandomBackgroundNoise(sample_rate, './noises'),
        ])
        augmented = pipeline(audio)
    """

    def __init__(self, transforms: List[Callable]): 
        
        # The class where all the transforms are applied to the audio

        # The List[Callable] indicates that the transforms variable should have a list of callable funcations
        self.transforms = transforms

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append(")")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx):
        return self.transforms[idx]


class OneOf:
    """Apply exactly one randomly-chosen transform from the list.

    Args:
        transforms: List of transforms to choose from.
        weights: Optional probability weights (unnormalized). If None, uniform.

    Example::

        augment = OneOf([
            RandomSpeedChange(sample_rate),
            AddGaussianNoise(),
            RandomPitchShift(sample_rate),
        ])
    """

    def __init__(self, transforms: List[Callable], weights: Optional[List[float]] = None): #The weights of randomly chosen transforms are stored
        self.transforms = transforms
        self.weights = weights

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self.weights:
            t = random.choices(self.transforms, weights=self.weights, k=1)[0] #if there are weights
        else:
            t = random.choice(self.transforms)  #if there's no weight
        return t(audio)

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append(")")
        return "\n".join(lines)


class SomeOf:
    """Apply a random subset of transforms (n chosen without replacement).

    Args:
        transforms: Pool of available transforms.
        n: Number of transforms to apply each call.
        shuffle: If True, apply the selected transforms in a random order.

    Example::

        augment = SomeOf([
            RandomGain(),
            AddGaussianNoise(),
            TimeShift(),
            RandomPitchShift(sample_rate),
        ], n=2)
    """

    # chooses n random transforms from a list and applies them sequentially to an audio tensor.

    def __init__(self, transforms: List[Callable], n: int = 2, shuffle: bool = True):
        if n > len(transforms):
            raise ValueError(f"n={n} exceeds number of transforms ({len(transforms)})")
        self.transforms = transforms
        self.n = n
        self.shuffle = shuffle
        
    # Makes the object callable like a function
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        selected = random.sample(self.transforms, self.n)  #selects n random transforms
        if self.shuffle:
            random.shuffle(selected)
        for t in selected:
            audio = t(audio)
        return audio

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(n={self.n},"]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append(")")
        return "\n".join(lines)


class RandomOrder:
    """Apply all transforms in a random order each call.

    Args:
        transforms: List of transforms.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        order = list(self.transforms)   # The transforms is copied in the order 
        random.shuffle(order)           # Then shuffled and applied to the audio
        for t in order:
            audio = t(audio)
        return audio

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append(")")
        return "\n".join(lines)


class Identity:
    """Pass-through transform — returns the audio unchanged.  Useful as a no-op placeholder."""

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:   #Identity is a transform that returns the input audio unchanged, mainly used as a placeholder or optional "do nothing" step in augmentation pipelines.
        return audio

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Lambda:
    """Wrap any callable as a named transform (mirrors torchvision.transforms.Lambda).

    Example::

        double = Lambda(lambda x: x * 2, name="Double")
    """
    # To add custom augmentation and convert it into a transform
    def __init__(self, func: Callable, name: str = "Lambda"):
        self.func = func
        self.name = name

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return self.func(audio)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
