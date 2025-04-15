import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import julius
import numpy as np


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)

def mel_to_hz(m):
    return 700 * (10**(m / 2595) - 1)

def mel_frequencies(n_mels, fmin, fmax):
    low = hz_to_mel(fmin)
    high = hz_to_mel(fmax)
    mels = np.linspace(low, high, n_mels)
    return mel_to_hz(mels)

class LowPassFilters(torch.nn.Module):
    """
    Bank of low pass filters.

    Args:
        cutoffs (list[float]): list of cutoff frequencies, in [0, 1] expressed as `f/f_s` where
            f_s is the samplerate.
        width (int): width of the filters (i.e. kernel_size=2 * width + 1).
            Default to `2 / min(cutoffs)`. Longer filters will have better attenuation
            but more side effects.
    Shape:
        - Input: `(*, T)`
        - Output: `(F, *, T` with `F` the len of `cutoffs`.
    """

    def __init__(self, cutoffs: list, width: int = None):
        super().__init__()
        self.cutoffs = cutoffs
        if width is None:
            width = int(2 / min(cutoffs))
        self.width = width
        window = torch.hamming_window(2 * width + 1, periodic=False)
        t = np.arange(-width, width + 1, dtype=np.float32)
        filters = []
        for cutoff in cutoffs:
            sinc = torch.from_numpy(np.sinc(2 * cutoff * t))
            filters.append(2 * cutoff * sinc * window)
        self.register_buffer("filters", torch.stack(filters).unsqueeze(1).float())

    def forward(self, input):
        *others, t = input.shape
        input = input.view(-1, 1, t)
        out = F.conv1d(input, self.filters, padding=self.width)
        return out.permute(1, 0, 2).reshape(-1, *others, t)

    def __repr__(self):
        return "LossPassFilters(width={},cutoffs={})".format(self.width, self.cutoffs)

class Remix(nn.Module):
    """Mixes the noise (first source) across the batch, leaving clean intact."""
    def forward(self, sources):
        noise, clean = sources
        bs = noise.shape[0]
        device = noise.device
        perm = torch.argsort(torch.rand(bs, device=device), dim=0)
        return torch.stack([noise[perm], clean])

class RevEcho(nn.Module):
    def __init__(self, proba=0.5, initial=0.3, rt60=(0.3, 1.3), first_delay=(0.01, 0.03),
                 repeat=3, jitter=0.1, keep_clean=0.1, sample_rate=16000):
        super().__init__()
        self.proba = proba
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.sample_rate = sample_rate

    def _reverb(self, source, initial, first_delay, rt60):
        length = source.shape[-1]
        reverb = torch.zeros_like(source)
        for _ in range(self.repeat):
            frac = 1.0
            echo = initial * source
            while frac > 1e-3:
                # Jitter the delay slightly.
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                delay = min(1 + int(jitter * first_delay * self.sample_rate), length)
                echo = F.pad(echo[..., :-delay], (delay, 0))
                reverb = reverb + echo
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                attenuation = 10 ** (-3 * jitter * first_delay / rt60)
                echo = echo * attenuation
                frac *= attenuation
        return reverb

    def forward(self, wav):
        if random.random() >= self.proba:
            return wav
        noise, clean = wav
        initial = random.random() * self.initial
        first_delay = random.uniform(*self.first_delay)
        rt60 = random.uniform(*self.rt60)
        reverb_noise = self._reverb(noise, initial, first_delay, rt60)
        noise = noise + reverb_noise
        reverb_clean = self._reverb(clean, initial, first_delay, rt60)
        clean = clean + self.keep_clean * reverb_clean
        noise = noise + (1 - self.keep_clean) * reverb_clean
        return torch.stack([noise, clean])

class BandMask(nn.Module):
    def __init__(self, maxwidth=0.2, bands=120, sample_rate=16000):
        super().__init__()
        self.maxwidth = maxwidth
        self.bands = bands
        self.sample_rate = sample_rate

    def forward(self, wav):
        # Use your dsp functions to compute mel frequencies and design a band-stop filter.
        # Here we assume dsp.mel_frequencies and dsp.LowPassFilters are implemented.
        bandwidth = int(abs(self.maxwidth) * self.bands)
        mels = mel_frequencies(self.bands, 40, self.sample_rate / 2) / self.sample_rate
        low_idx = random.randrange(self.bands)
        high_idx = random.randrange(low_idx, min(self.bands, low_idx + bandwidth))
        filters = LowPassFilters([mels[low_idx], mels[high_idx]]).to(wav.device)
        low, midlow = filters(wav)
        return wav - midlow + low

class Shift(nn.Module):
    def __init__(self, shift=8192, same=True):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        # wav: shape (sources, batch, channels, length)
        sources, batch, channels, length = wav.shape
        if self.shift > 0:
            new_length = length - self.shift
            if not self.training:
                wav = wav[..., :new_length]
            else:
                if self.same:
                    offsets = torch.randint(0, self.shift, (1, batch, 1, 1), device=wav.device)
                    offsets = offsets.expand(sources, -1, channels, -1)
                else:
                    offsets = torch.randint(0, self.shift, (sources, batch, 1, 1), device=wav.device)
                indexes = torch.arange(new_length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav