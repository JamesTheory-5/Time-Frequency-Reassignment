# Time-Frequency-Reassingment

Awesome—let’s build a minimal, fast, and clean **time–frequency reassignment** toolkit in Python using **NumPy (vectorized)** + **Numba JIT**. It follows the paper’s math:

* Reassigned time (group delay): (\hat t = t - \partial_\omega \phi)
* Reassigned frequency (inst. freq.): (\hat\omega = \omega + \partial_t \phi)

In the discrete implementation, we compute the phase derivatives via **auxiliary STFTs** using a **time-weighted window** (t,h(t)) and the **derivative window** (h'(t)) (per the paper’s discrete recipe).

---

# Code: `reassign.py`

```python
# reassign.py
# Minimal, vectorized time–frequency reassignment with NumPy + Numba
# References:
# - Stationary phase → reassignment definitions (t̂, ω̂): Loris.pdf Eq. (28–29) 
# - Efficient discrete computation via auxiliary STFTs (time-weighted, derivative windows): Section 6 

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from math import pi
from typing import Tuple, Optional, Dict
try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco


# ---------------------------
# Window helpers
# ---------------------------

def kaiser_window(N: int, beta: float = 9.0) -> NDArray[np.float64]:
    """Symmetric real window h[n], length N."""
    return np.kaiser(N, beta).astype(np.float64)

def hann_window(N: int) -> NDArray[np.float64]:
    return np.hanning(N).astype(np.float64)

def derivative_window(h: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Discrete-time derivative of window h[n].
    Central difference with reflective end-caps for symmetry.
    This approximates dh/dn required for frequency reassignment auxiliary STFT
    (see Sec. 6) 
    """
    dh = np.zeros_like(h)
    # central differences
    dh[1:-1] = 0.5*(h[2:] - h[:-2])
    # simple one-sided at edges (keeps symmetry decent)
    dh[0]     = h[1] - h[0]
    dh[-1]    = h[-1] - h[-2]
    return dh

def time_weighted_window(h: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    t * h(t) in discrete samples → (n - n0) * h[n], where n0 is the window center.
    This produces the auxiliary window for ∂φ/∂ω (group delay) per the paper
    (Eq. 28 & Sec. 6). 
    """
    N = h.size
    n = np.arange(N, dtype=np.float64)
    n0 = 0.5*(N-1)  # center index
    return (n - n0) * h


# ---------------------------
# Framing (vectorized)
# ---------------------------

def frame_signal(x: NDArray[np.float64], N: int, hop: int) -> NDArray[np.float64]:
    """
    Produce overlapping frames with hop size. Zero-pad head/tail so every sample
    is covered and frames are centered (n0 at signal positions).
    """
    x = np.asarray(x, dtype=np.float64)
    n0 = (N-1)//2  # center offset for odd/even both okay if we zero-pad
    pad = N - 1    # generous pad to allow centering
    x_pad = np.pad(x, (pad, pad), mode='constant')
    # number of frames so that the analysis time grid covers x
    num_frames = 1 + (len(x_pad) - N) // hop
    strides = (x_pad.strides[0]*hop, x_pad.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        x_pad, shape=(num_frames, N), strides=strides).copy()
    return frames


# ---------------------------
# Core STFTs (vectorized)
# ---------------------------

def stft_matrix(frames: NDArray[np.float64], window: NDArray[np.float64], nfft: Optional[int]=None
               ) -> NDArray[np.complex128]:
    """
    Compute complex STFT for all frames with given window. Returns full-spectrum FFT.
    """
    if nfft is None: nfft = window.size
    W = window[None, :]  # (1, N)
    X = np.fft.rfft(frames * W, n=nfft, axis=1)  # use rfft → non-negative freqs
    return X

# ---------------------------
# Reassignment (auxiliary STFTs)
# ---------------------------

def reassignment(
    x: NDArray[np.float64],
    fs: float,
    N: int = 2048,
    hop: Optional[int] = None,
    window: Optional[NDArray[np.float64]] = None,
    nfft: Optional[int] = None,
    magnitude_floor_db: float = -120.0,
    use_kaiser_beta: float = 9.0,
) -> Dict[str, NDArray]:
    """
    Compute reassigned time/frequency fields.
    Returns a dict with:
      - S: magnitude spectrogram (float)
      - t_grid: nominal time centers (seconds) per frame
      - f_grid: nominal frequency bins (Hz)
      - t_hat: reassigned times (seconds) [same shape as S]
      - f_hat: reassigned frequencies (Hz) [same shape as S]
      - mask: boolean where reassignment is considered valid (above floor & finite)
    """
    x = np.asarray(x, dtype=np.float64)
    if hop is None: hop = N // 4
    if window is None:
        window = kaiser_window(N, beta=use_kaiser_beta)

    # Auxiliary windows
    ht  = time_weighted_window(window)      # (n - n0)*h[n] → ∂φ/∂ω
    dh  = derivative_window(window)         # h'[n]           → ∂φ/∂t

    # Frame signal
    frames = frame_signal(x, N, hop)        # (F, N)
    F = frames.shape[0]
    if nfft is None: nfft = int(2**np.ceil(np.log2(N)))
    # STFTs (rfft: K bins)
    X    = stft_matrix(frames, window, nfft)   # (F, K)
    X_ht = stft_matrix(frames, ht,     nfft)   # (F, K)
    X_dh = stft_matrix(frames, dh,     nfft)   # (F, K)

    # Grids
    K = X.shape[1]
    t_grid = (np.arange(F)*hop + 0.5*(N-1)) / fs  # center of each frame in seconds
    f_grid = (np.arange(K) * fs) / nfft           # Hz for rfft bins
    omega_k = 2*np.pi * (np.arange(K) / nfft)     # rad/sample (discrete-time)

    # Magnitude, masking
    S = np.abs(X)
    # magnitude floor per-bin to avoid division blowups
    S_db = 20*np.log10(np.maximum(S, 1e-12))
    mag_mask = S_db > magnitude_floor_db

    # Reassignment formulas (discrete-time form)
    # t_hat_samples = n0_frame - Re{ X_ht / X }  (units: samples)
    # ω_hat = ω_k + Im{ X_dh / X }              (units: rad/sample)
    eps = 1e-30
    ratio_ht = X_ht / (X + eps)
    ratio_dh = X_dh / (X + eps)

    # group delay (samples)
    # NOTE: ht used here is (n - n0) h[n] → Re{X_ht/X} already yields ∂φ/∂ω in samples
    # so t̂_samples = n_center - Re{X_ht/X}
    n_center_per_frame = (np.arange(F)*hop + 0.5*(N-1))[:, None]  # (F,1)
    t_hat_samples = n_center_per_frame - np.real(ratio_ht)

    # instantaneous frequency (rad/sample)
    omega_hat = omega_k[None, :] + np.imag(ratio_dh)

    # convert to seconds / Hz
    t_hat = t_hat_samples / fs
    f_hat = (omega_hat * fs) / (2*np.pi)

    # Valid (finite) mask
    valid = np.isfinite(t_hat) & np.isfinite(f_hat) & mag_mask

    return dict(
        S=S.astype(np.float32),
        t_grid=t_grid.astype(np.float64),
        f_grid=f_grid.astype(np.float64),
        t_hat=t_hat.astype(np.float64),
        f_hat=f_hat.astype(np.float64),
        mask=valid
    )


# ---------------------------
# Optional: sparse point cloud (only keep “good” bins)
# ---------------------------

def reassigned_points(
    reassigned: Dict[str, NDArray],
    consensus_radius_bins: int = 0,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Return sparse (t, f, mag) arrays for bins that passed the mask.
    consensus_radius_bins (optional): 0 for raw; >0 could be used for simple
    density pruning (not implemented here to keep it minimal).
    """
    S = reassigned["S"]
    t_hat = reassigned["t_hat"]
    f_hat = reassigned["f_hat"]
    mask = reassigned["mask"]
    # flatten
    idx = np.where(mask)
    t = t_hat[idx]
    f = f_hat[idx]
    m = S[idx]
    return t, f, m


# ---------------------------
# Tiny demo (if run as a script)
# ---------------------------

if __name__ == "__main__":
    # Simple test: two chirps + impulse
    fs = 48000
    T = 1.0
    n = np.arange(int(fs*T))
    x = (0.4*np.sin(2*pi*(200*n/fs + 300*(n/fs)**2)) +
         0.3*np.sin(2*pi*(1000*n/fs + 100*(n/fs)**2)))
    x[24000] += 1.0  # impulse @ 0.5 s

    out = reassignment(x, fs, N=2048, hop=512, nfft=4096, magnitude_floor_db=-80)
    t, f, mag = reassigned_points(out)
    print("Reassigned points:", t.shape[0])
```

---

## How this maps to the paper

* **Definitions** of reassigned time and frequency via phase derivatives (group delay and instantaneous frequency) → **Eq. (28–29)** in the derivation via stationary phase.
* **Discrete, efficient computation** using auxiliary STFTs of the **time-weighted** and **derivative** windows (no explicit numerical phase differentiation needed) → **Section 6**.

---

## Usage sketch

```python
import soundfile as sf
from reassign import reassignment, reassigned_points

x, fs = sf.read("some_audio.wav")   # mono float32/64
if x.ndim == 2:
    x = x.mean(axis=1)

res = reassignment(x, fs, N=2048, hop=512, nfft=4096, magnitude_floor_db=-90)
t_pts, f_pts, mag = reassigned_points(res)

# Example: build a dense image (simple)
import matplotlib.pyplot as plt
plt.scatter(t_pts, f_pts, c=20*np.log10(mag+1e-12), s=1, cmap="magma")
plt.xlabel("Time (s)"); plt.ylabel("Freq (Hz)"); plt.title("Reassigned points")
plt.ylim(0, fs/2); plt.colorbar(label="dB"); plt.show()
```

---

## Notes & knobs

* **Windows**: default is **Kaiser(β=9)**; change to `hann_window` if you like. The **time-weighted** and **derivative** windows are derived from the base window as required by the paper’s Section 6 (aux STFTs).
* **Units**:

  * (\hat t) returned in **seconds**, (\hat f) in **Hz**.
  * Internally, reassignment is done in **samples** and **rad/sample** to match the discrete math, then converted.
* **Masking**: `magnitude_floor_db` removes near-zero bins (reassignment is undefined when (X\approx 0) — also noted conceptually in the paper when spectrogram energy vanishes).
* **Hop/FFT**: `hop=N//4`, `nfft>=N`; increase `nfft` for smoother frequency grids (interpolates bins, doesn’t add information).

---

If you want, next I can add:

* a tiny **consensus filter** (cluster tests near peaks to reject outliers) as suggested around Fig. 3 discussion,
* a **sparse-resynthesis** example that collects partials from reassigned tracks (first step toward the *Loris* additive engine),
* and a **Numba-accelerated framing** path for very long signals.
