"""Array-pure signal utilities.

Numeric helpers that operate on plain NumPy arrays sampled along an axis — no :class:`~pybvh.bvh.Bvh` involved. The centerpiece is :func:`finite_difference`, the one finite-difference convention in pybvh: the kinematics ladder of :mod:`pybvh.analysis` differentiates through it at the order it needs (``order=2`` for accelerations, 3 for jerk), and the geometry derivative kernels of :mod:`pybvh.geometry` through its first order. A higher order applies the stencil that many times and pads or crops once, at the end; the function's docstring says when that differs from calling it repeatedly. The rest are self-contained statistics, smoothing, spectrum, and simplification tools (no scipy).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import numpy.typing as npt


def finite_difference(
    arr: npt.NDArray[np.float64],
    dt: float,
    *,
    order: int = 1,
    stencil: str = "central",
    pad: str = "edge",
    axis: int = 0,
) -> npt.NDArray[np.float64]:
    """Differentiate a sampled array ``order`` times along one axis.

    The one finite-difference convention in pybvh. The kinematics ladder
    of :mod:`pybvh.analysis` (velocities, accelerations, the speed
    derivative, jerk and the ``root_trajectory`` velocities) and the
    geometry derivative kernels of :mod:`pybvh.geometry` (``curvature``,
    ``torsion``, ``movement_phase``) differentiate through it. A current
    limitation: the geometry kernels call it at order 1 once per
    derivative rather than with ``order``, so under
    ``stencil="forward"`` with ``pad="edge"`` their second and third
    derivatives carry the tail of repeated calls described next.

    A higher ``order`` applies the stencil ``order`` times and pads or
    crops **once**, at the end. The alternative, calling this function
    ``order`` times, pads after every pass. Where both are defined, the
    two agree for ``stencil="central"`` and for ``pad="none"``. They are
    not always both defined: under ``"central"`` with ``pad="none"``, a
    call can crop a short input to one sample, which the next call
    refuses, while ``order`` returns the empty result. With
    ``stencil="forward"`` and ``pad="edge"`` they differ at order 2 and
    above in the last ``order`` samples. There the one-pass form repeats
    the last valid ``order``-th difference, while repeated calls
    difference each padded copy against the sample it copies, which
    leaves the last two samples exactly 0, a derivative the signal does
    not have.

    Parameters
    ----------
    arr : ndarray
        Samples taken at a uniform step ``dt`` along ``axis``.
    dt : float
        Sample spacing (e.g. ``frame_time``). The result is in units of
        ``arr`` per ``dt**order``.
    order : int, keyword-only, optional
        How many times the stencil is applied: 1 (default) is the first
        derivative, 2 the second, and so on.
    stencil : {"central", "forward"}, optional
        ``"central"`` (default): ``np.gradient`` — second-order accurate
        interior, one-sided at the boundary.  ``"forward"``:
        ``(arr[i+1] - arr[i]) / dt``, first-order, causal.
    pad : {"edge", "none"}, optional
        ``"edge"`` (default): output keeps the input length along
        ``axis``; ``"central"`` gets its boundary samples from the
        one-sided differences of ``np.gradient``, and ``"forward"``
        repeats its last valid value over the last ``order`` samples.
        ``"none"``: drop the boundary samples the repeated stencil
        cannot define — central drops ``order`` at each end, forward
        drops the trailing ``order``. An input too short to keep one
        sample gives an empty result.
    axis : int, optional
        Axis to differentiate along (default 0, the frame axis).

    Returns
    -------
    ndarray
        The ``order``-th derivative. Same shape as ``arr`` when
        ``pad="edge"``; shorter along ``axis`` by ``2 * order``
        (central) or ``order`` (forward), down to empty, when
        ``pad="none"``.

    Raises
    ------
    TypeError
        If ``order`` is not an int. A bool is refused rather than read
        as 0 or 1.
    ValueError
        If ``stencil`` or ``pad`` is invalid, ``order`` is below 1, or
        ``arr`` is too short along ``axis`` to compute anything: fewer
        than 2 samples under ``"central"`` (``np.gradient`` needs two,
        at any order and either pad), or fewer than ``order + 1`` under
        ``"forward"`` with ``pad="edge"`` (no valid ``order``-th
        difference to repeat). A caller with a stricter notion of
        enough samples, such as the clip minimums of
        :mod:`pybvh.analysis`, checks it before calling.
    """
    if stencil not in ("central", "forward"):
        raise ValueError(f"stencil must be 'central' or 'forward', got {stencil!r}")
    if pad not in ("edge", "none"):
        raise ValueError(f"pad must be 'edge' or 'none', got {pad!r}")
    # bool is an int subclass; catch it before the int check.
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise TypeError(f"order must be an int, got {type(order).__name__} {order!r}")
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")
    # An unsigned NumPy integer would wrap in the -order crop below.
    order = int(order)

    arr = np.asarray(arr, dtype=np.float64)
    n_samples = arr.shape[axis]
    if stencil == "central" and n_samples < 2:
        raise ValueError(
            f"stencil='central' requires at least 2 samples along axis {axis} (have {n_samples})."
        )
    if stencil == "forward" and pad == "edge" and n_samples < order + 1:
        raise ValueError(
            f"stencil='forward', pad='edge' at order {order} requires at least "
            f"{order + 1} samples along axis {axis} (have {n_samples})."
        )

    out = arr
    if stencil == "central":
        for _ in range(order):
            out = np.gradient(out, dt, axis=axis)
        if pad == "edge":
            return out
        interior = [slice(None)] * arr.ndim
        interior[axis] = slice(order, -order)
        return out[tuple(interior)]

    # stencil == "forward"
    for _ in range(order):
        out = np.diff(out, axis=axis) / dt
    if pad == "none":
        return out
    last = np.take(out, [-1], axis=axis)
    tail = np.repeat(last, order, axis=axis)
    return np.concatenate([out, tail], axis=axis)


TemporalStats = namedtuple("TemporalStats", ["mean", "std", "min", "max", "skewness", "kurtosis"])


def temporal_stats(
    signal: npt.NDArray[np.float64],
    axis: int = 0,
) -> TemporalStats:
    """Summary statistics of a signal along an axis.

    Returns mean, std, min, max, and the third/fourth standardized moments
    (skewness and *excess* kurtosis), all reduced over ``axis``. Skew and
    kurtosis are computed by hand (no scipy). Where the std is ~0 (a
    constant signal) skewness and kurtosis are ``nan``.

    All estimators are the **population** (biased) forms: ``std`` uses
    ``ddof=0``, and the moments are plain ``1/N`` sums, giving
    ``g₁ = m₃/s³`` and ``g₂ = m₄/s⁴ − 3``. The bias-corrected **sample**
    forms — what ``pandas.Series.skew()`` / ``.kurt()`` / ``.std()``
    return, and what scipy gives with ``bias=False`` — carry extra
    ``N``-dependent factors and differ materially on short signals
    (converging as ``N`` grows). Kurtosis is *excess* (normal ⇒ 0), not
    raw (normal ⇒ 3). The population convention matches ``cov3dj`` and
    the rest of pybvh's descriptors, which treat a clip as the whole
    population rather than a sample of one.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    axis : int, optional
        Axis to reduce over (default 0).

    Returns
    -------
    TemporalStats
        Named tuple ``(mean, std, min, max, skewness, kurtosis)`` with
        ``axis`` removed.
    """
    signal = np.asarray(signal, dtype=np.float64)
    mean = signal.mean(axis=axis)
    std = signal.std(axis=axis)
    centered = signal - np.expand_dims(mean, axis)
    m3 = (centered**3).mean(axis=axis)
    m4 = (centered**4).mean(axis=axis)
    valid = std > 1e-12
    safe = np.where(valid, std, 1.0)
    skewness = np.where(valid, m3 / safe**3, np.nan)
    kurtosis = np.where(valid, m4 / safe**4 - 3.0, np.nan)
    return TemporalStats(
        mean, std, signal.min(axis=axis), signal.max(axis=axis), skewness, kurtosis
    )


def box_filter_smooth(
    signal: npt.NDArray[np.float64],
    window: int,
    axis: int = 0,
) -> npt.NDArray[np.float64]:
    """Moving-average smoothing with a box kernel of width ``window``.

    Edge samples are handled by edge-padding (the alternatives — reflect,
    zero-pad, or a shrinking window at the ends — bias the first and last
    samples differently) so the output keeps the input length. Fully
    vectorized via a cumulative-sum sliding window (no Python loop over
    the signal).

    An **even** ``window`` cannot be centered on a sample; this takes the
    extra sample from the future (``(window-1)//2`` back,
    ``window//2`` forward), so the output leads the input by half a
    sample. The mirror choice lags by the same amount. Use an odd window
    when that half-sample matters, or when matching another
    implementation's even-window output.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    window : int
        Box width in samples (``>= 1``). ``window == 1`` is a no-op.
    axis : int, optional
        Axis to smooth along (default 0).

    Returns
    -------
    ndarray
        The smoothed signal, same shape as ``signal``.

    Raises
    ------
    ValueError
        If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    signal = np.asarray(signal, dtype=np.float64)
    if window == 1:
        return signal.copy()

    pad_lo, pad_hi = (window - 1) // 2, window // 2
    pad_width = [(0, 0)] * signal.ndim
    pad_width[axis] = (pad_lo, pad_hi)
    padded = np.pad(signal, pad_width, mode="edge")

    cumsum = np.cumsum(padded, axis=axis)
    zero_shape = list(cumsum.shape)
    zero_shape[axis] = 1
    cumsum = np.concatenate([np.zeros(zero_shape), cumsum], axis=axis)
    n = signal.shape[axis]
    hi = np.take(cumsum, np.arange(window, window + n), axis=axis)
    lo = np.take(cumsum, np.arange(0, n), axis=axis)
    return (hi - lo) / window


def fft_magnitude(
    signal: npt.NDArray[np.float64],
    fs: float = 1.0,
    axis: int = 0,
    *,
    norm: str = "backward",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """One-sided FFT magnitude spectrum of a real signal.

    The default is **unnormalized** — the raw ``|rfft(signal)|``, with
    no rectangular window correction and no scaling. Raw magnitude
    scales with signal length, so values are comparable across bins and
    across signals of the *same* length only; select a normalization
    via ``norm`` before comparing spectra of different lengths or
    against published amplitudes. :func:`sparc` normalizes by its own
    peak, so this choice does not affect it.

    Parameters
    ----------
    signal : ndarray
        Real input signal.
    fs : float, optional
        Sampling rate in Hz (default 1.0).
    axis : int, optional
        Axis to transform along (default 0).
    norm : {"backward", "ortho", "forward", "amplitude"}, keyword-only, optional
        Normalization of the returned magnitude. The first three follow
        numpy's ``rfft`` vocabulary: ``"backward"`` (default) applies no
        scaling — the raw ``|rfft|``; ``"ortho"`` divides by ``√N``;
        ``"forward"`` divides by ``N``. ``"amplitude"`` is the
        single-sided amplitude spectrum the signal-processing literature
        plots — ``2|X|/N``, with the DC bin (and the Nyquist bin, for
        even ``N``) *not* doubled, since those frequencies have no
        negative-frequency twin to fold in — so a pure sine of amplitude
        ``A`` peaks at ``A``. Note ``"amplitude"`` is **not** one of
        numpy's norms: numpy's ``"forward"`` lacks the one-sided
        doubling, so it reads half the sine's amplitude.

    Returns
    -------
    freqs : ndarray, shape (T//2 + 1,)
        Non-negative frequency bins in Hz.
    magnitude : ndarray
        ``|rfft(signal)|`` along ``axis``, scaled per ``norm``.

    Raises
    ------
    ValueError
        If ``norm`` is not one of the four options.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.shape[axis]
    magnitude = np.abs(np.fft.rfft(signal, axis=axis))
    if norm == "ortho":
        magnitude /= np.sqrt(n)
    elif norm == "forward":
        magnitude /= n
    elif norm == "amplitude":
        magnitude *= 2.0 / n
        # DC (and Nyquist, when it exists as its own bin) appear once in
        # the full spectrum, so folding to one side must not double them.
        edge: list[int | slice] = [slice(None)] * magnitude.ndim
        edge[axis] = 0
        magnitude[tuple(edge)] /= 2.0
        if n % 2 == 0:
            edge[axis] = -1
            magnitude[tuple(edge)] /= 2.0
    elif norm != "backward":
        raise ValueError(
            f"norm must be 'backward', 'ortho', 'forward' or 'amplitude', got {norm!r}"
        )
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, magnitude  # type: ignore[return-value]  # NumPy 2.2's stub types rfftfreq as floating, not float64


def dominant_frequency(
    signal: npt.NDArray[np.float64],
    fs: float,
    axis: int = 0,
) -> npt.NDArray[np.float64]:
    """Frequency (Hz) of the largest non-DC spectral component.

    The DC bin is excluded so a non-zero mean doesn't dominate.

    The peak is the argmax of the raw spectrum at native bin resolution
    ``fs / T``: no zero-padding, no window, no sub-bin interpolation.
    Resolution is therefore ``fs / T``, and rectangular-window leakage
    can move the winning bin for a tone falling between bins — pad the
    signal or use quadratic peak interpolation if you need better than
    bin precision. Exact ties go to the **lowest** frequency
    (``np.argmax``).

    Parameters
    ----------
    signal : ndarray
        Real input signal.
    fs : float
        Sampling rate in Hz.
    axis : int, optional
        Axis to analyze along (default 0).

    Returns
    -------
    ndarray
        Dominant frequency with ``axis`` removed (a scalar for 1-D input).
    """
    freqs, magnitude = fft_magnitude(signal, fs, axis=axis)
    moved = np.moveaxis(magnitude, axis, 0).copy()
    moved[0] = 0.0  # exclude the DC bin so a non-zero mean doesn't win
    peak = np.argmax(moved, axis=0)
    return freqs[peak]


def ramer_douglas_peucker(
    curve: npt.NDArray[np.float64],
    eps: float,
) -> npt.NDArray[np.float64]:
    """Simplify a polyline with the Ramer–Douglas–Peucker algorithm.

    Drops points that lie within ``eps`` of the simplified path, keeping the
    overall shape. Operates on a single curve (recursion over the curve's
    own points, not over a batch); each split's perpendicular distances are
    computed vectorized.

    Parameters
    ----------
    curve : ndarray, shape (P, D)
        Ordered points of one curve (any dimension ``D``).
    eps : float
        Maximum allowed perpendicular deviation.

    Returns
    -------
    ndarray, shape (K, D)
        The retained points (endpoints always kept), in order.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape[0] <= 2:
        return curve.copy()
    keep = np.zeros(curve.shape[0], dtype=bool)
    keep[0] = keep[-1] = True
    # Explicit stack of index ranges (not recursion) so a pathological
    # high-frequency curve cannot overflow Python's recursion limit.
    stack: list[tuple[int, int]] = [(0, curve.shape[0] - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        inner = curve[lo + 1 : hi]
        start, end = curve[lo], curve[hi]
        seg = end - start
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-12:
            dist = np.linalg.norm(inner - start, axis=-1)
        else:
            u = seg / seg_len
            rel = inner - start
            proj = (rel @ u)[:, None] * u
            dist = np.linalg.norm(rel - proj, axis=-1)
        offset = int(np.argmax(dist))
        if dist[offset] > eps:
            split = lo + 1 + offset
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))
    return curve[keep]
