"""
Core SSI-COV (Stochastic Subspace Identification - Covariance) implementation.

This module handles the mathematical core of the SSI-COV algorithm:
- Building block-Toeplitz matrices from acceleration data
- Performing SVD decomposition
- Extracting natural frequencies, damping ratios, and mode shapes
- Hard Criteria (HC) filtering: complex conjugates, damping, frequency limits
- Computing stabilisation diagrams and spectral analysis

Inspired by pyOMA2 (Pasca et al.) and the MATLAB SSI-COV toolbox (Cheynet, 2020).

References
----------
[1] Peeters, B., & De Roeck, G. (1999). Reference-based stochastic subspace
    identification for output-only modal analysis. MSSP, 13(6), 855-878.
[2] Reynders, E. (2012). System identification methods for (operational) modal
    analysis: Review and comparison. ACME, 19, 51-124.
[3] Magalhaes, F., Cunha, A., & Caetano, E. (2009). Online automatic
    identification of modal parameters. MSSP, 23(2), 316-329.
"""

import numpy as np
from scipy import signal
from scipy.linalg import svd, eig
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Autocorrelation & Toeplitz
# ---------------------------------------------------------------------------

def compute_autocorrelation(acceleration_data, max_lag=None, detrend=True):
    """
    Compute the cross-correlation matrix for each lag.

    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
        Acceleration time series data from all sensors.
    max_lag : int, optional
        Maximum lag for autocorrelation. If None, use n_samples // 2.
    detrend : bool, default True
        Whether to detrend the data before computing correlations.

    Returns
    -------
    acf : ndarray, shape (max_lag, n_sensors, n_sensors)
        Cross-correlation matrices for each lag.
    """
    if detrend:
        acceleration_data = signal.detrend(acceleration_data, axis=0)

    n_samples, n_sensors = acceleration_data.shape
    if max_lag is None:
        max_lag = n_samples // 2

    acf = np.zeros((max_lag, n_sensors, n_sensors))

    for lag in range(max_lag):
        if lag == 0:
            acf[lag] = (acceleration_data.T @ acceleration_data) / n_samples
        else:
            acf[lag] = (
                acceleration_data[:-lag].T @ acceleration_data[lag:]
            ) / (n_samples - lag)

    return acf


def build_toeplitz_matrix(acf, block_rows):
    """
    Build the block-Toeplitz matrix from autocorrelation data.

    Parameters
    ----------
    acf : ndarray, shape (max_lag, n_sensors, n_sensors)
        Cross-correlation matrices for each lag.
    block_rows : int
        Number of block rows (relates to the model order).

    Returns
    -------
    H : ndarray, shape (block_rows * n_sensors, block_rows * n_sensors)
        Block-Toeplitz matrix.
    """
    n_lags, n_sensors, _ = acf.shape
    H = np.zeros((block_rows * n_sensors, block_rows * n_sensors))

    for i in range(block_rows):
        for j in range(block_rows):
            lag = abs(i - j)
            if lag < n_lags:
                H[
                    i * n_sensors : (i + 1) * n_sensors,
                    j * n_sensors : (j + 1) * n_sensors,
                ] = acf[lag]

    return H


# ---------------------------------------------------------------------------
# Hard Criteria (HC) — following pyOMA2 conventions
# ---------------------------------------------------------------------------

def _hc_conjugate_pairs(eigenvalues):
    """
    Hard criterion: keep only one eigenvalue from each complex-conjugate pair.

    For each pair, keep the eigenvalue with *positive* imaginary part and
    discard the negative conjugate.  Real eigenvalues (imag approx 0) are kept.

    Returns
    -------
    mask : ndarray of bool
        True for eigenvalues to keep.
    """
    mask = np.ones(len(eigenvalues), dtype=bool)
    used = set()

    for i in range(len(eigenvalues)):
        if i in used:
            continue
        ev = eigenvalues[i]
        if np.abs(ev.imag) < 1e-12:
            continue  # real eigenvalue — keep
        # Find conjugate partner
        for j in range(i + 1, len(eigenvalues)):
            if j in used:
                continue
            if np.abs(ev - eigenvalues[j].conj()) < 1e-10:
                # Keep positive imaginary part
                if ev.imag >= 0:
                    mask[j] = False
                else:
                    mask[i] = False
                used.add(i)
                used.add(j)
                break
        else:
            # No conjugate found — keep only if positive imaginary
            if ev.imag < 0:
                mask[i] = False

    return mask


def _hc_damping(damping_ratios, xi_max=1.0):
    """
    Hard criterion: keep only poles with 0 < damping < xi_max.

    Parameters
    ----------
    damping_ratios : ndarray
    xi_max : float
        Maximum plausible damping ratio (default 1.0).

    Returns
    -------
    mask : ndarray of bool
    """
    return (damping_ratios > 0) & (damping_ratios < xi_max)


def _hc_frequency_limits(frequencies, sampling_freq):
    """
    Hard criterion: keep only poles with 0 < f < fs/2 (Nyquist).

    Returns
    -------
    mask : ndarray of bool
    """
    return (frequencies > 0) & (frequencies < sampling_freq / 2.0)


# ---------------------------------------------------------------------------
# Core SSI-COV
# ---------------------------------------------------------------------------

def perform_ssi_cov(
    acceleration_data,
    order,
    max_lag=None,
    use_randomized_svd=True,
    sampling_freq=1.0,
    xi_max=1.0,
):
    """
    Perform the SSI-COV algorithm to extract modal parameters.

    Implements the covariance-driven stochastic subspace identification
    following the classical formulation (Van Overschee & De Moor, 2012)
    with Hard Criteria filtering inspired by pyOMA2.

    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
        Acceleration time series data.
    order : int
        The model order (number of state-space dimensions = 2 * n_modes).
    max_lag : int, optional
        Maximum lag for autocorrelation. If None, use min(n_samples // 4, 1000).
    use_randomized_svd : bool, default True
        Use randomized SVD for faster computation on large matrices.
    sampling_freq : float, default 1.0
        Sampling frequency in Hz.
    xi_max : float, default 1.0
        Maximum plausible damping ratio for Hard Criteria filtering.

    Returns
    -------
    frequencies : ndarray, shape (n_modes,)
        Natural frequencies in Hz.
    damping_ratios : ndarray, shape (n_modes,)
        Damping ratios (0 to 1).
    mode_shapes : ndarray, shape (n_sensors, n_modes)
        Mode shape matrix (column vectors are mode shapes, real-valued).
    singular_values : ndarray
        Singular values from SVD (for model order selection / stabilisation).
    """
    n_samples, n_sensors = acceleration_data.shape
    dt = 1.0 / sampling_freq

    if max_lag is None:
        max_lag = min(int(n_samples // 4), 1000)

    # Step 1: Autocorrelation
    acf = compute_autocorrelation(acceleration_data, max_lag=max_lag, detrend=True)

    # Step 2: Block-Toeplitz matrix
    block_rows = max(order // n_sensors + 1, 2)
    H = build_toeplitz_matrix(acf, block_rows)

    # Step 3: SVD
    if use_randomized_svd:
        n_comp = min(2 * order, H.shape[0] - 1, H.shape[1] - 1)
        U, s, Vt = randomized_svd(H, n_components=n_comp, random_state=42)
    else:
        U, s, Vt = svd(H, full_matrices=False)

    # Step 4: Truncate to model order
    n_keep = min(2 * order, len(s))
    U_trunc = U[:, :n_keep]
    s_trunc = s[:n_keep]

    # Step 5: Observability matrix
    Sigma_sqrt = np.diag(np.sqrt(s_trunc))
    O = U_trunc @ Sigma_sqrt

    # Step 6: State matrix via block-shift property  A = O1^+ @ O2
    O1 = O[:-n_sensors, :]
    O2 = O[n_sensors:, :]

    try:
        A = O2 @ np.linalg.pinv(O1)
    except np.linalg.LinAlgError:
        A = np.linalg.lstsq(O1.T, O2.T, rcond=None)[0].T

    # Step 7: Eigendecomposition
    eigenvalues, eigenvectors = eig(A)

    # Step 8: HC — complex conjugate filtering (keep positive imag only)
    conj_mask = _hc_conjugate_pairs(eigenvalues)
    eigenvalues = eigenvalues[conj_mask]
    eigenvectors = eigenvectors[:, conj_mask]

    # Step 9: Discrete-to-continuous eigenvalue conversion & modal parameters
    frequencies = []
    damping_ratios = []
    mode_shapes_list = []

    C_matrix = U_trunc[:n_sensors, :]  # Output matrix

    for i in range(len(eigenvalues)):
        eig_val = eigenvalues[i]

        if np.abs(eig_val) < 1e-10:
            continue

        try:
            lambda_c = np.log(eig_val + 0j) / dt
        except (ValueError, FloatingPointError):
            continue

        real_part = lambda_c.real
        imag_part = lambda_c.imag

        wn = np.sqrt(real_part ** 2 + imag_part ** 2)
        if wn < 1e-10:
            continue

        zeta = -real_part / wn
        fn = np.abs(imag_part) / (2 * np.pi)

        frequencies.append(fn)
        damping_ratios.append(zeta)

        # Mode shape: C @ eigenvector
        mode_eig = eigenvectors[:, i]
        try:
            mode_shape = C_matrix @ mode_eig
            mode_shapes_list.append(mode_shape)
        except (ValueError, IndexError):
            frequencies.pop()
            damping_ratios.pop()

    frequencies = np.array(frequencies, dtype=float)
    damping_ratios = np.array(damping_ratios, dtype=float)

    if len(frequencies) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.empty((n_sensors, 0), dtype=float),
            s,
        )

    mode_shapes = np.column_stack(mode_shapes_list)

    # Step 10: HC — damping
    damp_mask = _hc_damping(damping_ratios, xi_max=xi_max)
    frequencies = frequencies[damp_mask]
    damping_ratios = damping_ratios[damp_mask]
    mode_shapes = mode_shapes[:, damp_mask]

    # Step 11: HC — frequency limits (Nyquist)
    freq_mask = _hc_frequency_limits(frequencies, sampling_freq)
    frequencies = frequencies[freq_mask]
    damping_ratios = damping_ratios[freq_mask]
    mode_shapes = mode_shapes[:, freq_mask]

    if len(frequencies) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.empty((n_sensors, 0), dtype=float),
            s,
        )

    # Normalise mode shapes to unit norm (real part)
    mode_shapes = np.real(mode_shapes)
    for i in range(mode_shapes.shape[1]):
        norm = np.linalg.norm(mode_shapes[:, i])
        if norm > 1e-10:
            mode_shapes[:, i] /= norm

    # Sort by frequency
    sort_idx = np.argsort(frequencies)
    frequencies = frequencies[sort_idx]
    damping_ratios = damping_ratios[sort_idx]
    mode_shapes = mode_shapes[:, sort_idx]

    return frequencies, damping_ratios, mode_shapes, s


# ---------------------------------------------------------------------------
# Stability filtering across ALL consecutive model orders
# ---------------------------------------------------------------------------

def extract_stable_poles(
    frequencies_list,
    damping_list,
    mode_shapes_list,
    freq_tol=0.01,
    damp_tol=0.05,
    mac_threshold=0.98,
):
    """
    Filter stable poles by comparing across ALL consecutive model orders.

    A pole in order *k* is labelled "stable" if a matching pole exists in
    order *k+1* satisfying frequency, damping, and MAC criteria.  Only poles
    that are stable in **every** consecutive pair they appear in are returned.

    Parameters
    ----------
    frequencies_list : list of ndarray
        Frequencies from each model order.
    damping_list : list of ndarray
        Damping ratios from each model order.
    mode_shapes_list : list of ndarray
        Mode shapes from each model order, each shape (n_sensors, n_poles_k).
    freq_tol : float, default 0.01
        Relative frequency tolerance (1%).
    damp_tol : float, default 0.05
        Absolute damping tolerance (5%).
    mac_threshold : float, default 0.98
        Modal Assurance Criterion threshold.

    Returns
    -------
    stable_freqs : ndarray
    stable_damps : ndarray
    stable_modes : ndarray, shape (n_sensors, n_stable)
    """
    from .automation import mac as compute_mac  # avoid circular import

    if len(frequencies_list) < 2:
        if len(frequencies_list) == 0:
            return np.array([]), np.array([]), np.empty((0, 0))
        return frequencies_list[0], damping_list[0], mode_shapes_list[0]

    n_orders = len(frequencies_list)

    # Start with the first order and propagate stability through all pairs.
    current_freqs = frequencies_list[0]
    current_damps = damping_list[0]
    current_modes = mode_shapes_list[0]
    stable_mask = np.ones(len(current_freqs), dtype=bool)

    for k in range(n_orders - 1):
        next_freqs = frequencies_list[k + 1]
        next_damps = damping_list[k + 1]
        next_modes = mode_shapes_list[k + 1]

        pair_mask = np.zeros(len(current_freqs), dtype=bool)

        for i in range(len(current_freqs)):
            if not stable_mask[i]:
                continue
            ref_f = current_freqs[i]
            ref_d = current_damps[i]
            ref_m = current_modes[:, i]

            for j in range(len(next_freqs)):
                freq_diff = np.abs(ref_f - next_freqs[j]) / (ref_f + 1e-12)
                damp_diff = np.abs(ref_d - next_damps[j])
                mac_val = compute_mac(ref_m, next_modes[:, j])

                if (
                    freq_diff < freq_tol
                    and damp_diff < damp_tol
                    and mac_val > mac_threshold
                ):
                    pair_mask[i] = True
                    break

        stable_mask &= pair_mask

    stable_freqs = current_freqs[stable_mask]
    stable_damps = current_damps[stable_mask]
    stable_modes = current_modes[:, stable_mask]

    return stable_freqs, stable_damps, stable_modes


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def compute_psd(acceleration_data, sampling_freq=1.0, method="welch", nperseg=None):
    """
    Compute Power Spectral Density using Welch or FFT method.

    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
    sampling_freq : float, default 1.0
    method : str, default 'welch'
        'welch' for Welch method or 'fft' for simple FFT.
    nperseg : int, optional
        Length of each segment for Welch method.

    Returns
    -------
    frequencies : ndarray
        Frequency array in Hz.
    psd_values : ndarray, shape (n_frequencies, n_sensors)
    """
    n_samples, n_sensors = acceleration_data.shape

    if nperseg is None:
        nperseg = min(n_samples // 4, 1024)
    nperseg = max(nperseg, 16)  # safety lower bound

    if method == "welch":
        psd_list = []
        for i in range(n_sensors):
            freqs, pxx = signal.welch(
                acceleration_data[:, i],
                fs=sampling_freq,
                nperseg=nperseg,
                scaling="spectrum",
            )
            psd_list.append(pxx)
        frequencies = freqs
        psd_values = np.column_stack(psd_list)
    else:
        fft_vals = np.fft.fft(acceleration_data, axis=0)
        frequencies = np.fft.fftfreq(n_samples, 1.0 / sampling_freq)
        pos = frequencies >= 0
        frequencies = frequencies[pos]
        psd_values = np.abs(fft_vals[pos, :]) ** 2 / (sampling_freq * n_samples)

    return frequencies, psd_values


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_stabilization_diagram(
    acceleration_data,
    order_range,
    sampling_freq=1.0,
    xi_max=1.0,
    figsize=(12, 8),
):
    """
    Plot stabilisation diagram for model order selection.

    Stable poles appear as near-vertical alignments across model orders.

    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
    order_range : iterable of int
    sampling_freq : float, default 1.0
    xi_max : float, default 1.0
    figsize : tuple, default (12, 8)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    frequencies_by_order = []

    for order in order_range:
        freqs, _, _, _ = perform_ssi_cov(
            acceleration_data,
            order=order,
            sampling_freq=sampling_freq,
            xi_max=xi_max,
        )
        frequencies_by_order.append(freqs)

    fig, ax = plt.subplots(figsize=figsize)

    for i, order in enumerate(order_range):
        freqs = frequencies_by_order[i]
        for freq in freqs:
            ax.plot(order, freq, "b.", markersize=6)

    ax.set_xlabel("Model Order", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title("SSI-COV Stabilisation Diagram", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_psd_with_peaks(
    acceleration_data,
    sampling_freq=1.0,
    frequencies=None,
    figsize=(14, 6),
):
    """
    Plot Power Spectral Density with identified natural frequencies marked.

    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
    sampling_freq : float, default 1.0
    frequencies : ndarray, optional
        Natural frequencies to highlight.
    figsize : tuple, default (14, 6)

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    freq_range, psd = compute_psd(acceleration_data, sampling_freq=sampling_freq)
    n_sensors = psd.shape[1]

    fig, axes = plt.subplots(n_sensors, 1, figsize=figsize, sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.semilogy(freq_range, psd[:, i], "b-", linewidth=1.5, label="PSD")

        if frequencies is not None:
            for freq in frequencies:
                ax.axvline(freq, color="r", linestyle="--", alpha=0.6, linewidth=1)

        ax.set_ylabel(f"Sensor {i + 1}\nPSD", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=12)
    fig.suptitle("Power Spectral Density Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, axes


def plot_singular_values(singular_values, figsize=(10, 6)):
    """
    Plot singular values from SVD with knee-point detection.

    Parameters
    ----------
    singular_values : ndarray
    figsize : tuple, default (10, 6)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    sv_normalized = singular_values / singular_values[0]
    ax.semilogy(
        range(len(singular_values)),
        sv_normalized,
        "bo-",
        linewidth=2,
        markersize=6,
    )

    cumsum_sv = np.cumsum(sv_normalized)
    cumsum_norm = cumsum_sv / cumsum_sv[-1]
    knee_idx = np.where(cumsum_norm > 0.95)[0]
    if len(knee_idx) > 0:
        ax.axvline(
            knee_idx[0],
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"95% Energy (Order ~ {knee_idx[0]})",
        )

    ax.set_xlabel("Singular Value Index", fontsize=12)
    ax.set_ylabel("Normalised Singular Value", fontsize=12)
    ax.set_title("SVD Singular Value Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    plt.tight_layout()
    return fig, ax
