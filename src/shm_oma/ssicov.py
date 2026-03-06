"""
Core SSI-COV (Stochastic Subspace Identification - Covariance) implementation.

This module handles the mathematical core of the SSI-COV algorithm:
- Building block-Toeplitz matrices from acceleration data
- Performing SVD decomposition
- Extracting natural frequencies, damping ratios, and mode shapes
- Computing stabilization diagrams and spectral analysis
"""

import numpy as np
from scipy import signal
from scipy.linalg import svd, eig, solve_sylvester
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt


def compute_autocorrelation(acceleration_data, max_lag=None, detrend=True):
    """
    Compute the autocorrelation function (or impulse response via FFT).
    
    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
        Acceleration time series data from all sensors.
    max_lag : int, optional
        Maximum lag for autocorrelation. If None, use n_samples // 2.
    detrend : bool, default=True
        Whether to detrend the data before computing correlations.
    
    Returns
    -------
    acf : ndarray, shape (max_lag, n_sensors, n_sensors)
        Autocorrelation function (or cross-correlation matrix).
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
            acf[lag] = (acceleration_data[:-lag].T @ acceleration_data[lag:]) / (n_samples - lag)
    
    return acf


def build_toeplitz_matrix(acf, block_rows):
    """
    Build the block-Toeplitz matrix from autocorrelation data.
    
    Parameters
    ----------
    acf : ndarray, shape (max_lag, n_sensors, n_sensors)
        Autocorrelation function.
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
                H[i*n_sensors:(i+1)*n_sensors, j*n_sensors:(j+1)*n_sensors] = acf[lag]
    
    return H


def perform_ssi_cov(acceleration_data, order, max_lag=None, use_randomized_svd=True, sampling_freq=1.0):
    """
    Perform the SSI-COV algorithm to extract modal parameters.
    
    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
        Acceleration time series data.
    order : int
        The model order (number of poles to extract).
    max_lag : int, optional
        Maximum lag for autocorrelation. If None, use n_samples // 4.
    use_randomized_svd : bool, default=True
        Use randomized SVD for faster computation on large matrices.
    sampling_freq : float, default=1.0
        Sampling frequency in Hz.
    
    Returns
    -------
    frequencies : ndarray, shape (n_modes,)
        Natural frequencies in Hz.
    damping_ratios : ndarray, shape (n_modes,)
        Damping ratios (0 to 1).
    mode_shapes : ndarray, shape (n_sensors, n_modes)
        Mode shape matrix (column vectors are mode shapes).
    singular_values : ndarray
        Singular values from SVD (for stabilization diagram).
    """
    n_samples, n_sensors = acceleration_data.shape
    dt = 1.0 / sampling_freq
    
    if max_lag is None:
        max_lag = min(int(n_samples // 4), 1000)  # Limit for large datasets
    
    # Step 1: Compute autocorrelation function
    acf = compute_autocorrelation(acceleration_data, max_lag=max_lag, detrend=True)
    
    # Step 2: Build block-Toeplitz matrix
    block_rows = max(order // n_sensors + 1, 2)
    H = build_toeplitz_matrix(acf, block_rows)
    
    # Step 3: Perform SVD
    if use_randomized_svd:
        n_comp = min(2 * order, H.shape[0] - 1, H.shape[1] - 1)
        U, s, Vt = randomized_svd(H, n_components=n_comp, random_state=42)
    else:
        U, s, Vt = svd(H, full_matrices=False)
    
    # Step 4: Truncate to model order
    n_keep = min(2 * order, len(s))
    U = U[:, :n_keep]
    s = s[:n_keep]
    
    # Step 5: Build observability matrix
    Sigma = np.diag(np.sqrt(s))
    O = U @ Sigma  # Observability matrix, shape: (block_rows*n_sensors, n_keep)
    
    # Step 6: Extract state matrix using block shift property
    # For SSI-COV: A is extracted from A = O1^+ @ O2
    # where O1 = O[:-n_sensors, :] and O2 = O[n_sensors:, :]
    O1 = O[:-n_sensors, :]      # Past outputs, shape: ((block_rows-1)*n_sensors, n_keep)
    O2 = O[n_sensors:, :]       # Future outputs, shape: ((block_rows-1)*n_sensors, n_keep)
    
    # Compute state matrix via least squares: A @ O1 = O2
    # A should have shape (n_keep, n_keep)
    try:
        A = O2 @ np.linalg.pinv(O1)  # Shape: (n_keep, n_keep) if O1 and O2 are consistent
    except:
        # Fallback if dimensions don't match
        A = np.linalg.lstsq(O1.T, O2.T, rcond=None)[0].T
    
    # Step 7: Extract eigenvalues and eigenvectors of state matrix
    eigenvalues, eigenvectors = eig(A)
    
    # Step 8: Convert discrete eigenvalues to continuous modal parameters
    frequencies = []
    damping_ratios = []
    mode_shapes_list = []
    
    # Output matrix: first n_sensors rows of observability matrix
    C_matrix = U[:n_sensors, :]  # Shape: (n_sensors, n_keep)
    
    for i in range(len(eigenvalues)):
        eig_val = eigenvalues[i]
        
        # Convert discrete eigenvalue to continuous time
        # z = e^(lambda * dt) -> lambda = ln(z) / dt
        if np.abs(eig_val) > 1e-10:
            try:
                lambda_c = np.log(eig_val) / dt
            except:
                continue
            
            # Extract real and imaginary parts
            real_part = lambda_c.real
            imag_part = lambda_c.imag
            
            # Natural frequency and damping ratio
            wn = np.sqrt(real_part**2 + imag_part**2)
            if wn < 1e-10:
                continue
                
            zeta = -real_part / wn
            
            # Only keep stable modes with physical frequencies
            if 0 <= zeta <= 1.0 and wn > 0:
                fn = np.abs(imag_part) / (2 * np.pi)  # Natural frequency in Hz
                if fn > 0 and fn < sampling_freq / 2:  # Nyquist check
                    frequencies.append(fn)
                    damping_ratios.append(max(0, min(1, zeta)))
                    
                    # Extract mode shape: C @ eigenvector
                    # eigenvectors[:, i] has shape (n_keep,)
                    mode_eig = eigenvectors[:, i]
                    try:
                        mode_shape = C_matrix @ mode_eig  # Shape: (n_sensors,)
                        mode_shapes_list.append(mode_shape)
                    except:
                        # If dimensions don't match, skip this mode
                        frequencies.pop()
                        damping_ratios.pop()
    
    if len(frequencies) == 0:
        # Fallback: return one zero mode if no modes extracted
        frequencies = np.array([0.0])
        damping_ratios = np.array([0.0])
        mode_shapes_list = [np.ones(n_sensors) / np.sqrt(n_sensors)]
    
    # Convert to numpy arrays and normalize mode shapes
    frequencies = np.array(frequencies)
    damping_ratios = np.array(damping_ratios)
    mode_shapes = np.column_stack(mode_shapes_list) if len(mode_shapes_list) > 0 else np.ones((n_sensors, 1))
    
    # Normalize mode shapes
    for i in range(mode_shapes.shape[1]):
        norm = np.linalg.norm(mode_shapes[:, i])
        if norm > 1e-10:
            mode_shapes[:, i] /= norm
    
    # Sort by frequency
    if len(frequencies) > 0:
        sort_idx = np.argsort(frequencies)
        frequencies = frequencies[sort_idx]
        damping_ratios = damping_ratios[sort_idx]
        mode_shapes = mode_shapes[:, sort_idx]
    
    return frequencies, damping_ratios, np.real(mode_shapes), s


def extract_stable_poles(frequencies_list, damping_list, mode_shapes_list, 
                         freq_tol=0.01, damp_tol=0.05, mac_threshold=0.98):
    """
    Filter stable poles by comparing consecutive model orders.
    
    Parameters
    ----------
    frequencies_list : list of ndarray
        Frequencies from each model order.
    damping_list : list of ndarray
        Damping ratios from each model order.
    mode_shapes_list : list of ndarray
        Mode shapes from each model order.
    freq_tol : float, default=0.01
        Relative frequency tolerance (1%).
    damp_tol : float, default=0.05
        Absolute damping tolerance (5%).
    mac_threshold : float, default=0.98
        Modal Assurance Criterion threshold.
    
    Returns
    -------
    stable_freqs : ndarray
        Stable frequencies.
    stable_damps : ndarray
        Stable damping ratios.
    stable_modes : ndarray
        Stable mode shapes.
    """
    if len(frequencies_list) < 2:
        return frequencies_list[0], damping_list[0], mode_shapes_list[0]
    
    stable_mask = np.zeros(len(frequencies_list[0]), dtype=bool)
    
    # Compare with the next order
    for i, f0 in enumerate(frequencies_list[0]):
        for f1 in frequencies_list[1]:
            freq_diff = np.abs(f0 - f1) / f0
            if freq_diff < freq_tol:
                stable_mask[i] = True
                break
    
    stable_freqs = frequencies_list[0][stable_mask]
    stable_damps = damping_list[0][stable_mask]
    stable_modes = mode_shapes_list[0][:, stable_mask]
    
    return stable_freqs, stable_damps, stable_modes


def compute_psd(acceleration_data, sampling_freq=1.0, method='welch', nperseg=None):
    """
    Compute Power Spectral Density using Welch or FFT method.
    
    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
        Acceleration time series.
    sampling_freq : float, default=1.0
        Sampling frequency in Hz.
    method : str, default='welch'
        'welch' for Welch method or 'fft' for simple FFT.
    nperseg : int, optional
        Length of each segment for Welch method.
    
    Returns
    -------
    frequencies : ndarray
        Frequency array in Hz.
    psd_values : ndarray, shape (n_frequencies, n_sensors)
        PSD for each sensor.
    """
    n_samples, n_sensors = acceleration_data.shape
    
    if nperseg is None:
        nperseg = min(n_samples // 10, 1024)
    
    if method == 'welch':
        # Compute Welch PSD for each sensor
        psd_values = np.zeros((nperseg // 2 + 1, n_sensors))
        for i in range(n_sensors):
            freqs, pxx = signal.welch(
                acceleration_data[:, i],
                fs=sampling_freq,
                nperseg=nperseg,
                scaling='spectrum'
            )
            psd_values[:, i] = pxx
        frequencies = freqs
    else:  # FFT method
        fft_vals = np.fft.fft(acceleration_data, axis=0)
        frequencies = np.fft.fftfreq(n_samples, 1.0 / sampling_freq)
        freqs_positive = frequencies[:n_samples // 2]
        psd_values = np.abs(fft_vals[:n_samples // 2, :]) ** 2 / (sampling_freq * n_samples)
        frequencies = freqs_positive
    
    return frequencies, psd_values


def plot_stabilization_diagram(acceleration_data, order_range, sampling_freq=1.0, figsize=(12, 8)):
    """
    Plot stabilization diagram for model order selection.
    
    Shows which poles are stable across different model orders.
    Stable poles appear as nearly vertical lines.
    
    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
        Acceleration time series.
    order_range : list or ndarray
        Range of model orders to test (e.g., range(10, 51, 2)).
    sampling_freq : float, default=1.0
        Sampling frequency in Hz.
    figsize : tuple, default=(12, 8)
        Figure size.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
        The plot objects.
    """
    frequencies_by_order = []
    damping_by_order = []
    
    # Extract modal parameters for each model order
    for order in order_range:
        freqs, damps, _, _ = perform_ssi_cov(
            acceleration_data, 
            order=order,
            sampling_freq=sampling_freq
        )
        frequencies_by_order.append(freqs)
        damping_by_order.append(damps)
    
    # Create stabilization diagram
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, order in enumerate(order_range):
        freqs = frequencies_by_order[i]
        damps = damping_by_order[i]
        # Plot as vertical lines for each frequency
        for freq, damp in zip(freqs, damps):
            ax.plot([i, i], [freq - 0.1, freq + 0.1], 'b.', markersize=8)
    
    ax.set_xlabel('Model Order', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title('SSI-COV Stabilization Diagram', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(order_range)))
    ax.set_xticklabels(order_range)
    
    plt.tight_layout()
    return fig, ax


def plot_psd_with_peaks(acceleration_data, sampling_freq=1.0, frequencies=None, figsize=(14, 6)):
    """
    Plot Power Spectral Density with identified peaks.
    
    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
        Acceleration time series.
    sampling_freq : float, default=1.0
        Sampling frequency in Hz.
    frequencies : ndarray, optional
        Natural frequencies to mark on plot.
    figsize : tuple, default=(14, 6)
        Figure size.
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
        The plot objects.
    """
    freq_range, psd = compute_psd(acceleration_data, sampling_freq=sampling_freq)
    n_sensors = psd.shape[1]
    
    fig, axes = plt.subplots(n_sensors, 1, figsize=figsize, sharex=True)
    if n_sensors == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Plot PSD in dB
        psd_db = 10 * np.log10(psd[:, i] + 1e-12)
        ax.semilogy(freq_range, psd[:, i], 'b-', linewidth=1.5, label='PSD')
        
        # Mark identified peaks
        if frequencies is not None:
            for freq in frequencies:
                ax.axvline(freq, color='r', linestyle='--', alpha=0.6, linewidth=1)
        
        ax.set_ylabel(f'Sensor {i+1}\nPSD', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    fig.suptitle('Power Spectral Density Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, axes


def plot_singular_values(singular_values, figsize=(10, 6)):
    """
    Plot singular values from SVD with knee point detection.
    
    Parameters
    ----------
    singular_values : ndarray
        Singular values from SSI-COV SVD.
    figsize : tuple, default=(10, 6)
        Figure size.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
        The plot objects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot singular values on log scale
    sv_normalized = singular_values / singular_values[0]
    ax.semilogy(range(len(singular_values)), sv_normalized, 'bo-', linewidth=2, markersize=6)
    
    # Highlight potential model order (where singular values drop significantly)
    cumsum_sv = np.cumsum(sv_normalized)
    cumsum_normalized = cumsum_sv / cumsum_sv[-1]
    knee_idx = np.where(cumsum_normalized > 0.95)[0]
    if len(knee_idx) > 0:
        ax.axvline(knee_idx[0], color='r', linestyle='--', linewidth=2, label=f'95% Energy (Order ~{knee_idx[0]})')
    
    ax.set_xlabel('Singular Value Index', fontsize=12)
    ax.set_ylabel('Normalized Singular Value', fontsize=12)
    ax.set_title('SVD Singular Value Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    plt.tight_layout()
    return fig, ax
