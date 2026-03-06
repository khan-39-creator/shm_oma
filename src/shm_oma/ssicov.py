"""
Core SSI-COV (Stochastic Subspace Identification - Covariance) implementation.

This module handles the mathematical core of the SSI-COV algorithm:
- Building block-Toeplitz matrices from acceleration data
- Performing SVD decomposition
- Extracting natural frequencies, damping ratios, and mode shapes
"""

import numpy as np
from scipy import signal
from scipy.linalg import svd, eig
from sklearn.utils.extmath import randomized_svd


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


def perform_ssi_cov(acceleration_data, order, max_lag=None, use_randomized_svd=True):
    """
    Perform the SSI-COV algorithm to extract modal parameters.
    
    Parameters
    ----------
    acceleration_data : ndarray, shape (n_samples, n_sensors)
        Acceleration time series data.
    order : int
        The model order (number of poles to extract).
    max_lag : int, optional
        Maximum lag for autocorrelation. If None, use n_samples // 2.
    use_randomized_svd : bool, default=True
        Use randomized SVD for faster computation on large matrices.
    
    Returns
    -------
    frequencies : ndarray, shape (order,)
        Natural frequencies in Hz.
    damping_ratios : ndarray, shape (order,)
        Damping ratios.
    mode_shapes : ndarray, shape (n_sensors, order)
        Mode shape matrix (column vectors are mode shapes).
    """
    n_samples, n_sensors = acceleration_data.shape
    
    if max_lag is None:
        max_lag = n_samples // 2
    
    # Step 1: Compute autocorrelation
    acf = compute_autocorrelation(acceleration_data, max_lag=max_lag)
    
    # Step 2: Build block-Toeplitz matrix
    block_rows = (order + n_sensors - 1) // n_sensors
    H = build_toeplitz_matrix(acf, block_rows)
    
    # Step 3: Perform SVD
    if use_randomized_svd:
        # Use randomized SVD for efficiency on large matrices
        U, s, Vt = randomized_svd(H, n_components=min(order, H.shape[0] - 1), random_state=42)
    else:
        U, s, Vt = svd(H, full_matrices=False)
    
    # Step 4: Extract observability matrix and construct state matrix
    # We retain order singular values/vectors
    n_keep = min(order, len(s))
    U = U[:, :n_keep]
    s = s[:n_keep]
    
    # Construct the state matrix from the observability matrix
    # Using shifts in the SVD basis
    O = U @ np.diag(np.sqrt(s))  # Observability matrix
    
    # Extract state matrix via shift relationship
    # A = O^+ @ O_shift (pseudoinverse approach)
    O_shift = O[n_sensors:, :]  # Shifted observability matrix
    O_current = O[:-n_sensors, :]  # Current observability matrix
    
    A = np.linalg.pinv(O_current) @ O_shift
    
    # Step 5: Extract natural frequencies and damping from eigenvalues of A
    eigenvalues = np.linalg.eigvals(A)
    
    # Convert eigenvalues to natural frequencies and damping ratios
    # Assuming continuous-time model
    frequencies = []
    damping_ratios = []
    mode_shapes_list = []
    
    for eig_val in eigenvalues:
        if eig_val.imag >= 0:  # Only retain upper half-plane
            # Natural frequency (rad/s) -> Hz
            wn = np.abs(eig_val)
            fn = wn / (2 * np.pi)
            frequencies.append(fn)
            
            # Damping ratio
            zeta = -eig_val.real / np.abs(eig_val)
            damping_ratios.append(zeta)
    
    frequencies = np.array(frequencies)
    damping_ratios = np.array(damping_ratios)
    
    # Step 6: Extract mode shapes from the observation of A
    # Mode shapes are the output of the system at each mode
    n_frequencies = len(frequencies)
    mode_shapes = np.zeros((n_sensors, n_frequencies), dtype=complex)
    
    # Get eigenvalues and eigenvectors of A
    eigenvalues_A, eigenvectors_A = eig(A)
    
    col_idx = 0
    for i, eig_val in enumerate(eigenvalues_A):
        if eig_val.imag >= 0:
            # Extract the mode shape using the output matrix C
            # C is the first n_sensors rows of the observation matrix
            C_matrix = U[:n_sensors, :]
            # Compute mode shape as C @ eigenvector
            mode_shape = C_matrix @ eigenvectors_A[:, i]
            mode_shapes[:, col_idx] = mode_shape
            col_idx += 1
    
    # Normalize mode shapes to unit norm
    for i in range(mode_shapes.shape[1]):
        mode_shapes[:, i] /= (np.linalg.norm(mode_shapes[:, i]) + 1e-10)
    
    # Return real parts of mode shapes and frequencies
    return frequencies, damping_ratios, np.real(mode_shapes)


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
