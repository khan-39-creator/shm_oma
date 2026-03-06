"""
Automation module for filtering spurious poles and clustering physical modes.

Uses DBSCAN and agglomerative clustering to automatically identify stable
modes from a stabilisation diagram.  Implements the Modal Assurance Criterion
(MAC) with correct handling of complex-valued mode shapes.

References
----------
[1] Reynders, E., Houbrechts, J., & De Roeck, G. (2012). Fully automated
    (operational) modal analysis. MSSP, 29, 228-250.
[2] Magalhaes, F., Cunha, A., & Caetano, E. (2009). Online automatic
    identification of modal parameters. MSSP, 23(2), 316-329.
"""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering


# ---------------------------------------------------------------------------
# Modal Assurance Criterion (MAC)
# ---------------------------------------------------------------------------

def mac(phi1, phi2):
    """
    Calculate Modal Assurance Criterion between two mode shapes.

    Handles both real and complex mode shapes correctly by using
    the Hermitian inner product (np.vdot).

    Parameters
    ----------
    phi1 : ndarray, shape (n_sensors,)
        First mode shape vector.
    phi2 : ndarray, shape (n_sensors,)
        Second mode shape vector.

    Returns
    -------
    mac_value : float
        MAC value between 0 and 1 (1 = perfect correlation).
    """
    numerator = np.abs(np.vdot(phi1, phi2)) ** 2
    denominator = np.real(np.vdot(phi1, phi1)) * np.real(np.vdot(phi2, phi2))

    if denominator < 1e-30:
        return 0.0

    return float(numerator / denominator)


def vectorized_mac(mode_shapes):
    """
    Compute MAC matrix between all pairs of mode shapes efficiently.

    Uses the Hermitian dot product so it is correct for both real and
    complex mode shapes.

    Parameters
    ----------
    mode_shapes : ndarray, shape (n_sensors, n_modes)
        Matrix where columns are mode shape vectors.

    Returns
    -------
    mac_matrix : ndarray, shape (n_modes, n_modes)
        MAC matrix with MAC values for all pairs.
    """
    # Hermitian cross-product matrix  |phi_i^H @ phi_j|^2
    cross_product = np.abs(mode_shapes.conj().T @ mode_shapes) ** 2

    # Self-energies  (phi_i^H @ phi_i) — always real
    self_energy = np.real(np.einsum("ij,ij->j", mode_shapes.conj(), mode_shapes))

    # Normalise
    outer = np.outer(self_energy, self_energy)
    mac_matrix = cross_product / (outer + 1e-30)

    return mac_matrix


# ---------------------------------------------------------------------------
# DBSCAN clustering
# ---------------------------------------------------------------------------

def automate_poles_dbscan(
    frequencies,
    damping_ratios,
    mode_shapes,
    eps_freq=0.05,
    min_samples=3,
):
    """
    Automate pole selection using DBSCAN clustering on stable frequencies.

    Parameters
    ----------
    frequencies : ndarray, shape (n_poles,)
    damping_ratios : ndarray, shape (n_poles,)
    mode_shapes : ndarray, shape (n_sensors, n_poles)
    eps_freq : float, default 0.05
        Maximum frequency distance for DBSCAN neighbourhood (Hz).
    min_samples : int, default 3
        Minimum samples per cluster.

    Returns
    -------
    unique_modes : list of tuples
        ``(frequency, damping_ratio, mode_shape)`` for each identified mode.
    """
    if len(frequencies) == 0:
        return []

    X = frequencies.reshape(-1, 1)
    clustering = DBSCAN(eps=eps_freq, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    unique_modes = []

    for label in set(labels):
        if label == -1:
            continue  # noise

        idx = np.where(labels == label)[0]
        mean_freq = float(np.mean(frequencies[idx]))
        mean_damp = float(np.mean(damping_ratios[idx]))

        cluster_shapes = mode_shapes[:, idx]

        if cluster_shapes.shape[1] == 1:
            rep_mode_shape = cluster_shapes[:, 0]
        else:
            # Phase-aligned average of mode shapes
            ref = cluster_shapes[:, 0]
            aligned = np.zeros_like(cluster_shapes)
            for k in range(cluster_shapes.shape[1]):
                # Align sign by checking MAC inner product sign
                dot = np.real(np.vdot(ref, cluster_shapes[:, k]))
                aligned[:, k] = cluster_shapes[:, k] * np.sign(dot + 1e-30)
            rep_mode_shape = np.mean(aligned, axis=1)
            norm = np.linalg.norm(rep_mode_shape)
            if norm > 1e-10:
                rep_mode_shape /= norm

        unique_modes.append((mean_freq, mean_damp, rep_mode_shape))

    # Sort by frequency
    unique_modes.sort(key=lambda m: m[0])
    return unique_modes


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------

def automate_poles_hierarchical(
    frequencies,
    damping_ratios,
    mode_shapes,
    distance_threshold=0.1,
    linkage="ward",
):
    """
    Automate pole selection using Agglomerative (Hierarchical) Clustering.

    Parameters
    ----------
    frequencies : ndarray, shape (n_poles,)
    damping_ratios : ndarray, shape (n_poles,)
    mode_shapes : ndarray, shape (n_sensors, n_poles)
    distance_threshold : float, default 0.1
    linkage : str, default 'ward'
        Linkage criterion ('ward', 'complete', 'average', 'single').

    Returns
    -------
    unique_modes : list of tuples
        ``(frequency, damping_ratio, mode_shape)`` for each identified mode.
    """
    if len(frequencies) == 0:
        return []
    if len(frequencies) == 1:
        return [(float(frequencies[0]), float(damping_ratios[0]), mode_shapes[:, 0])]

    # Normalised feature matrix
    freq_norm = frequencies / (np.max(np.abs(frequencies)) + 1e-12)
    damp_norm = damping_ratios / (np.max(np.abs(damping_ratios)) + 1e-12)
    features = np.column_stack([freq_norm, damp_norm])

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=linkage,
    )
    labels = clustering.fit_predict(features)

    unique_modes = []

    for label in set(labels):
        idx = np.where(labels == label)[0]
        mean_freq = float(np.mean(frequencies[idx]))
        mean_damp = float(np.mean(damping_ratios[idx]))

        cluster_shapes = mode_shapes[:, idx]

        if cluster_shapes.shape[1] == 1:
            rep_mode_shape = cluster_shapes[:, 0]
        else:
            ref = cluster_shapes[:, 0]
            aligned = np.zeros_like(cluster_shapes)
            for k in range(cluster_shapes.shape[1]):
                dot = np.real(np.vdot(ref, cluster_shapes[:, k]))
                aligned[:, k] = cluster_shapes[:, k] * np.sign(dot + 1e-30)
            rep_mode_shape = np.mean(aligned, axis=1)
            norm = np.linalg.norm(rep_mode_shape)
            if norm > 1e-10:
                rep_mode_shape /= norm

        unique_modes.append((mean_freq, mean_damp, rep_mode_shape))

    unique_modes.sort(key=lambda m: m[0])
    return unique_modes


# ---------------------------------------------------------------------------
# Stability filtering across multiple model orders
# ---------------------------------------------------------------------------

def filter_spurious_poles(
    all_frequencies,
    all_damping,
    all_mode_shapes,
    freq_tol=0.01,
    damp_tol=0.05,
    mac_threshold=0.85,
):
    """
    Filter spurious poles by comparing across ALL consecutive model orders.

    A pole is considered "stable" if it appears consistently across **every**
    consecutive pair of model orders (not just the first two).

    Parameters
    ----------
    all_frequencies : list of ndarray
        Frequencies for each model order.
    all_damping : list of ndarray
        Damping ratios for each model order.
    all_mode_shapes : list of ndarray
        Mode shapes for each model order.
    freq_tol : float, default 0.01
        Relative frequency tolerance.
    damp_tol : float, default 0.05
        Absolute damping tolerance.
    mac_threshold : float, default 0.85
        Modal Assurance Criterion threshold.

    Returns
    -------
    stable_freqs : ndarray
    stable_damps : ndarray
    stable_modes : ndarray, shape (n_sensors, n_stable)
    """
    if len(all_frequencies) < 2:
        return all_frequencies[0], all_damping[0], all_mode_shapes[0]

    # Reference is the first order
    ref_freqs = all_frequencies[0]
    ref_damps = all_damping[0]
    ref_modes = all_mode_shapes[0]

    stable_mask = np.ones(len(ref_freqs), dtype=bool)

    # Check stability against EVERY subsequent order
    for k in range(1, len(all_frequencies)):
        next_freqs = all_frequencies[k]
        next_damps = all_damping[k]
        next_modes = all_mode_shapes[k]

        pair_mask = np.zeros(len(ref_freqs), dtype=bool)

        for i in range(len(ref_freqs)):
            if not stable_mask[i]:
                continue

            ref_f = ref_freqs[i]
            ref_d = ref_damps[i]
            ref_m = ref_modes[:, i]

            for j in range(len(next_freqs)):
                freq_diff = np.abs(ref_f - next_freqs[j]) / (ref_f + 1e-12)
                damp_diff = np.abs(ref_d - next_damps[j])
                mac_val = mac(ref_m, next_modes[:, j])

                if (
                    freq_diff < freq_tol
                    and damp_diff < damp_tol
                    and mac_val > mac_threshold
                ):
                    pair_mask[i] = True
                    break

        stable_mask &= pair_mask

    stable_freqs = ref_freqs[stable_mask]
    stable_damps = ref_damps[stable_mask]
    stable_modes = ref_modes[:, stable_mask]

    return stable_freqs, stable_damps, stable_modes
