"""
Automation module for filtering spurious poles and clustering physical modes.

This module uses DBSCAN and agglomerative clustering to automatically 
identify stable modes from the stabilization diagram.
"""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform


def mac(phi1, phi2):
    """
    Calculate Modal Assurance Criterion between two mode shapes.
    
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
    denominator = (np.real(np.vdot(phi1, phi1)) * 
                   np.real(np.vdot(phi2, phi2)))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def vectorized_mac(mode_shapes):
    """
    Compute MAC matrix between all pairs of mode shapes efficiently.
    
    Parameters
    ----------
    mode_shapes : ndarray, shape (n_sensors, n_modes)
        Matrix where columns are mode shape vectors.
    
    Returns
    -------
    mac_matrix : ndarray, shape (n_modes, n_modes)
        MAC matrix with MAC values for all pairs.
    """
    n_sensors, n_modes = mode_shapes.shape
    
    # Compute norms for each mode
    norms = np.linalg.norm(mode_shapes, axis=0)
    
    # Compute cross-product matrix
    cross_product = np.abs(mode_shapes.T.conj() @ mode_shapes) ** 2
    
    # Normalize by outer product of norms
    mac_matrix = cross_product / (norms[:, np.newaxis] * norms[np.newaxis, :] + 1e-12)
    
    return mac_matrix


def automate_poles_dbscan(frequencies, damping_ratios, mode_shapes, 
                          eps_freq=0.05, min_samples=3):
    """
    Automate pole selection using DBSCAN clustering on stable frequencies.
    
    Parameters
    ----------
    frequencies : ndarray, shape (n_poles,)
        Frequencies from SSI-COV analysis (possibly multiple model orders).
    damping_ratios : ndarray, shape (n_poles,)
        Damping ratios corresponding to frequencies.
    mode_shapes : ndarray, shape (n_sensors, n_poles)
        Mode shapes as column vectors.
    eps_freq : float, default=0.05
        Maximum frequency distance for DBSCAN neighborhood (in Hz).
    min_samples : int, default=3
        Minimum samples per cluster.
    
    Returns
    -------
    unique_modes : list of tuples
        List of (frequency, damping_ratio, mode_shape) for each identified mode.
    """
    if len(frequencies) == 0:
        return []
    
    # Reshape for sklearn
    X = frequencies.reshape(-1, 1)
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps_freq, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    
    unique_modes = []
    
    for label in set(labels):
        if label == -1:
            # Skip noise points
            continue
        
        # Extract cluster indices
        idx = np.where(labels == label)[0]
        
        # Average frequency for the cluster
        mean_freq = np.mean(frequencies[idx])
        
        # Average damping
        mean_damp = np.mean(damping_ratios[idx])
        
        # For mode shapes, compute the cluster representative
        # Option: Use the mode shape with highest energy or average them
        cluster_shapes = mode_shapes[:, idx]
        
        if cluster_shapes.shape[1] == 1:
            rep_mode_shape = cluster_shapes[:, 0]
        else:
            # Average mode shapes (with phase alignment)
            rep_mode_shape = np.mean(cluster_shapes, axis=1)
            rep_mode_shape /= np.linalg.norm(rep_mode_shape)
        
        unique_modes.append((mean_freq, mean_damp, rep_mode_shape))
    
    return unique_modes


def automate_poles_hierarchical(frequencies, damping_ratios, mode_shapes,
                                 distance_threshold=0.1, linkage='ward'):
    """
    Automate pole selection using Agglomerative (Hierarchical) Clustering.
    
    Parameters
    ----------
    frequencies : ndarray, shape (n_poles,)
        Frequencies from SSI-COV analysis.
    damping_ratios : ndarray, shape (n_poles,)
        Damping ratios corresponding to frequencies.
    mode_shapes : ndarray, shape (n_sensors, n_poles)
        Mode shapes as column vectors.
    distance_threshold : float, default=0.1
        Distance threshold for cluster merging.
    linkage : str, default='ward'
        Linkage criterion ('ward', 'complete', 'average', 'single').
    
    Returns
    -------
    unique_modes : list of tuples
        List of (frequency, damping_ratio, mode_shape) for each identified mode.
    """
    if len(frequencies) == 0:
        return []
    
    if len(frequencies) == 1:
        return [(frequencies[0], damping_ratios[0], mode_shapes[:, 0])]
    
    # Build feature matrix: [normalized_freq, normalized_damping, MAC features]
    features = np.column_stack([
        frequencies / (np.max(frequencies) + 1e-8),
        damping_ratios / (np.max(damping_ratios) + 1e-8),
    ])
    
    # Apply hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=linkage
    )
    labels = clustering.fit_predict(features)
    
    unique_modes = []
    
    for label in set(labels):
        idx = np.where(labels == label)[0]
        
        mean_freq = np.mean(frequencies[idx])
        mean_damp = np.mean(damping_ratios[idx])
        
        cluster_shapes = mode_shapes[:, idx]
        
        if cluster_shapes.shape[1] == 1:
            rep_mode_shape = cluster_shapes[:, 0]
        else:
            rep_mode_shape = np.mean(cluster_shapes, axis=1)
            rep_mode_shape /= np.linalg.norm(rep_mode_shape)
        
        unique_modes.append((mean_freq, mean_damp, rep_mode_shape))
    
    return unique_modes


def filter_spurious_poles(all_frequencies, all_damping, all_mode_shapes,
                          freq_tol=0.01, damp_tol=0.05, mac_threshold=0.85):
    """
    Filter spurious poles by comparing across model orders.
    
    A pole is considered "stable" if it appears consistently across orders.
    
    Parameters
    ----------
    all_frequencies : list of ndarray
        Frequencies for each model order.
    all_damping : list of ndarray
        Damping ratios for each model order.
    all_mode_shapes : list of ndarray
        Mode shapes for each model order.
    freq_tol : float, default=0.01
        Relative frequency tolerance.
    damp_tol : float, default=0.05
        Absolute damping tolerance.
    mac_threshold : float, default=0.85
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
    if len(all_frequencies) < 2:
        return all_frequencies[0], all_damping[0], all_mode_shapes[0]
    
    # Start with the first model order
    reference_freqs = all_frequencies[0]
    reference_damps = all_damping[0]
    reference_modes = all_mode_shapes[0]
    
    stable_mask = np.zeros(len(reference_freqs), dtype=bool)
    
    # Check stability against the next model order
    next_freqs = all_frequencies[1]
    next_damps = all_damping[1]
    next_modes = all_mode_shapes[1]
    
    for i, (ref_f, ref_d, ref_mode) in enumerate(
        zip(reference_freqs, reference_damps, reference_modes.T)):
        
        for j, (next_f, next_d, next_mode) in enumerate(
            zip(next_freqs, next_damps, next_modes.T)):
            
            # Check frequency stability
            freq_diff = np.abs(ref_f - next_f) / (ref_f + 1e-8)
            
            # Check damping stability
            damp_diff = np.abs(ref_d - next_d)
            
            # Check MAC
            mac_val = mac(ref_mode, next_mode)
            
            # If all criteria satisfied, mark as stable
            if (freq_diff < freq_tol and 
                damp_diff < damp_tol and 
                mac_val > mac_threshold):
                stable_mask[i] = True
                break
    
    stable_freqs = reference_freqs[stable_mask]
    stable_damps = reference_damps[stable_mask]
    stable_modes = reference_modes[:, stable_mask]
    
    return stable_freqs, stable_damps, stable_modes
