"""
shm_oma: Automated SSI-COV and Frequency Tracking for Structural Health Monitoring.

This library provides tools for modal parameter identification using Stochastic
Subspace Identification with Covariance (SSI-COV) and automated frequency tracking
over time using clustering and Modal Assurance Criterion (MAC).

Main Components
---------------
ssicov      Core SSI-COV algorithm with Hard Criteria filtering.
automation  Automated pole filtering and clustering (DBSCAN / hierarchical).
tracking    Frequency tracking and time-series analysis for continuous SHM.
"""

__version__ = "0.2.0"
__author__ = "Adil Poshad Khan"

# Core SSI-COV
from .ssicov import (
    perform_ssi_cov,
    compute_autocorrelation,
    build_toeplitz_matrix,
    extract_stable_poles,
    compute_psd,
    plot_stabilization_diagram,
    plot_psd_with_peaks,
    plot_singular_values,
)

# Automation & clustering
from .automation import (
    automate_poles_dbscan,
    automate_poles_hierarchical,
    filter_spurious_poles,
    mac,
    vectorized_mac,
)

# Tracking
from .tracking import (
    FrequencyTracker,
    ModalTrackingAnalyzer,
)

__all__ = [
    # SSI-COV
    "perform_ssi_cov",
    "compute_autocorrelation",
    "build_toeplitz_matrix",
    "extract_stable_poles",
    "compute_psd",
    "plot_stabilization_diagram",
    "plot_psd_with_peaks",
    "plot_singular_values",
    # Automation
    "automate_poles_dbscan",
    "automate_poles_hierarchical",
    "filter_spurious_poles",
    "mac",
    "vectorized_mac",
    # Tracking
    "FrequencyTracker",
    "ModalTrackingAnalyzer",
]
