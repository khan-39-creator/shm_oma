"""
shm_oma: Automated SSI-COV and Frequency Tracking for Structural Health Monitoring.

This library provides tools for modal parameter identification using Stochastic
Subspace Identification with Covariance (SSI-COV) and automated frequency tracking
over time using clustering and Modal Assurance Criterion (MAC).

Main Components:
- ssicov: Core SSI-COV algorithm implementation
- automation: Automated pole filtering and clustering
- tracking: Frequency tracking and time-series analysis
"""

__version__ = "0.1.0"
__author__ = "Adil Poshad khan"

# Import main public API
from .ssicov import (
    perform_ssi_cov,
    compute_autocorrelation,
    build_toeplitz_matrix,
    extract_stable_poles,
)

from .automation import (
    automate_poles_dbscan,
    automate_poles_hierarchical,
    filter_spurious_poles,
    mac,
    vectorized_mac,
)

from .tracking import (
    FrequencyTracker,
    ModalTrackingAnalyzer,
)

# Define public API
__all__ = [
    # SSI-COV functions
    "perform_ssi_cov",
    "compute_autocorrelation",
    "build_toeplitz_matrix",
    "extract_stable_poles",
    
    # Automation functions
    "automate_poles_dbscan",
    "automate_poles_hierarchical",
    "filter_spurious_poles",
    "mac",
    "vectorized_mac",
    
    # Tracking classes
    "FrequencyTracker",
    "ModalTrackingAnalyzer",
]
