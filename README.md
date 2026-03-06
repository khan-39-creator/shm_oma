# shm_oma: Automated SSI-COV for Structural Health Monitoring

A professional Python library for automated modal parameter identification using **Stochastic Subspace Identification with Covariance (SSI-COV)** and **frequency tracking** over time. Designed for continuous Structural Health Monitoring (SHM) systems.

## Features

- **Automated SSI-COV Algorithm**: Core implementation of the SSI-COV algorithm for modal parameter extraction from ambient vibration data
- **Randomized SVD**: High-performance computation using scikit-learn's randomized SVD for large-scale problems
- **Automated Pole Filtering**: Removes spurious mathematical poles using stability criteria:
  - Frequency stability (< 1% relative difference)
  - Damping stability (< 5% absolute difference)
  - Modal Assurance Criterion (MAC > 0.98)
- **Clustering Methods**: 
  - DBSCAN for automatic mode grouping
  - Agglomerative (hierarchical) clustering with configurable linkage
- **Frequency Tracking**: Continuous monitoring of modal parameters over time with automatic mode matching
- **SHM Analysis Tools**: Statistical trend analysis and anomaly detection

## Installation

### Prerequisites
- Python ≥ 3.9
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- scikit-learn ≥ 1.0

### From Source (Development Mode)

```bash
# Clone the repository
git clone <repository-url>
cd shm_oma

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e .[dev]
```

### Test Installation

```bash
pytest tests/
```

## Quick Start

### Example 1: Basic SSI-COV Analysis

```python
import numpy as np
from shm_oma import perform_ssi_cov

# Load or generate acceleration data
# Shape: (n_samples, n_sensors)
acceleration_data = np.random.randn(5000, 3)  # 5000 samples, 3 sensors

# Perform SSI-COV with model order 50
frequencies, damping_ratios, mode_shapes = perform_ssi_cov(
    acceleration_data,
    order=50,
    max_lag=500,
    use_randomized_svd=True  # For faster computation on large data
)

print(f"Identified {len(frequencies)} modes")
print(f"Frequencies (Hz): {frequencies}")
print(f"Damping ratios: {damping_ratios}")
print(f"Mode shapes shape: {mode_shapes.shape}")
```

### Example 2: Automated Pole Filtering and Clustering

```python
from shm_oma import perform_ssi_cov, automate_poles_dbscan, filter_spurious_poles

# Extract modal parameters across multiple model orders
orders = [20, 25, 30, 35, 40]
all_freqs, all_damps, all_modes = [], [], []

for order in orders:
    f, d, m = perform_ssi_cov(acceleration_data, order=order)
    all_freqs.append(f)
    all_damps.append(d)
    all_modes.append(m)

# Filter spurious poles
stable_freqs, stable_damps, stable_modes = filter_spurious_poles(
    all_freqs, all_damps, all_modes,
    freq_tol=0.01,      # 1% frequency tolerance
    damp_tol=0.05,      # 5% damping tolerance
    mac_threshold=0.98  # MAC > 0.98
)

# Cluster stable poles
unique_modes = automate_poles_dbscan(
    stable_freqs, stable_damps, stable_modes,
    eps_freq=0.05,      # Max frequency distance for clustering
    min_samples=3       # Minimum cluster size
)

print(f"Identified {len(unique_modes)} physical modes")
for i, (freq, damp, mode_shape) in enumerate(unique_modes):
    print(f"Mode {i+1}: {freq:.2f} Hz, damping: {damp:.4f}")
```

### Example 3: Frequency Tracking Over Time

```python
from shm_oma import FrequencyTracker, ModalTrackingAnalyzer
from datetime import datetime, timedelta

# Initialize tracker
tracker = FrequencyTracker(
    mac_threshold=0.85,
    freq_tol=0.10,      # 10% relative frequency tolerance
    damping_tol=0.10,
    adaptation_rate=0.05  # Slow adaptation to baseline
)

# Process continuous data windows (e.g., every 30 minutes)
timestamp = datetime.now()

for window_idx in range(10):
    # Get 30-minute window of acceleration data
    window_data = load_30min_window(window_idx)
    
    # Extract modes from this window
    freqs, damps, modes = perform_ssi_cov(window_data, order=30)
    
    # Update tracker
    tracker.update(timestamp, freqs, damps, modes)
    
    # Move to next window
    timestamp += timedelta(minutes=30)

# Analyze results
analyzer = ModalTrackingAnalyzer(tracker)

for mode_id in [1, 2, 3]:
    # Get frequency trend
    timestamps, frequencies = tracker.get_frequency_trends(mode_id)
    
    # Compute statistics
    stats = analyzer.compute_frequency_statistics(mode_id)
    print(f"Mode {mode_id}: Mean={stats['mean_freq']:.2f} Hz, "
          f"Std={stats['std_freq']:.4f} Hz")
    
    # Detect anomalies
    anomalies = analyzer.detect_frequency_anomalies(mode_id, threshold=3.0)
    if anomalies:
        print(f"  Anomalies detected at indices: {anomalies}")
    
    # Compute trend
    trend = analyzer.compute_trend(mode_id)
    print(f"  Frequency trend: {trend['slope']:.6f} Hz/sec (R²={trend['r_squared']:.4f})")
```

## Module Architecture

### `ssicov.py` - Core Algorithm
- `perform_ssi_cov()`: Main SSI-COV implementation
- `compute_autocorrelation()`: Autocorrelation function computation
- `build_toeplitz_matrix()`: Block-Toeplitz matrix assembly
- `extract_stable_poles()`: Filter poles across model orders

### `automation.py` - Clustering & Automation
- `mac()`: Modal Assurance Criterion calculation
- `vectorized_mac()`: Fast batch MAC computation
- `automate_poles_dbscan()`: DBSCAN-based pole grouping
- `automate_poles_hierarchical()`: Hierarchical clustering for poles
- `filter_spurious_poles()`: Hard thresholding based stability filtering

### `tracking.py` - Frequency Tracking
- `FrequencyTracker`: Main tracking class for continuous monitoring
  - `initialize()`: Set baseline modes
  - `update()`: Process new time window
  - `get_history()`: Retrieve mode time series
  - `export_to_dict()`: Serialize tracking data
- `ModalTrackingAnalyzer`: Analysis tools
  - `compute_frequency_statistics()`: Mean, std, min, max
  - `detect_frequency_anomalies()`: Z-score based anomaly detection
  - `compute_trend()`: Linear trend analysis

## Performance Optimization Tips

1. **Randomized SVD**: Always use `use_randomized_svd=True` for large datasets
   ```python
   frequencies, damping, modes = perform_ssi_cov(
       large_data, order=50, use_randomized_svd=True
   )
   ```

2. **Vectorized MAC**: For multiple comparisons, use `vectorized_mac()`
   ```python
   mac_matrix = vectorized_mac(mode_shapes)  # Much faster than nested loops
   ```

3. **Max Lag Selection**: Use 1/2 of data length for optimal autocorrelation
   ```python
   # For 10,000 samples: max_lag ~ 5,000
   perform_ssi_cov(data, order=30, max_lag=len(data)//2)
   ```

## Testing

Run the full test suite:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_ssicov.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=shm_oma
```

## Project Structure

```
shm_oma/
├── pyproject.toml              # Project configuration
├── README.md                   # This file
├── .gitignore
├── src/
│   └── shm_oma/
│       ├── __init__.py         # Public API
│       ├── ssicov.py           # Core SSI-COV
│       ├── automation.py       # Clustering & automation
│       └── tracking.py         # Frequency tracking
└── tests/
    ├── __init__.py
    ├── test_ssicov.py
    └── test_tracking.py
```

## Citation

If you use this library in your research, please cite:
```
@software{shm_oma2026,
  title={shm_oma: Automated SSI-COV for Structural Health Monitoring},
  author={Khan, Adil Poshad},
  year={2026},
  url={https://github.com/yourusername/shm_oma}
}
```

## References

1. **Reynders, E., Pintelon, R., & De Roeck, G.** (2008). 
   "Uncertainty bounds on modal parameters obtained from stochastic subspace identification."
   *Mechanical Systems and Signal Processing*, 22(4), 948-969.

2. **Peeters, B., & De Roeck, G.** (2001).
   "Stochastic system identification for operational modal analysis: A review."
   *Journal of Dynamic Systems, Measurement, and Control*, 123(4), 659-667.

3. **Magalhaes, F., Cunha, A., & Caetano, E.** (2008).
   "Online automatic identification of the modal parameters of a long span arch bridge."
   *Mechanical Systems and Signal Processing*, 23(2), 316-329.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or pull request with your improvements.

## Contact

For questions or support, contact: adil.poshad@example.com
