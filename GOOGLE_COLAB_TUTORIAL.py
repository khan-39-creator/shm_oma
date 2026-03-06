"""
GOOGLE COLAB SETUP FOR SHM_OMA

Copy and paste this entire script into a Google Colab cell
and run it. It will:
1. Install the library
2. Test with sample data
3. Show how to upload your own data
"""

# ============================================================================
# CELL 1: Install shm_oma library (run this first)
# ============================================================================

!pip install numpy scipy scikit-learn

# Clone the shm_oma library
!git clone https://github.com/yourusername/shm_oma.git /content/shm_oma
!cd /content/shm_oma && pip install -e .

print("✅ Installation complete!")

# ============================================================================
# CELL 2: Import libraries and test
# ============================================================================

import numpy as np
from datetime import datetime, timedelta
from shm_oma import perform_ssi_cov, FrequencyTracker, ModalTrackingAnalyzer

print("✅ Imports successful!")

# ============================================================================
# CELL 3: Run analysis on sample data
# ============================================================================

# Generate synthetic data
print("\n📊 Generating sample data...")
np.random.seed(42)
data = np.random.randn(5000, 3)  # 5000 samples, 3 sensors

# Run SSI-COV
print("🔍 Running SSI-COV analysis...")
frequencies, damping, modes = perform_ssi_cov(data, order=30)

# Display results
print(f"\n✓ Found {len(frequencies)} modes\n")
print(f"{'#':<5} {'Frequency (Hz)':<20} {'Damping':<15}")
print("-" * 40)
for i, (f, d) in enumerate(zip(frequencies[:5], damping[:5])):
    print(f"{i+1:<5} {f:<20.4f} {d:<15.6f}")

# ============================================================================
# CELL 4: Upload your own CSV file
# ============================================================================

from google.colab import files
import pandas as pd

print("\n📂 Upload your CSV file:")
print("   Format: (rows should be samples, columns should be sensors)")
print("   Example: accel_x, accel_y, accel_z")
print()

uploaded = files.upload()

for filename in uploaded.keys():
    print(f"✓ Uploaded: {filename}")
    
    # Load the data
    df = pd.read_csv(filename)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Extract sensor data (numeric columns)
    data = df.select_dtypes(include=[np.number]).values
    
    # Run analysis
    print(f"\n🔍 Analyzing your data...")
    frequencies, damping, modes = perform_ssi_cov(data, order=25)
    
    print(f"✓ Found {len(frequencies)} modes\n")
    print(f"{'#':<5} {'Frequency (Hz)':<20} {'Damping':<15}")
    print("-" * 40)
    for i, (f, d) in enumerate(zip(frequencies[:10], damping[:10])):
        print(f"{i+1:<5} {f:<20.4f} {d:<15.6f}")

# ============================================================================
# CELL 5: Continuous tracking example
# ============================================================================

print("\n⏱️ Simulating continuous frequency tracking...")

tracker = FrequencyTracker()
timestamp = datetime.now()

# Simulate 10 time windows
for window in range(10):
    # In practice: load each time window from your data
    window_data = np.random.randn(2000, 3)
    
    freqs, damps, modes = perform_ssi_cov(window_data, order=20)
    tracker.update(timestamp, freqs, damps, modes)
    
    timestamp += timedelta(hours=1)

# Analyze
analyzer = ModalTrackingAnalyzer(tracker)

print(f"\n{'Mode':<8} {'Mean Freq':<18} {'Std Dev':<15}")
print("-" * 40)

for mode_id in range(1, len(tracker.reference_modes) + 1):
    stats = analyzer.compute_frequency_statistics(mode_id)
    if stats:
        print(f"{mode_id:<8} {stats['mean_freq']:<18.4f} {stats['std_freq']:<15.6f}")

# ============================================================================
# CELL 6: Advanced - Multiple model orders with filtering
# ============================================================================

from shm_oma import filter_spurious_poles, automate_poles_dbscan

print("\n🔍 Running advanced analysis with pole filtering...")

# Generate data
data = np.random.randn(5000, 3)

# Extract at multiple orders
all_freqs, all_damps, all_modes = [], [], []
print("\nExtracting poles at orders: [25, 30, 35, 40]")

for order in [25, 30, 35, 40]:
    f, d, m = perform_ssi_cov(data, order=order)
    all_freqs.append(f)
    all_damps.append(d)
    all_modes.append(m)
    print(f"  Order {order}: {len(f)} poles")

# Filter spurious poles
print("\nFiltering spurious poles...")
stable_f, stable_d, stable_m = filter_spurious_poles(
    all_freqs, all_damps, all_modes,
    freq_tol=0.01,
    damp_tol=0.05,
    mac_threshold=0.98
)

print(f"✓ {len(stable_f)} stable poles identified")

# Cluster physical modes
if len(stable_f) > 0:
    print("\nClustering physical modes...")
    physical_modes = automate_poles_dbscan(
        stable_f, stable_d, stable_m,
        eps_freq=0.1
    )
    
    print(f"✓ {len(physical_modes)} physical modes identified\n")
    print(f"{'#':<5} {'Frequency (Hz)':<20} {'Damping':<15}")
    print("-" * 40)
    
    for i, (freq, damp, mode) in enumerate(physical_modes):
        print(f"{i+1:<5} {freq:<20.4f} {damp:<15.6f}")

# ============================================================================
# CELL 7: Download results
# ============================================================================

from google.colab import files

# Save results to file
results_text = f"""
ANALYSIS RESULTS
================

Frequencies identified: {len(frequencies)}

Top 5 modes:
"""

for i, (f, d) in enumerate(zip(frequencies[:5], damping[:5])):
    results_text += f"\nMode {i+1}: {f:.4f} Hz, Damping: {d:.6f}"

# Save to file
with open('shm_oma_results.txt', 'w') as f:
    f.write(results_text)

print("✓ Results saved to shm_oma_results.txt")
print("\nDownloading results...")
files.download('shm_oma_results.txt')
