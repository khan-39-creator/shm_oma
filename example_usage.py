"""
Example script demonstrating the full workflow of shm_oma library.

This script shows:
1. Basic SSI-COV modal parameter identification
2. Automated pole filtering and clustering
3. Continuous frequency tracking over time
4. Statistical analysis of tracked modes
"""

import numpy as np
from datetime import datetime, timedelta
from shm_oma import (
    perform_ssi_cov,
    automate_poles_dbscan,
    filter_spurious_poles,
    FrequencyTracker,
    ModalTrackingAnalyzer,
)


def example_1_basic_ssi_cov():
    """Example 1: Basic SSI-COV Analysis"""
    print("\n" + "="*70)
    print("Example 1: Basic SSI-COV Modal Parameter Identification")
    print("="*70)
    
    # Generate synthetic acceleration data (typically from sensors)
    np.random.seed(42)
    n_samples = 5000
    n_sensors = 3
    
    # Create synthetic data with two dominant modes
    t = np.linspace(0, 100, n_samples)
    f1, f2 = 10.0, 25.0  # Natural frequencies in Hz
    d1, d2 = 0.05, 0.03  # Damping ratios
    
    # Create modal responses
    y1 = np.exp(-d1 * 2 * np.pi * f1 * t) * np.sin(2 * np.pi * f1 * t)
    y2 = np.exp(-d2 * 2 * np.pi * f2 * t) * np.sin(2 * np.pi * f2 * t)
    
    # Combine and add noise
    mode_shapes = np.array([[1.0, 0.5], [0.8, 0.7], [0.6, 0.9]])
    acceleration_data = (
        np.outer(y1, mode_shapes[:, 0]) + 
        np.outer(y2, mode_shapes[:, 1]) + 
        0.1 * np.random.randn(n_samples, n_sensors)
    )
    
    print(f"Data shape: {acceleration_data.shape} (samples, sensors)")
    
    # Perform SSI-COV with model order 30
    frequencies, damping_ratios, mode_shapes_out = perform_ssi_cov(
        acceleration_data,
        order=30,
        max_lag=500,
        use_randomized_svd=True
    )
    
    print(f"\nIdentified {len(frequencies)} poles:")
    print(f"{'Frequency (Hz)':<20} {'Damping Ratio':<20} {'Norm':<15}")
    print("-" * 55)
    for i, (f, d, mode) in enumerate(zip(frequencies[:10], damping_ratios[:10], mode_shapes_out.T[:10])):
        norm = np.linalg.norm(mode)
        print(f"{f:<20.4f} {d:<20.6f} {norm:<15.6f}")


def example_2_automated_pole_filtering():
    """Example 2: Automated Pole Filtering and Clustering"""
    print("\n" + "="*70)
    print("Example 2: Automated Pole Filtering and Clustering")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    acceleration_data = np.random.randn(5000, 3)
    
    # Extract poles across multiple model orders (stabilization diagram)
    all_freqs, all_damps, all_modes = [], [], []
    
    for order in [20, 25, 30, 35, 40]:
        f, d, m = perform_ssi_cov(acceleration_data, order=order, max_lag=500)
        all_freqs.append(f)
        all_damps.append(d)
        all_modes.append(m)
        print(f"Order {order}: {len(f)} poles identified")
    
    # Filter spurious poles by comparing across orders
    print("\nFiltering spurious poles...")
    stable_freqs, stable_damps, stable_modes = filter_spurious_poles(
        all_freqs, all_damps, all_modes,
        freq_tol=0.01,      # 1% frequency tolerance
        damp_tol=0.05,      # 5% damping tolerance
        mac_threshold=0.98
    )
    
    print(f"Stable poles after filtering: {len(stable_freqs)}")
    
    # Cluster the stable poles
    print("\nClustering stable poles...")
    unique_modes = automate_poles_dbscan(
        stable_freqs, stable_damps, stable_modes,
        eps_freq=0.1,
        min_samples=2
    )
    
    print(f"Physical modes identified: {len(unique_modes)}")
    print(f"\n{'Mode':<10} {'Frequency (Hz)':<20} {'Damping Ratio':<20}")
    print("-" * 50)
    for i, (freq, damp, mode) in enumerate(unique_modes):
        print(f"{i+1:<10} {freq:<20.4f} {damp:<20.6f}")


def example_3_continuous_frequency_tracking():
    """Example 3: Continuous Frequency Tracking Over Time"""
    print("\n" + "="*70)
    print("Example 3: Continuous Frequency Tracking Over Time")
    print("="*70)
    
    # Initialize frequency tracker
    tracker = FrequencyTracker(
        mac_threshold=0.85,
        freq_tol=0.10,      # 10% frequency tolerance
        damping_tol=0.10,
        adaptation_rate=0.05
    )
    
    # Simulate continuous monitoring over 10 days
    timestamp = datetime(2026, 3, 5, 0, 0, 0)
    np.random.seed(42)
    
    print("Processing 24 time windows (every hour)...")
    
    for window_idx in range(24):
        # Simulate changing frequencies (e.g., temperature effect)
        base_freqs = np.array([10.5, 25.3])
        temp_variation = 0.05 * np.sin(2 * np.pi * window_idx / 24)
        modified_freqs = base_freqs * (1 + temp_variation)
        
        # Generate synthetic response
        synthetic_data = np.random.randn(1000, 3)
        
        # Extract modes (in practice, would use actual SSI-COV results)
        damps = np.array([0.05, 0.03])
        modes = np.array([[1.0, 0.5], [0.8, 0.7], [0.6, 0.9]])
        
        # Update tracker
        tracker.update(timestamp, modified_freqs, damps, modes)
        
        timestamp += timedelta(hours=1)
    
    print(f"Tracking complete. Monitored {len(tracker.reference_modes)} modes")
    
    # Analyze results
    analyzer = ModalTrackingAnalyzer(tracker)
    
    print(f"\n{'Mode':<10} {'Mean Freq':<20} {'Std Dev':<20} {'Observations':<15}")
    print("-" * 65)
    
    for mode_id in [1, 2]:
        stats = analyzer.compute_frequency_statistics(mode_id)
        if stats:
            print(f"{mode_id:<10} {stats['mean_freq']:<20.4f} "
                  f"{stats['std_freq']:<20.6f} {stats['n_observations']:<15}")


def example_4_trend_analysis():
    """Example 4: Trend Analysis and Anomaly Detection"""
    print("\n" + "="*70)
    print("Example 4: Trend Analysis and Anomaly Detection")
    print("="*70)
    
    # Initialize and populate tracker
    tracker = FrequencyTracker()
    
    # Simulate degradation scenario
    timestamp = datetime(2026, 3, 5, 0, 0, 0)
    base_freq = 10.5
    degradation_rate = -0.002  # Small frequency decrease per window
    
    np.random.seed(42)
    
    print("Simulating structural degradation over 30 time windows...")
    
    for window_idx in range(30):
        # Simulate frequency degradation (e.g., stiffness loss)
        degraded_freq = base_freq + degradation_rate * window_idx
        modified_freqs = np.array([degraded_freq, 25.3])
        
        damps = np.array([0.05, 0.03])
        modes = np.array([[1.0, 0.5], [0.8, 0.7], [0.6, 0.9]])
        
        if window_idx == 0:
            tracker.initialize(timestamp, modified_freqs, damps, modes)
        else:
            tracker.update(timestamp, modified_freqs, damps, modes)
        
        timestamp += timedelta(hours=6)
    
    # Analyze MODE 1 for degradation
    analyzer = ModalTrackingAnalyzer(tracker)
    
    print("\nMode 1 Analysis:")
    print("-" * 50)
    
    # Get statistics
    stats = analyzer.compute_frequency_statistics(1)
    print(f"Mean frequency: {stats['mean_freq']:.4f} Hz")
    print(f"Frequency std dev: {stats['std_freq']:.6f} Hz")
    print(f"Frequency range: {stats['freq_range']:.6f} Hz")
    print(f"Number of observations: {stats['n_observations']}")
    
    # Compute trend (should show negative slope due to degradation)
    trend = analyzer.compute_trend(1)
    print(f"\nFrequency Trend:")
    print(f"Slope: {trend['slope']:.8f} Hz/sec (≈ {trend['slope']*3600:.6f} Hz/hour)")
    print(f"R²: {trend['r_squared']:.6f}")
    
    # Detect anomalies
    anomalies = analyzer.detect_frequency_anomalies(1, threshold=3.0)
    print(f"\nAnomalies detected: {len(anomalies)}")
    if anomalies:
        print(f"Anomalous indices: {anomalies}")


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  SHM_OMA - SSI-COV Example Demonstrations".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        # Run examples
        example_1_basic_ssi_cov()
        example_2_automated_pole_filtering()
        example_3_continuous_frequency_tracking()
        example_4_trend_analysis()
        
        print("\n" + "█"*70)
        print("█" + "  All examples completed successfully!".center(68) + "█")
        print("█"*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
