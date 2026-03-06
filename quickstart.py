"""
Practical guide: How to run shm_oma on real data

This script demonstrates loading data and performing modal analysis.
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


def generate_sensor_data(duration_sec=60, sampling_rate=100, n_sensors=3):
    """
    Generate synthetic acceleration data from a 3-DOF structure.
    
    Parameters:
    - duration_sec: Total duration in seconds
    - sampling_rate: Sampling frequency in Hz
    - n_sensors: Number of acceleration sensors
    
    Returns:
    - data: ndarray of shape (n_samples, n_sensors)
    - t: Time array
    """
    n_samples = duration_sec * sampling_rate
    t = np.linspace(0, duration_sec, n_samples)
    
    # Define 3 structural modes
    modes = {
        'mode1': {'freq': 2.5, 'damp': 0.03, 'shape': [1.0, 0.8, 0.5]},
        'mode2': {'freq': 7.2, 'damp': 0.02, 'shape': [1.0, -0.5, 0.8]},
        'mode3': {'freq': 12.1, 'damp': 0.04, 'shape': [1.0, 0.3, -0.9]},
    }
    
    # Build response
    data = np.zeros((n_samples, n_sensors))
    
    for mode_name, mode_info in modes.items():
        f = mode_info['freq']
        d = mode_info['damp']
        shape = np.array(mode_info['shape'])
        
        # Damped oscillation: y(t) = exp(-2*pi*zeta*f*t) * sin(2*pi*f*t)
        wn = 2 * np.pi * f
        response = np.exp(-2 * np.pi * d * f * t) * np.sin(wn * t)
        
        # Add to each sensor according to mode shape
        for sensor in range(n_sensors):
            data[:, sensor] += response * shape[sensor]
    
    # Add measurement noise
    data += 0.05 * np.random.randn(n_samples, n_sensors)
    
    return data, t


def quick_analysis():
    """Quick modal analysis on generated data"""
    print("\n" + "="*70)
    print("QUICK ANALYSIS ON GENERATED DATA")
    print("="*70)
    
    # Generate data
    print("\n1. Generating synthetic acceleration data...")
    data, t = generate_sensor_data(duration_sec=120, sampling_rate=100, n_sensors=3)
    print(f"   Data shape: {data.shape} (samples: {data.shape[0]}, sensors: {data.shape[1]})")
    print(f"   Duration: 120 seconds, Sampling rate: 100 Hz")
    
    # Run SSI-COV
    print("\n2. Running SSI-COV analysis with model order 20...")
    freqs, damps, mode_shapes = perform_ssi_cov(
        data, 
        order=20, 
        max_lag=500,
        use_randomized_svd=True
    )
    
    print(f"   Identified {len(freqs)} poles")
    
    # Show top 5 modes by frequency
    print("\n   Top 5 modes by frequency:")
    print(f"   {'#':<5} {'Frequency (Hz)':<20} {'Damping':<15} {'Mode Shape Norm':<20}")
    print(f"   {'-'*5} {'-'*20} {'-'*15} {'-'*20}")
    
    sorted_idx = np.argsort(freqs)[:5]
    for i, idx in enumerate(sorted_idx):
        norm = np.linalg.norm(mode_shapes[:, idx])
        print(f"   {i+1:<5} {freqs[idx]:<20.4f} {damps[idx]:<15.6f} {norm:<20.6f}")


def load_csv_example():
    """Example: How to load data from CSV file"""
    print("\n" + "="*70)
    print("HOW TO LOAD DATA FROM CSV FILE")
    print("="*70)
    
    print("""
    # If you have a CSV file with acceleration data:
    # File format (example):
    #   time,sensor1,sensor2,sensor3
    #   0.0,0.001,0.002,0.0015
    #   0.01,0.0012,0.0025,0.0018
    #   ...
    
    import pandas as pd
    
    # Load data
    df = pd.read_csv('acceleration_data.csv')
    
    # Extract sensor columns
    data = df[['sensor1', 'sensor2', 'sensor3']].values
    
    # Run analysis
    from shm_oma import perform_ssi_cov
    freqs, damps, modes = perform_ssi_cov(data, order=30)
    """)


def numpy_file_example():
    """Example: How to work with NumPy binary files"""
    print("\n" + "="*70)
    print("HOW TO SAVE AND LOAD DATA WITH NUMPY")
    print("="*70)
    
    print("""
    import numpy as np
    from shm_oma import perform_ssi_cov
    
    # YOUR DATA HERE (from sensors, files, etc.)
    data = np.random.randn(5000, 4)  # 5000 samples, 4 sensors
    
    # SAVE the data
    np.save('my_acceleration_data.npy', data)
    print("Data saved to my_acceleration_data.npy")
    
    # LOAD the data later
    loaded_data = np.load('my_acceleration_data.npy')
    
    # Run SSI-COV
    frequencies, damping_ratios, mode_shapes = perform_ssi_cov(
        loaded_data, 
        order=25,
        max_lag=1000
    )
    
    print(f"Found {len(frequencies)} modes")
    for i, f in enumerate(frequencies[:5]):
        print(f"Mode {i+1}: {f:.2f} Hz")
    """)


def continuous_monitoring_example():
    """Example: Continuous SHM monitoring workflow"""
    print("\n" + "="*70)
    print("CONTINUOUS MONITORING WORKFLOW")
    print("="*70)
    
    print("""
    from shm_oma import FrequencyTracker, perform_ssi_cov
    from datetime import datetime, timedelta
    import numpy as np
    
    # Initialize tracker
    tracker = FrequencyTracker(mac_threshold=0.85, freq_tol=0.10)
    
    # Load data files (every 30 minutes)
    timestamp = datetime(2026, 3, 5, 0, 0, 0)
    
    for hour in range(24):
        # In practice: load from data file or sensor stream
        window_data = np.load(f'data_window_{hour}.npy')
        
        # Extract modal parameters
        freqs, damps, modes = perform_ssi_cov(window_data, order=30)
        
        # Update tracker (matches modes across time)
        tracker.update(timestamp, freqs, damps, modes)
        
        timestamp += timedelta(minutes=30)
    
    # Analyze trends
    from shm_oma import ModalTrackingAnalyzer
    analyzer = ModalTrackingAnalyzer(tracker)
    
    for mode_id in [1, 2, 3]:
        stats = analyzer.compute_frequency_statistics(mode_id)
        print(f"Mode {mode_id}: {stats['mean_freq']:.2f} ± {stats['std_freq']:.4f} Hz")
        
        trend = analyzer.compute_trend(mode_id)
        if trend['slope'] < -0.001:
            print(f"  ⚠️ WARNING: Frequency decreasing at {trend['slope']:.6f} Hz/sec")
    """)


def step_by_step_guide():
    """Step-by-step guide for users"""
    print("\n" + "="*70)
    print("STEP-BY-STEP GUIDE")
    print("="*70)
    
    guide = """
    STEP 1: Prepare your acceleration data
    ======================================
    Your data should be a numpy array with shape (n_samples, n_sensors)
    
    Example with 10,000 samples and 4 sensors:
        data = np.random.randn(10000, 4)  # Replace with your data
    
    
    STEP 2: Basic modal analysis (single window)
    ============================================
    from shm_oma import perform_ssi_cov
    
    frequencies, damping_ratios, mode_shapes = perform_ssi_cov(
        data,
        order=30,           # Model order (try 20-50)
        max_lag=1000,      # Autocorrelation lag
        use_randomized_svd=True
    )
    
    
    STEP 3: Multiple model orders (better stability)
    ================================================
    from shm_oma import filter_spurious_poles, automate_poles_dbscan
    
    # Extract poles at different orders
    all_freqs, all_damps, all_modes = [], [], []
    for order in [25, 30, 35, 40]:
        f, d, m = perform_ssi_cov(data, order=order)
        all_freqs.append(f)
        all_damps.append(d)
        all_modes.append(m)
    
    # Filter spurious poles
    stable_f, stable_d, stable_m = filter_spurious_poles(
        all_freqs, all_damps, all_modes
    )
    
    # Cluster physical modes
    physical_modes = automate_poles_dbscan(
        stable_f, stable_d, stable_m,
        eps_freq=0.05
    )
    
    for freq, damp, mode in physical_modes:
        print(f"Mode: {freq:.2f} Hz, Damping: {damp:.4f}")
    
    
    STEP 4: Continuous tracking over time
    ====================================
    from shm_oma import FrequencyTracker, ModalTrackingAnalyzer
    from datetime import datetime, timedelta
    
    tracker = FrequencyTracker()
    timestamp = datetime.now()
    
    for i in range(100):  # Process 100 time windows
        window_data = load_window_i(i)  # Your data loading function
        freqs, damps, modes = perform_ssi_cov(window_data, order=30)
        tracker.update(timestamp, freqs, damps, modes)
        timestamp += timedelta(minutes=30)
    
    # Analyze
    analyzer = ModalTrackingAnalyzer(tracker)
    for mode_id in tracker.tracker.reference_modes:
        stats = analyzer.compute_frequency_statistics(mode_id)
        print(f"Mode {mode_id}: {stats['mean_freq']:.2f} Hz")
    """
    
    print(guide)


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  SHM_OMA - QUICK START GUIDE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Run quick analysis
    quick_analysis()
    
    # Show examples
    load_csv_example()
    numpy_file_example()
    continuous_monitoring_example()
    step_by_step_guide()
    
    print("\n" + "█"*70)
    print("█" + "  Ready to use! Try the code snippets above".center(68) + "█")
    print("█"*70 + "\n")
