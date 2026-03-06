"""
=============================================================================
SHM_OMA - Complete Data Analysis Script
=============================================================================

This script provides everything you need to analyze your acceleration data
using the shm_oma library. Just customize the CONFIGURATION section below
and run!

Features:
  ✓ Load data from CSV or NumPy files
  ✓ Perform SSI-COV modal analysis
  ✓ Automated pole filtering
  ✓ Continuous frequency tracking
  ✓ Export results to CSV

Usage:
    python analyze_data.py
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Import shm_oma functions
from shm_oma import (
    perform_ssi_cov,
    automate_poles_dbscan,
    filter_spurious_poles,
    FrequencyTracker,
    ModalTrackingAnalyzer,
)


# ============================================================================
# CONFIGURATION: Customize these settings
# ============================================================================

class Config:
    """User configuration"""
    
    # INPUT DATA
    DATA_SOURCE = 'generate'  # Options: 'generate', 'csv', 'npy'
    
    # For CSV files
    CSV_FILE = 'your_data.csv'
    CSV_COLUMNS = ['sensor1', 'sensor2', 'sensor3']  # Columns to use
    
    # For .npy files
    NPY_FILE = 'acceleration_data.npy'
    
    # For generated data
    GENERATE_DURATION = 120  # seconds
    GENERATE_SAMPLING_RATE = 100  # Hz
    GENERATE_N_SENSORS = 3
    
    # SSI-COV PARAMETERS
    MODEL_ORDER = 30  # Try 20-50 depending on data quality
    MAX_LAG = None  # None = automatic (n_samples // 2)
    USE_RANDOMIZED_SVD = True  # Fast approximation
    
    # POLE FILTERING (for multiple model orders)
    APPLY_FILTERING = False  # Set True for better stability
    FILTER_ORDERS = [25, 30, 35, 40]
    FREQ_TOLERANCE = 0.01  # 1% frequency tolerance
    DAMP_TOLERANCE = 0.05  # 5% damping tolerance
    MAC_THRESHOLD = 0.98
    
    # CLUSTERING
    APPLY_CLUSTERING = False  # Set True to group physical modes
    CLUSTER_EPS_FREQ = 0.1  # Frequency clustering distance
    CLUSTER_MIN_SAMPLES = 2
    
    # FREQUENCY TRACKING (for time series data)
    APPLY_TRACKING = False  # Set True for continuous monitoring
    N_TIME_WINDOWS = 10  # Number of time windows to process
    WINDOW_DURATION = 60  # seconds per window
    
    # OUTPUT
    SAVE_RESULTS = True
    OUTPUT_FILE = 'analysis_results.txt'
    EXPORT_CSV = False
    CSV_OUTPUT = 'modes_results.csv'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_csv_data(filepath, columns=None):
    """Load data from CSV file"""
    try:
        import pandas as pd
        print(f"\n📂 Loading CSV: {filepath}")
        df = pd.read_csv(filepath)
        
        if columns:
            data = df[columns].values
        else:
            # Use all numeric columns
            data = df.select_dtypes(include=[np.number]).values
        
        print(f"   Shape: {data.shape}")
        return data
    except ImportError:
        print("⚠️  pandas not installed, using numpy.loadtxt")
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return data


def load_numpy_data(filepath):
    """Load data from .npy file"""
    print(f"\n📂 Loading NumPy file: {filepath}")
    data = np.load(filepath)
    print(f"   Shape: {data.shape}")
    return data


def generate_synthetic_data(duration, fs, n_sensors):
    """Generate synthetic acceleration data for testing"""
    print(f"\n🔄 Generating synthetic data...")
    print(f"   Duration: {duration}s, Sampling: {fs}Hz, Sensors: {n_sensors}")
    
    n_samples = duration * fs
    t = np.linspace(0, duration, n_samples)
    data = np.zeros((n_samples, n_sensors))
    
    # Define 3 modes
    modes = [
        {'freq': 2.5, 'damp': 0.03, 'shape': [1.0, 0.8, 0.5]},
        {'freq': 7.2, 'damp': 0.02, 'shape': [1.0, -0.5, 0.8]},
        {'freq': 12.1, 'damp': 0.04, 'shape': [1.0, 0.3, -0.9]},
    ]
    
    for mode in modes:
        f = mode['freq']
        d = mode['damp']
        shape = np.array(mode['shape'])
        
        response = np.exp(-2*np.pi*d*f*t) * np.sin(2*np.pi*f*t)
        for sensor in range(n_sensors):
            idx = sensor % len(shape)
            data[:, sensor] += response * shape[idx]
    
    # Add noise
    data += 0.05 * np.random.randn(n_samples, n_sensors)
    
    print(f"   Created: {data.shape}")
    return data


def display_results(frequencies, damping_ratios, mode_shapes, max_modes=10):
    """Display analysis results in a formatted table"""
    if len(frequencies) == 0:
        print("\n❌ No modes identified.")
        return
    
    print("\n" + "="*75)
    print("MODAL ANALYSIS RESULTS")
    print("="*75)
    
    # Sort by frequency
    sort_idx = np.argsort(frequencies)
    n_show = min(max_modes, len(frequencies))
    
    print(f"\nIdentified {len(frequencies)} poles (showing top {n_show}):\n")
    print(f"{'#':<5} {'Frequency':<18} {'Damping':<15} {'Mode Norm':<15}")
    print(f"{'':5} {'(Hz)':<18} {'Ratio':<15}")
    print("-" * 75)
    
    for rank, idx in enumerate(sort_idx[-n_show:][::-1]):
        f = frequencies[idx]
        d = damping_ratios[idx]
        norm = np.linalg.norm(mode_shapes[:, idx]) if mode_shapes.shape[1] > 0 else 0
        status = "✓" if f > 0 and 0 <= d <= 1 else "⚠"
        print(f"{rank+1:<5} {f:<18.4f} {d:<15.6f} {norm:<15.6f} {status}")


def apply_stability_filtering(data, config):
    """Run SSI-COV at multiple orders and filter spurious poles"""
    print("\n" + "="*75)
    print("POLE FILTERING AND CLUSTERING")
    print("="*75)
    
    print(f"\n📊 Running SSI-COV at model orders: {config.FILTER_ORDERS}")
    
    all_freqs, all_damps, all_modes = [], [], []
    for order in config.FILTER_ORDERS:
        f, d, m = perform_ssi_cov(data, order=order, use_randomized_svd=config.USE_RANDOMIZED_SVD)
        all_freqs.append(f)
        all_damps.append(d)
        all_modes.append(m)
        print(f"   Order {order}: {len(f)} poles")
    
    # Filter spurious poles
    print(f"\n🔍 Filtering spurious poles...")
    print(f"   Frequency tolerance: {config.FREQ_TOLERANCE*100:.1f}%")
    print(f"   Damping tolerance: {config.DAMP_TOLERANCE:.2f}")
    print(f"   MAC threshold: {config.MAC_THRESHOLD:.2f}")
    
    stable_freqs, stable_damps, stable_modes = filter_spurious_poles(
        all_freqs, all_damps, all_modes,
        freq_tol=config.FREQ_TOLERANCE,
        damp_tol=config.DAMP_TOLERANCE,
        mac_threshold=config.MAC_THRESHOLD
    )
    
    print(f"   ✓ {len(stable_freqs)} stable poles identified")
    
    if config.APPLY_CLUSTERING and len(stable_freqs) > 0:
        print(f"\n🎯 Clustering stable poles...")
        
        physical_modes = automate_poles_dbscan(
            stable_freqs, stable_damps, stable_modes,
            eps_freq=config.CLUSTER_EPS_FREQ,
            min_samples=config.CLUSTER_MIN_SAMPLES
        )
        
        print(f"   ✓ {len(physical_modes)} physical modes identified\n")
        
        print(f"{'#':<5} {'Frequency':<18} {'Damping':<15}")
        print(f"{'':5} {'(Hz)':<18} {'Ratio':<15}")
        print("-" * 40)
        
        for i, (freq, damp, mode) in enumerate(physical_modes):
            print(f"{i+1:<5} {freq:<18.4f} {damp:<15.6f}")
        
        return stable_freqs, stable_damps, stable_modes, physical_modes
    
    return stable_freqs, stable_damps, stable_modes, None


def apply_tracking(data, config):
    """Run continuous frequency tracking on simulated time windows"""
    print("\n" + "="*75)
    print("CONTINUOUS FREQUENCY TRACKING")
    print("="*75)
    
    # For real data, you would load different time windows
    # Here we simulate by adding time-varying noise
    
    print(f"\n⏱️  Processing {config.N_TIME_WINDOWS} time windows...")
    
    tracker = FrequencyTracker(
        mac_threshold=0.85,
        freq_tol=0.10,
        damping_tol=0.10,
        adaptation_rate=0.05
    )
    
    timestamp = datetime.now()
    window_size = len(data) // config.N_TIME_WINDOWS
    
    for i in range(config.N_TIME_WINDOWS):
        # Extract window
        start_idx = i * window_size
        end_idx = min((i+1) * window_size, len(data))
        window_data = data[start_idx:end_idx]
        
        # Analyze window
        freqs, damps, modes = perform_ssi_cov(
            window_data,
            order=config.MODEL_ORDER,
            use_randomized_svd=config.USE_RANDOMIZED_SVD
        )
        
        # Track modes
        tracker.update(timestamp, freqs, damps, modes)
        timestamp += timedelta(minutes=30)
        
        print(f"   Window {i+1}/{config.N_TIME_WINDOWS}: {len(freqs)} poles")
    
    # Analyze tracked modes
    print(f"\n🔎 Analyzing tracked modes...")
    analyzer = ModalTrackingAnalyzer(tracker)
    
    print(f"\n{'Mode':<8} {'Mean Freq':<18} {'Std Dev':<15} {'N Obs':<8}")
    print(f"{'':8} {'(Hz)':<18} {'(Hz)':<15}")
    print("-" * 50)
    
    for mode_id in range(1, len(tracker.reference_modes) + 1):
        stats = analyzer.compute_frequency_statistics(mode_id)
        if stats:
            print(f"{mode_id:<8} {stats['mean_freq']:<18.4f} "
                  f"{stats['std_freq']:<15.6f} {stats['n_observations']:<8}")
    
    return tracker, analyzer


def save_results(frequencies, damping_ratios, config, tracker=None, analyzer=None):
    """Save results to file"""
    if not config.SAVE_RESULTS:
        return
    
    print(f"\n💾 Saving results to {config.OUTPUT_FILE}...")
    
    with open(config.OUTPUT_FILE, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SHM_OMA - MODAL ANALYSIS RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now()}\n\n")
        
        f.write(f"IDENTIFIED MODES (Total: {len(frequencies)})\n")
        f.write("-"*70 + "\n")
        f.write(f"{'#':<5} {'Frequency (Hz)':<20} {'Damping Ratio':<20}\n")
        f.write("-"*70 + "\n")
        
        sort_idx = np.argsort(frequencies)
        for rank, idx in enumerate(sort_idx):
            f.write(f"{rank+1:<5} {frequencies[idx]:<20.4f} {damping_ratios[idx]:<20.6f}\n")
        
        if tracker and analyzer:
            f.write("\n\nFREQUENCY TRACKING ANALYSIS\n")
            f.write("-"*70 + "\n")
            
            for mode_id in range(1, len(tracker.reference_modes) + 1):
                stats = analyzer.compute_frequency_statistics(mode_id)
                if stats:
                    f.write(f"\nMode {mode_id}:\n")
                    f.write(f"  Mean Frequency: {stats['mean_freq']:.4f} Hz\n")
                    f.write(f"  Std Dev: {stats['std_freq']:.6f} Hz\n")
                    f.write(f"  Min: {stats['min_freq']:.4f} Hz\n")
                    f.write(f"  Max: {stats['max_freq']:.4f} Hz\n")
                    f.write(f"  Observations: {stats['n_observations']}\n")
    
    print(f"✓ Results saved")


def export_to_csv(frequencies, damping_ratios, config):
    """Export results to CSV"""
    if not config.EXPORT_CSV:
        return
    
    try:
        import pandas as pd
        
        df = pd.DataFrame({
            'Frequency_Hz': frequencies,
            'Damping_Ratio': damping_ratios,
        })
        
        df.to_csv(config.CSV_OUTPUT, index=False)
        print(f"✓ Data exported to {config.CSV_OUTPUT}")
    except ImportError:
        print("⚠️  pandas not installed, skipping CSV export")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Main analysis pipeline"""
    
    print("\n" + "█"*75)
    print("█" + " "*73 + "█")
    print("█" + "  SHM_OMA - COMPLETE DATA ANALYSIS".center(73) + "█")
    print("█" + " "*73 + "█")
    print("█"*75)
    
    try:
        # Load data
        print("\n" + "="*75)
        print("STEP 1: LOAD DATA")
        print("="*75)
        
        if Config.DATA_SOURCE == 'csv':
            if not Path(Config.CSV_FILE).exists():
                print(f"❌ File not found: {Config.CSV_FILE}")
                return
            data = load_csv_data(Config.CSV_FILE, Config.CSV_COLUMNS)
        elif Config.DATA_SOURCE == 'npy':
            if not Path(Config.NPY_FILE).exists():
                print(f"❌ File not found: {Config.NPY_FILE}")
                return
            data = load_numpy_data(Config.NPY_FILE)
        else:  # generate
            data = generate_synthetic_data(
                Config.GENERATE_DURATION,
                Config.GENERATE_SAMPLING_RATE,
                Config.GENERATE_N_SENSORS
            )
        
        # Basic SSI-COV analysis
        print("\n" + "="*75)
        print("STEP 2: MODAL ANALYSIS (SSI-COV)")
        print("="*75)
        
        max_lag = Config.MAX_LAG if Config.MAX_LAG else len(data) // 2
        print(f"\n⚙️  Running SSI-COV with model order {Config.MODEL_ORDER}")
        print(f"   Max lag: {max_lag}")
        print(f"   Using randomized SVD: {Config.USE_RANDOMIZED_SVD}")
        
        frequencies, damping_ratios, mode_shapes = perform_ssi_cov(
            data,
            order=Config.MODEL_ORDER,
            max_lag=max_lag,
            use_randomized_svd=Config.USE_RANDOMIZED_SVD
        )
        
        display_results(frequencies, damping_ratios, mode_shapes)
        
        # Optional: Pole filtering and clustering
        if Config.APPLY_FILTERING:
            stable_f, stable_d, stable_m, phys_modes = apply_stability_filtering(data, Config)
            frequencies, damping_ratios, mode_shapes = stable_f, stable_d, stable_m
        
        # Optional: Frequency tracking
        tracker = None
        analyzer = None
        if Config.APPLY_TRACKING:
            tracker, analyzer = apply_tracking(data, Config)
        
        # Save results
        save_results(frequencies, damping_ratios, Config, tracker, analyzer)
        export_to_csv(frequencies, damping_ratios, Config)
        
        print("\n" + "█"*75)
        print("█" + "  ✓ ANALYSIS COMPLETE".center(73) + "█")
        print("█"*75 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
