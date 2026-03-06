#!/usr/bin/env python
"""
Interactive demo: Analyze your acceleration data with shm_oma

Usage:
    python interactive_demo.py

Options:
    1. Analyze generated data
    2. Load your CSV file
    3. Load your .npy file
    4. Interactive mode
"""

import numpy as np
from pathlib import Path
from shm_oma import (
    perform_ssi_cov,
    automate_poles_dbscan,
    filter_spurious_poles,
    FrequencyTracker,
    ModalTrackingAnalyzer,
)


def display_results(frequencies, damping_ratios, mode_shapes):
    """Display modal analysis results nicely"""
    print("\n" + "="*70)
    print("MODAL ANALYSIS RESULTS")
    print("="*70)
    
    if len(frequencies) == 0:
        print("No modes identified.")
        return
    
    print(f"\nTotal poles identified: {len(frequencies)}\n")
    
    # Sort by frequency
    sort_idx = np.argsort(frequencies)
    
    print(f"{'#':<5} {'Frequency (Hz)':<20} {'Damping':<15} {'Mode Norm':<15}")
    print("-" * 55)
    
    for rank, idx in enumerate(sort_idx[:10]):  # Show top 10
        f = frequencies[idx]
        d = damping_ratios[idx]
        norm = np.linalg.norm(mode_shapes[:, idx])
        print(f"{rank+1:<5} {f:<20.4f} {d:<15.6f} {norm:<15.6f}")


def analyze_generated_data():
    """Option 1: Generate and analyze synthetic data"""
    print("\n" + "="*70)
    print("OPTION 1: ANALYZE GENERATED DATA")
    print("="*70)
    
    # Parameters
    print("\nData parameters:")
    duration = input("  Duration (seconds) [default: 60]: ") or "60"
    duration = float(duration)
    
    sampling_rate = input("  Sampling rate (Hz) [default: 100]: ") or "100"
    sampling_rate = int(sampling_rate)
    
    n_sensors = input("  Number of sensors [default: 3]: ") or "3"
    n_sensors = int(n_sensors)
    
    model_order = input("  Model order for SSI-COV [default: 25]: ") or "25"
    model_order = int(model_order)
    
    # Generate data
    print("\nGenerating data...")
    n_samples = duration * sampling_rate
    t = np.linspace(0, duration, n_samples)
    
    # Create damped oscillations at different frequencies
    data = np.zeros((n_samples, n_sensors))
    
    frequencies_true = [2.5, 7.2, 12.1]  # Hz
    for freq in frequencies_true[:n_sensors]:
        damp = 0.03
        response = np.exp(-2*np.pi*damp*freq*t) * np.sin(2*np.pi*freq*t)
        for sensor in range(n_sensors):
            data[:, sensor] += response * (1 + 0.3*np.sin(2*np.pi*sensor/n_sensors))
    
    # Add noise
    data += 0.05 * np.random.randn(n_samples, n_sensors)
    
    print(f"Generated data: {data.shape}")
    print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
    
    # Run analysis
    print(f"\nRunning SSI-COV with order {model_order}...")
    max_lag = int(n_samples / 5)
    frequencies, damping_ratios, mode_shapes = perform_ssi_cov(
        data,
        order=model_order,
        max_lag=max_lag,
        use_randomized_svd=True
    )
    
    display_results(frequencies, damping_ratios, mode_shapes)


def analyze_csv_file():
    """Option 2: Load and analyze CSV file"""
    print("\n" + "="*70)
    print("OPTION 2: LOAD AND ANALYZE CSV FILE")
    print("="*70)
    
    filepath = input("\nEnter path to CSV file: ").strip()
    
    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return
    
    try:
        # Try with pandas first
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            print(f"\nLoaded CSV: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Ask which columns are sensor data
            sensor_cols = input("Enter sensor columns (comma-separated) [or 'all' for numeric columns]: ").strip()
            
            if sensor_cols.lower() == 'all':
                data = df.select_dtypes(include=[np.number]).values
            else:
                cols = [c.strip() for c in sensor_cols.split(',')]
                data = df[cols].values
            
        except ImportError:
            # Fall back to numpy
            print("(pandas not available, using numpy.loadtxt)")
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        
        print(f"Data shape: {data.shape}")
        
        # Ask for analysis parameters
        model_order = input("Model order for SSI-COV [default: 25]: ") or "25"
        model_order = int(model_order)
        
        print(f"\nRunning SSI-COV with order {model_order}...")
        max_lag = max(100, data.shape[0] // 5)
        
        frequencies, damping_ratios, mode_shapes = perform_ssi_cov(
            data,
            order=model_order,
            max_lag=max_lag,
            use_randomized_svd=True
        )
        
        display_results(frequencies, damping_ratios, mode_shapes)
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")


def analyze_numpy_file():
    """Option 3: Load and analyze .npy file"""
    print("\n" + "="*70)
    print("OPTION 3: LOAD AND ANALYZE NUMPY FILE (.npy)")
    print("="*70)
    
    filepath = input("\nEnter path to .npy file: ").strip()
    
    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return
    
    try:
        data = np.load(filepath)
        print(f"Loaded data: {data.shape}")
        
        if len(data.shape) != 2:
            print("❌ Data must be 2D array (n_samples, n_sensors)")
            return
        
        # Ask for analysis parameters
        model_order = input("Model order for SSI-COV [default: 25]: ") or "25"
        model_order = int(model_order)
        
        print(f"\nRunning SSI-COV with order {model_order}...")
        max_lag = max(100, data.shape[0] // 5)
        
        frequencies, damping_ratios, mode_shapes = perform_ssi_cov(
            data,
            order=model_order,
            max_lag=max_lag,
            use_randomized_svd=True
        )
        
        display_results(frequencies, damping_ratios, mode_shapes)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  SHM_OMA - INTERACTIVE ANALYSIS TOOL".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    menu = """
Choose an option:
  1 - Analyze generated synthetic data
  2 - Load and analyze CSV file
  3 - Load and analyze NumPy file (.npy)
  4 - Exit
"""
    
    while True:
        print(menu)
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            analyze_generated_data()
        elif choice == '2':
            analyze_csv_file()
        elif choice == '3':
            analyze_numpy_file()
        elif choice == '4':
            print("\n✓ Goodbye!")
            break
        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Exited by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
