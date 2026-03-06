"""
SIMPLEST WAY TO USE SHM_OMA IN GOOGLE COLAB

Just copy everything below into ONE cell and run it!
"""

# Install
!pip install numpy scipy scikit-learn -q
!pip install git+https://github.com/yourusername/shm_oma.git -q 2>/dev/null || echo "Note: Install from GitHub requires correct repo"

# Imports
import numpy as np
from datetime import datetime, timedelta

try:
    from shm_oma import perform_ssi_cov, FrequencyTracker, ModalTrackingAnalyzer
    print("✅ shm_oma imported successfully\n")
except ImportError as e:
    print(f"Note: {e}\nUsing local installation or manual setup\n")

# ============================================================================
# OPTION 1: Analyze Generated Data (No upload needed)
# ============================================================================

print("=" * 70)
print("OPTION 1: ANALYZE GENERATED DATA (Demo)")
print("=" * 70)

# Create sample data
np.random.seed(42)
data = np.random.randn(5000, 3)  # 5000 samples, 3 sensors
print(f"\n📊 Generated sample data: {data.shape}")

# Run SSI-COV
print("🔍 Running SSI-COV analysis...")
frequencies, damping, modes = perform_ssi_cov(data, order=30)

# Show results
print(f"\n✓ Found {len(frequencies)} modes\n")
print(f"{'#':<5} {'Frequency':<18} {'Damping':<15}")
print(f"{'':5} {'(Hz)':<18} {'Ratio':<15}")
print("-" * 40)
for i, (f, d) in enumerate(zip(frequencies[:10], damping[:10])):
    print(f"{i+1:<5} {f:<18.4f} {d:<15.6f}")

# ============================================================================
# OPTION 2: Upload Your CSV File
# ============================================================================

print("\n" + "=" * 70)
print("OPTION 2: UPLOAD YOUR OWN CSV FILE")
print("=" * 70)

try:
    # Try uploading
    from google.colab import files
    import pandas as pd
    
    print("\n📂 Choose your CSV file to upload:")
    uploaded = files.upload()
    
    if uploaded:
        filename = list(uploaded.keys())[0]
        print(f"\n✓ Uploaded: {filename}")
        
        # Load data
        df = pd.read_csv(filename)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Extract sensor data
        data = df.select_dtypes(include=[np.number]).values
        print(f"  Data extracted: {data.shape}\n")
        
        # Analyze
        print("🔍 Analyzing your data...")
        frequencies, damping, modes = perform_ssi_cov(data, order=25)
        
        print(f"\n✓ Found {len(frequencies)} modes\n")
        print(f"{'#':<5} {'Frequency':<18} {'Damping':<15}")
        print(f"{'':5} {'(Hz)':<18} {'Ratio':<15}")
        print("-" * 40)
        for i, (f, d) in enumerate(zip(frequencies[:10], damping[:10])):
            print(f"{i+1:<5} {f:<18.4f} {d:<15.6f}")
        
        # Option to download results
        print("\n💾 Saving results...")
        results_df = pd.DataFrame({
            'Frequency_Hz': frequencies,
            'Damping_Ratio': damping
        })
        results_df.to_csv('shm_oma_results.csv', index=False)
        
        print("✓ Results saved to shm_oma_results.csv")
        print("\n📥 Downloading results...")
        files.download('shm_oma_results.csv')
        
except ImportError:
    print("⚠️ Google Colab file upload not available in this environment")
except Exception as e:
    print(f"⚠️ Upload error: {e}")

# ============================================================================
# OPTION 3: Use Google Drive
# ============================================================================

print("\n" + "=" * 70)
print("OPTION 3: LOAD FROM GOOGLE DRIVE")
print("=" * 70)

try:
    print("""
To use files from Google Drive:

1. Run this code:
   from google.colab import drive
   drive.mount('/content/drive')

2. Then load your file:
   import pandas as pd
   df = pd.read_csv('/content/drive/MyDrive/your_data.csv')
   data = df.select_dtypes(include=[np.number]).values
   
3. Analyze:
   frequencies, damping, modes = perform_ssi_cov(data, order=25)
""")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# BONUS: Quick Frequency Tracking Example
# ============================================================================

print("\n" + "=" * 70)
print("BONUS: FREQUENCY TRACKING (Simulation)")
print("=" * 70)

print("\n⏱️ Simulating 10 time windows of sensor data...\n")

tracker = FrequencyTracker()
timestamp = datetime(2026, 3, 5, 0, 0, 0)

for i in range(10):
    # Simulate window data
    window_data = np.random.randn(2000, 3)
    freqs, damps, modes = perform_ssi_cov(window_data, order=20)
    tracker.update(timestamp, freqs, damps, modes)
    timestamp += timedelta(hours=1)
    print(f"  Window {i+1}: Updated {len(freqs)} modes")

# Analyze
analyzer = ModalTrackingAnalyzer(tracker)

print(f"\n{'Mode':<8} {'Mean Freq':<18} {'Std Dev':<15}")
print(f"{'':8} {'(Hz)':<18} {'(Hz)':<15}")
print("-" * 40)

for mode_id in range(1, len(tracker.reference_modes) + 1):
    stats = analyzer.compute_frequency_statistics(mode_id)
    if stats:
        print(f"{mode_id:<8} {stats['mean_freq']:<18.4f} {stats['std_freq']:<15.6f}")

print("\n" + "=" * 70)
print("✅ DONE! Now customize and run your analysis")
print("=" * 70)
