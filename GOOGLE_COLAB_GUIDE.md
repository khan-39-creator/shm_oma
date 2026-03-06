# 🚀 Google Colab Setup Guide for SHM_OMA

## Quick Start (5 minutes)

### Step 1: Open Google Colab

1. Go to **[colab.research.google.com](https://colab.research.google.com)**
2. Click **"New Notebook"**
3. Name it: `SHM_OMA Analysis`

---

### Step 2: Install Library (Copy & Paste into Cell 1)

```python
# Install dependencies
!pip install numpy scipy scikit-learn

# Install shm_oma from GitHub
!pip install git+https://github.com/yourusername/shm_oma.git

# Or install locally (if uploaded)
# !pip install -e /content/shm_oma

print("✅ Installation complete!")
```

**Then press `Ctrl+Enter` to run**

---

### Step 3: Quick Test (Copy & Paste into Cell 2)

```python
import numpy as np
from shm_oma import perform_ssi_cov

# Generate sample data
data = np.random.randn(5000, 3)  # 5000 samples, 3 sensors

# Run analysis
frequencies, damping, modes = perform_ssi_cov(data, order=30)

# Show results
print(f"Found {len(frequencies)} modes")
for i, f in enumerate(frequencies[:5]):
    print(f"Mode {i+1}: {f:.2f} Hz")
```

**Done!** ✅

---

## Working with Your Data

### Option A: Upload CSV File

```python
from google.colab import files
import pandas as pd

# Upload your file
uploaded = files.upload()

# Load CSV
df = pd.read_csv(list(uploaded.keys())[0])
data = df.select_dtypes(include=[np.number]).values

# Analyze
frequencies, damping, modes = perform_ssi_cov(data, order=25)
```

### Option B: Use Google Drive

```python
from google.colab import drive
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# Load file from Drive
df = pd.read_csv('/content/drive/MyDrive/your_data.csv')
data = df.select_dtypes(include=[np.number]).values

# Analyze
frequencies, damping, modes = perform_ssi_cov(data, order=25)
```

### Option C: Download from URL

```python
import pandas as pd

# Download CSV from URL
url = 'https://example.com/your_data.csv'
df = pd.read_csv(url)
data = df.select_dtypes(include=[np.number]).values

# Analyze
frequencies, damping, modes = perform_ssi_cov(data, order=25)
```

---

## Complete Examples

### Example 1: Basic Analysis

```python
import numpy as np
from shm_oma import perform_ssi_cov

# YOUR DATA HERE
data = np.random.randn(5000, 3)

# Analyze
frequencies, damping, modes = perform_ssi_cov(data, order=30)

# Display
print(f"\n{'#':<5} {'Frequency (Hz)':<20} {'Damping':<15}")
print("-" * 40)
for i, (f, d) in enumerate(zip(frequencies[:10], damping[:10])):
    print(f"{i+1:<5} {f:<20.4f} {d:<15.6f}")
```

### Example 2: Frequency Tracking

```python
from shm_oma import FrequencyTracker, ModalTrackingAnalyzer
from datetime import datetime, timedelta

# Initialize tracker
tracker = FrequencyTracker()
timestamp = datetime.now()

# Process 20 time windows
for i in range(20):
    window_data = np.random.randn(2000, 3)  # Your data
    freqs, damps, modes = perform_ssi_cov(window_data, order=25)
    tracker.update(timestamp, freqs, damps, modes)
    timestamp += timedelta(hours=1)

# Analyze
analyzer = ModalTrackingAnalyzer(tracker)
for mode_id in [1, 2]:
    stats = analyzer.compute_frequency_statistics(mode_id)
    print(f"Mode {mode_id}: {stats['mean_freq']:.2f} Hz ± {stats['std_freq']:.4f}")
```

### Example 3: Advanced with Filtering

```python
from shm_oma import filter_spurious_poles, automate_poles_dbscan

data = np.random.randn(5000, 3)

# Multiple orders
all_freqs, all_damps, all_modes = [], [], []
for order in [25, 30, 35, 40]:
    f, d, m = perform_ssi_cov(data, order=order)
    all_freqs.append(f)
    all_damps.append(d)
    all_modes.append(m)

# Filter spurious poles
stable_f, stable_d, stable_m = filter_spurious_poles(all_freqs, all_damps, all_modes)

# Cluster physical modes
physical_modes = automate_poles_dbscan(stable_f, stable_d, stable_m, eps_freq=0.1)

print(f"Identified {len(physical_modes)} physical modes")
for freq, damp, mode in physical_modes:
    print(f"  {freq:.2f} Hz, damping: {damp:.4f}")
```

---

## Visualization (with Matplotlib)

```python
import matplotlib.pyplot as plt

# Plot frequencies
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(range(len(frequencies)), sorted(frequencies)[:50], s=20)
plt.xlabel('Mode Number')
plt.ylabel('Frequency (Hz)')
plt.title('Identified Natural Frequencies')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(sorted(frequencies)[:50], damping[:50], s=20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Damping Ratio')
plt.title('Frequency vs Damping')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Save and Download Results

```python
# Save to CSV
import pandas as pd

results_df = pd.DataFrame({
    'Frequency_Hz': frequencies,
    'Damping_Ratio': damping
})

results_df.to_csv('modal_analysis_results.csv', index=False)

# Download
from google.colab import files
files.download('modal_analysis_results.csv')
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Import error | Run installation cell first |
| Out of memory | Reduce data size or use `use_randomized_svd=True` |
| Slow execution | Colab is slower than local; be patient |
| File not found | Use `files.upload()` or mount Google Drive |

---

## Pro Tips

✅ **Use GPU** for faster computation:
- Runtime → Change runtime type → GPU
- Then run: `!pip install cupy` for GPU acceleration

✅ **Save results to Drive**:
```python
results_df.to_csv('/content/drive/MyDrive/results.csv')
```

✅ **Share notebook** - Click Share button in top right

---

## Need Help?

- 📖 Full docs: Check README.md locally
- 💬 Issues: Report on GitHub
- 🚀 Try example notebooks in project folder

**Happy analyzing! 🎉**
