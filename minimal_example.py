"""
MINIMAL EXAMPLE: Analyze acceleration data in 10 lines
"""

import numpy as np
from shm_oma import perform_ssi_cov

# YOUR DATA (replace with your actual data)
data = np.random.randn(5000, 3)  # 5000 samples, 3 sensors

# RUN ANALYSIS
frequencies, damping, modes = perform_ssi_cov(data, order=30)

# DISPLAY RESULTS
print(f"Identified {len(frequencies)} modes\n")
for i, (f, d) in enumerate(zip(frequencies[:5], damping[:5])):
    print(f"Mode {i+1}: {f:.2f} Hz, Damping: {d:.6f}")
