"""
Corrected SSI-COV Implementation with Stabilization Diagram and PSD Analysis

This example demonstrates the IMPROVED SSI-COV algorithm with:
1. Correct state matrix extraction based on SSI-COV theory
2. Proper eigenvalue to frequency/damping conversion
3. Stabilization diagram for model order selection
4. PSD (Power Spectral Density) plots
5. Singular value analysis
6. Comparison with FFT PSD
"""

import numpy as np
import matplotlib.pyplot as plt
from shm_oma import (
    perform_ssi_cov,
    compute_psd,
    plot_stabilization_diagram,
    plot_psd_with_peaks,
    plot_singular_values,
)


def create_synthetic_mdof_system(
    true_frequencies,
    damping_ratios,
    duration=100,
    sampling_freq=100,
    noise_level=0.01,
    n_sensors=3,
):
    """
    Generate synthetic acceleration data from known modal parameters.
    
    Parameters
    ----------
    true_frequencies : list
        True natural frequencies in Hz
    damping_ratios : list
        True damping ratios
    duration : float
        Signal duration in seconds
    sampling_freq : float
        Sampling frequency in Hz
    noise_level : float
        Noise standard deviation
    n_sensors : int
        Number of sensors
    
    Returns
    -------
    data : ndarray, shape (n_samples, n_sensors)
        Synthetic acceleration data
    """
    n_samples = int(duration * sampling_freq)
    t = np.arange(n_samples) / sampling_freq
    dt = 1.0 / sampling_freq
    
    # Create mode shapes (example: 3-DOF system)
    if n_sensors == 3:
        mode_shapes = np.array([
            [1.0, -1.0, 1.0],       # Mode 1
            [1.0, 0.5, -0.5],       # Mode 2
            [1.0, 2.0, 1.5],        # Mode 3
        ]).T  # shape: (n_sensors, n_modes)
    else:
        mode_shapes = np.random.randn(n_sensors, len(true_frequencies))
        for i in range(mode_shapes.shape[1]):
            mode_shapes[:, i] /= np.linalg.norm(mode_shapes[:, i])
    
    data = np.zeros((n_samples, n_sensors))
    
    # Generate response for each mode
    for mode_idx, (freq, zeta) in enumerate(zip(true_frequencies, damping_ratios)):
        wn = 2 * np.pi * freq  # Natural frequency (rad/s)
        wd = wn * np.sqrt(1 - zeta**2)  # Damped frequency
        
        # Impulse response (mode contribution)
        h = np.exp(-zeta * wn * t) * np.sin(wd * t + 0.1)
        
        # Scale by mode shape
        mode_response = np.outer(h, mode_shapes[:, mode_idx])
        data += mode_response
    
    # Add noise
    data += np.random.randn(*data.shape) * noise_level
    
    return data


def main():
    print("=" * 80)
    print("CORRECTED SSI-COV ALGORITHM WITH VISUALIZATION")
    print("=" * 80)
    
    # Define true system parameters
    true_frequencies = np.array([2.5, 5.2, 8.1])
    true_damping = np.array([0.05, 0.03, 0.04])
    
    print(f"\nTrue Modal Parameters:")
    print(f"  Frequencies: {true_frequencies} Hz")
    print(f"  Damping ratios: {true_damping}")
    
    # Generate synthetic data
    print("\nGenerating synthetic MDOF system response...")
    data = create_synthetic_mdof_system(
        true_frequencies=true_frequencies,
        damping_ratios=true_damping,
        duration=100,
        sampling_freq=100,
        noise_level=0.002,
        n_sensors=3,
    )
    
    sampling_freq = 100.0
    print(f"Data shape: {data.shape} (samples, sensors)")
    
    # ========== 1. SINGLE SSI-COV ANALYSIS ==========
    print("\n" + "=" * 80)
    print("1. SSI-COV MODAL ANALYSIS (Single Model Order)")
    print("=" * 80)
    
    model_order = 30
    frequencies, damping_ratios, mode_shapes, singular_values = perform_ssi_cov(
        data,
        order=model_order,
        sampling_freq=sampling_freq,
    )
    
    print(f"\nIdentified {len(frequencies)} modes with order={model_order}:")
    for i, (f, d) in enumerate(zip(frequencies[:10], damping_ratios[:10])):
        error = abs(f - true_frequencies[i % len(true_frequencies)]) / true_frequencies[i % len(true_frequencies)] * 100
        print(f"  Mode {i+1}: {f:6.3f} Hz (error: {error:5.1f}%), ζ={d:.4f}")
    
    # ========== 2. STABILIZATION DIAGRAM ==========
    print("\n" + "=" * 80)
    print("2. STABILIZATION DIAGRAM (Model Order Selection)")
    print("=" * 80)
    print("Plotting stabilization diagram...")
    print("  - Shows which poles are stable across different model orders")
    print("  - Stable physical poles appear as vertical lines")
    print("  - Noise poles appear randomly scattered")
    
    order_range = list(range(15, 51, 2))
    fig_stab, ax_stab = plot_stabilization_diagram(
        data,
        order_range=order_range,
        sampling_freq=sampling_freq,
        figsize=(14, 8),
    )
    plt.savefig('stabilization_diagram.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: stabilization_diagram.png")
    
    # ========== 3. SINGULAR VALUE ANALYSIS ==========
    print("\n" + "=" * 80)
    print("3. SINGULAR VALUE ANALYSIS")
    print("=" * 80)
    print("Plotting singular values from SVD decomposition...")
    print("  - Shows energy distribution in data")
    print("  - Knee point indicates appropriate model order")
    
    fig_sv, ax_sv = plot_singular_values(singular_values, figsize=(10, 6))
    plt.savefig('singular_values.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: singular_values.png")
    
    # ========== 4. PSD ANALYSIS ==========
    print("\n" + "=" * 80)
    print("4. POWER SPECTRAL DENSITY (PSD) ANALYSIS")
    print("=" * 80)
    
    # Compute PSD
    freq_range, psd = compute_psd(data, sampling_freq=sampling_freq, method='welch')
    
    print(f"PSD computed for {len(freq_range)} frequencies")
    print(f"Frequency range: {freq_range[0]:.2f} - {freq_range[-1]:.2f} Hz")
    
    # Plot PSD with identified peaks
    fig_psd, axes_psd = plot_psd_with_peaks(
        data,
        sampling_freq=sampling_freq,
        frequencies=frequencies[:5],  # Mark first 5 identified modes
        figsize=(14, 8),
    )
    plt.savefig('psd_with_peaks.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: psd_with_peaks.png")
    
    # ========== 5. COMPARISON: IDENTIFIED vs TRUE FREQUENCIES ==========
    print("\n" + "=" * 80)
    print("5. FREQUENCY COMPARISON: IDENTIFIED vs TRUE")
    print("=" * 80)
    
    fig_compare, ax_compare = plt.subplots(figsize=(12, 6))
    
    # Plot true frequencies
    for i, (f_true, d_true) in enumerate(zip(true_frequencies, true_damping)):
        ax_compare.axvline(f_true, color='g', linestyle='--', linewidth=2, alpha=0.7)
        ax_compare.text(f_true, -0.5, f'True {i+1}', ha='center', fontsize=10, color='g')
    
    # Plot identified frequencies
    for i, (f_id, d_id) in enumerate(zip(frequencies[:len(true_frequencies)], damping_ratios[:len(true_frequencies)])):
        ax_compare.axvline(f_id, color='r', linestyle=':', linewidth=2, alpha=0.7)
        ax_compare.text(f_id, 0.5, f'ID {i+1}', ha='center', fontsize=10, color='r')
    
    # Add PSD overlay
    ax_compare_psd = ax_compare.twinx()
    ax_compare_psd.semilogy(freq_range, psd[:, 0], 'b-', alpha=0.3, linewidth=1, label='PSD (Sensor 1)')
    ax_compare_psd.set_ylabel('PSD', fontsize=11)
    
    ax_compare.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_compare.set_ylabel('Detection', fontsize=12)
    ax_compare.set_title('Frequency Identification Comparison', fontsize=14, fontweight='bold')
    ax_compare.set_xlim([0, max(frequencies[:5].max(), true_frequencies.max()) + 2])
    ax_compare.grid(True, alpha=0.3)
    
    plt.savefig('frequency_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: frequency_comparison.png")
    
    # ========== 6. MODE SHAPES ==========
    print("\n" + "=" * 80)
    print("6. IDENTIFIED MODE SHAPES")
    print("=" * 80)
    
    n_modes_to_plot = min(3, mode_shapes.shape[1])
    fig_modes, axes_modes = plt.subplots(1, n_modes_to_plot, figsize=(12, 4))
    if n_modes_to_plot == 1:
        axes_modes = [axes_modes]
    
    for i in range(n_modes_to_plot):
        ax = axes_modes[i]
        mode = mode_shapes[:, i]
        colors = ['r' if m >= 0 else 'b' for m in mode]
        ax.bar(range(len(mode)), np.abs(mode), color=colors, alpha=0.7)
        ax.set_xlabel('Sensor', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'Mode {i+1}\n{frequencies[i]:.2f} Hz', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Identified Mode Shapes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mode_shapes.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: mode_shapes.png")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Successfully extracted {len(frequencies)} modal parameters")
    print(f"✓ Generated {n_modes_to_plot} visualizations")
    print("\nFiles generated:")
    print("  1. stabilization_diagram.png - Model order selection")
    print("  2. singular_values.png - SVD energy distribution")
    print("  3. psd_with_peaks.png - PSD with identified peaks")
    print("  4. frequency_comparison.png - True vs identified frequencies")
    print("  5. mode_shapes.png - Identified mode shapes")
    
    print("\n" + "=" * 80)
    print("KEY IMPROVEMENTS IN THIS VERSION:")
    print("=" * 80)
    print("""
1. ✓ CORRECTED STATE MATRIX EXTRACTION
   - Proper SSI-COV block shift relationship: A = O2 @ O1^+
   - O1 = O[:-n_sensors, :] and O2 = O[n_sensors:, :]

2. ✓ FIXED EIGENVALUE TO MODAL PARAMETER CONVERSION
   - Proper discrete-to-continuous time transformation
   - lambda_c = ln(z) / dt
   - Natural frequency: wn = sqrt(real²+ imag²)
   - Damping ratio: zeta = -real / wn

3. ✓ PHYSICAL FREQUENCY FILTERING
   - Only returns physically meaningful modes (0 ≤ ζ ≤ 1)
   - Checks Nyquist frequency limit
   - Filters noise poles automatically

4. ✓ STABILIZATION DIAGRAM
   - Shows mode stability across different model orders
   - Helps identify true physical modes vs noise poles
   - Improves model order selection

5. ✓ PSD ANALYSIS
   - Welch method for robust spectral estimation
   - Visual confirmation of extracted frequencies
   - Cross-validation with time-domain results

6. ✓ SINGULAR VALUE PLOTS
   - Shows energy distribution in SVD
   - Helps choose appropriate model order
   - Knee point detection included
    """)
    
    plt.show()


if __name__ == "__main__":
    main()
