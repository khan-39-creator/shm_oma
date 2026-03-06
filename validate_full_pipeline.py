"""
Synthetic Data Generator & Full Pipeline Validation for shm_oma
================================================================

Creates a simulated 5-DOF shear building excited by white noise.
The exact modal parameters (frequencies, damping, mode shapes) are
computed analytically from the mass and stiffness matrices, so we
have a **ground-truth** to validate every module in shm_oma.

Ground-truth modal parameters
------------------------------
The script prints the exact values at the top of its output so you
can compare them against the SSI-COV identification results.

Modules exercised
-----------------
1. perform_ssi_cov        — core identification
2. extract_stable_poles   — multi-order stability
3. filter_spurious_poles  — automation stability
4. automate_poles_dbscan  — DBSCAN clustering
5. automate_poles_hierarchical — hierarchical clustering
6. mac / vectorized_mac   — mode shape correlation
7. FrequencyTracker       — continuous tracking
8. ModalTrackingAnalyzer  — statistics & anomaly detection
9. plot_stabilization_diagram, plot_psd_with_peaks, plot_singular_values
10. compute_psd
"""

import numpy as np
from scipy import linalg as la
from scipy import signal
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt

# ── shm_oma imports ──────────────────────────────────────────────
from shm_oma import (
    perform_ssi_cov,
    extract_stable_poles,
    compute_autocorrelation,
    build_toeplitz_matrix,
    compute_psd,
    plot_stabilization_diagram,
    plot_psd_with_peaks,
    plot_singular_values,
    automate_poles_dbscan,
    automate_poles_hierarchical,
    filter_spurious_poles,
    mac,
    vectorized_mac,
    FrequencyTracker,
    ModalTrackingAnalyzer,
)
from datetime import datetime, timedelta


# =====================================================================
# 1.  DEFINE THE STRUCTURE  (5-DOF shear building)
# =====================================================================
n_dof   = 5           # number of floors / sensors
fs      = 200.0       # sampling frequency [Hz]
dt      = 1.0 / fs
T_total = 200.0       # total duration [s]  — long record for good statistics
n_samples = int(T_total * fs)

# Mass and stiffness — tuned so modes are in 2-20 Hz range
m_floor = 1.0         # [kg]  (unit mass for simplicity)
k_storey = 4000.0     # [N/m]  → gives modes in ~3-19 Hz range

M = np.eye(n_dof) * m_floor
K = np.zeros((n_dof, n_dof))
for i in range(n_dof):
    if i == 0:
        K[i, i]   =  2 * k_storey
        K[i, i+1] = -k_storey
    elif i == n_dof - 1:
        K[i, i]   =  k_storey
        K[i, i-1] = -k_storey
    else:
        K[i, i]   =  2 * k_storey
        K[i, i-1] = -k_storey
        K[i, i+1] = -k_storey

# ── Solve eigenvalue problem  K φ = ω² M φ ──────────────────────
eigvals, eigvecs = la.eigh(K, M)          # returns sorted ascending
omega_n  = np.sqrt(eigvals)               # rad/s
freq_true = omega_n / (2 * np.pi)         # Hz

# Normalise mode shapes to unit norm (same convention as shm_oma)
mode_shapes_true = eigvecs.copy()
for i in range(n_dof):
    mode_shapes_true[:, i] /= np.linalg.norm(mode_shapes_true[:, i])

# Assign Rayleigh damping  C = α M + β K
# Target: ζ₁ ≈ 2 %  and  ζ₅ ≈ 5 %
zeta_targets = np.array([0.02, 0.05])
omega_pair   = np.array([omega_n[0], omega_n[-1]])
AB = np.array([[1/(2*omega_pair[0]), omega_pair[0]/2],
               [1/(2*omega_pair[1]), omega_pair[1]/2]])
alpha_beta = np.linalg.solve(AB, zeta_targets)
alpha, beta = alpha_beta
C = alpha * M + beta * K

# True damping ratios for all modes
zeta_true = alpha / (2 * omega_n) + beta * omega_n / 2


# =====================================================================
# 2.  PRINT GROUND TRUTH
# =====================================================================
print("=" * 72)
print("  GROUND-TRUTH MODAL PARAMETERS  (5-DOF shear building)")
print("=" * 72)
print(f"  Sampling freq : {fs:.1f} Hz")
print(f"  Duration      : {T_total:.1f} s  ({n_samples} samples)")
print(f"  Rayleigh alpha : {alpha:.6f}")
print(f"  Rayleigh beta  : {beta:.6f}")
print()
print(f"  {'Mode':>4s}  {'Freq [Hz]':>10s}  {'Damping [%]':>12s}")
print(f"  {'----':>4s}  {'----------':>10s}  {'----------':>12s}")
for i in range(n_dof):
    print(f"  {i+1:4d}  {freq_true[i]:10.4f}  {zeta_true[i]*100:12.4f}")
print()
print("  Mode shapes (columns = modes, rows = floors):")
header = "  Floor  " + "  ".join([f"  Mode{i+1:d}" for i in range(n_dof)])
print(header)
for r in range(n_dof):
    row = f"  {r+1:5d}  "
    row += "  ".join([f"{mode_shapes_true[r, c]:+8.4f}" for c in range(n_dof)])
    print(row)
print()


# =====================================================================
# 3.  SIMULATE RESPONSE  (Newmark-β integration, white noise input)
# =====================================================================
np.random.seed(2026)
F_ext = np.random.randn(n_samples, n_dof) * 10.0   # white noise on every DOF

# State-space:  x = [q; q_dot],  A_ss, B_ss
M_inv = np.linalg.inv(M)
A_ss = np.zeros((2*n_dof, 2*n_dof))
A_ss[:n_dof, n_dof:] = np.eye(n_dof)
A_ss[n_dof:, :n_dof] = -M_inv @ K
A_ss[n_dof:, n_dof:] = -M_inv @ C

B_ss = np.zeros((2*n_dof, n_dof))
B_ss[n_dof:, :] = M_inv

C_ss = np.zeros((n_dof, 2*n_dof))
C_ss[:, :n_dof] = -M_inv @ K          # acceleration output
C_ss[:, n_dof:] = -M_inv @ C

D_ss = M_inv   # direct feed-through for acceleration

sys_c = signal.StateSpace(A_ss, B_ss, C_ss, D_ss)
sys_d = sys_c.to_discrete(dt=dt, method="zoh")

# Simulate
t_vec = np.arange(n_samples) * dt
_, acc_data, _ = signal.dlsim(sys_d, F_ext)

print(f"  Simulated acceleration shape: {acc_data.shape}")
print(f"  RMS per sensor: {np.std(acc_data, axis=0).round(2)}")
print()


# =====================================================================
# 4.  SINGLE-ORDER SSI-COV IDENTIFICATION
# =====================================================================
print("=" * 72)
print("  SSI-COV IDENTIFICATION  (single order = 50)")
print("=" * 72)

freqs, damps, modes, sv = perform_ssi_cov(
    acc_data, order=50, max_lag=5000, sampling_freq=fs, xi_max=0.15
)

print(f"  Identified {len(freqs)} modes\n")
print(f"  {'#':>3s}  {'Freq [Hz]':>10s}  {'Damp [%]':>10s}  {'Closest true':>14s}  {'Freq err%':>10s}  {'MAC':>8s}")
print(f"  {'---':>3s}  {'----------':>10s}  {'--------':>10s}  {'-----------':>14s}  {'---------':>10s}  {'------':>8s}")

for i in range(min(len(freqs), 10)):
    # find best MAC match to any true mode
    mac_vals = [mac(modes[:, i], mode_shapes_true[:, j]) for j in range(n_dof)]
    idx_closest = np.argmax(mac_vals)
    freq_err = abs(freqs[i] - freq_true[idx_closest]) / freq_true[idx_closest] * 100
    mac_val = mac_vals[idx_closest]
    print(f"  {i+1:3d}  {freqs[i]:10.4f}  {damps[i]*100:10.4f}  "
          f"Mode {idx_closest+1:d} ({freq_true[idx_closest]:.2f})  "
          f"{freq_err:10.3f}  {mac_val:8.4f}")
print()


# =====================================================================
# 5.  MULTI-ORDER STABILITY ANALYSIS
# =====================================================================
print("=" * 72)
print("  MULTI-ORDER STABILITY  (orders 30-70, step 5)")
print("=" * 72)

orders = list(range(30, 75, 5))
all_freqs, all_damps, all_modes = [], [], []

for order in orders:
    f, d, m, _ = perform_ssi_cov(acc_data, order=order, max_lag=5000, sampling_freq=fs, xi_max=0.15)
    all_freqs.append(f)
    all_damps.append(d)
    all_modes.append(m)

sf, sd, sm = extract_stable_poles(
    all_freqs, all_damps, all_modes,
    freq_tol=0.02, damp_tol=0.10, mac_threshold=0.90
)

print(f"  Stable poles found: {len(sf)}")
for i in range(len(sf)):
    idx = np.argmin(np.abs(freq_true - sf[i]))
    print(f"    {sf[i]:.4f} Hz  (ζ={sd[i]*100:.3f}%)  ← true Mode {idx+1} = {freq_true[idx]:.4f} Hz")
print()


# =====================================================================
# 6.  AUTOMATION — filter_spurious_poles
# =====================================================================
print("=" * 72)
print("  FILTER SPURIOUS POLES  (across all orders)")
print("=" * 72)

sf2, sd2, sm2 = filter_spurious_poles(
    all_freqs, all_damps, all_modes,
    freq_tol=0.02, damp_tol=0.10, mac_threshold=0.85
)
print(f"  Stable poles after filtering: {len(sf2)}")
for i in range(len(sf2)):
    idx = np.argmin(np.abs(freq_true - sf2[i]))
    print(f"    {sf2[i]:.4f} Hz  (ζ={sd2[i]*100:.3f}%)  ← true Mode {idx+1}")
print()


# =====================================================================
# 7.  DBSCAN CLUSTERING
# =====================================================================
print("=" * 72)
print("  DBSCAN CLUSTERING  (on pooled multi-order poles)")
print("=" * 72)

# Pool all poles from all orders
pooled_f = np.concatenate(all_freqs)
pooled_d = np.concatenate(all_damps)
pooled_m = np.hstack(all_modes)

clusters = automate_poles_dbscan(pooled_f, pooled_d, pooled_m, eps_freq=0.5, min_samples=3)
print(f"  Clusters found: {len(clusters)}")
for freq, damp, shape in clusters:
    idx = np.argmin(np.abs(freq_true - freq))
    mac_val = mac(shape, mode_shapes_true[:, idx])
    print(f"    {freq:.4f} Hz  (ζ={damp*100:.3f}%)  MAC={mac_val:.4f}  ← true Mode {idx+1}")
print()


# =====================================================================
# 8.  HIERARCHICAL CLUSTERING
# =====================================================================
print("=" * 72)
print("  HIERARCHICAL CLUSTERING")
print("=" * 72)

clusters_h = automate_poles_hierarchical(pooled_f, pooled_d, pooled_m, distance_threshold=0.15)
print(f"  Clusters found: {len(clusters_h)}")
for freq, damp, shape in clusters_h[:10]:
    idx = np.argmin(np.abs(freq_true - freq))
    mac_val = mac(shape, mode_shapes_true[:, idx])
    print(f"    {freq:.4f} Hz  (ζ={damp*100:.3f}%)  MAC={mac_val:.4f}  ← true Mode {idx+1}")
print()


# =====================================================================
# 9.  MAC & VECTORIZED MAC VALIDATION
# =====================================================================
print("=" * 72)
print("  MAC VALIDATION")
print("=" * 72)

mac_matrix = vectorized_mac(mode_shapes_true)
print("  MAC matrix of TRUE mode shapes (should be identity):")
print(np.array2string(mac_matrix, precision=4, suppress_small=True))
print()

# Cross-MAC: identified vs true
if len(freqs) >= n_dof:
    print("  Cross-MAC: identified (rows) vs true (cols) for first 5 modes:")
    for i in range(min(5, len(freqs))):
        idx_closest = np.argmin(np.abs(freq_true - freqs[i]))
        vals = [mac(modes[:, i], mode_shapes_true[:, j]) for j in range(n_dof)]
        best = np.argmax(vals)
        line = "    " + "  ".join([f"{v:.3f}" for v in vals])
        line += f"  ← best match: Mode {best+1}"
        print(line)
print()


# =====================================================================
# 10. FREQUENCY TRACKING  (split data into 10 windows)
# =====================================================================
print("=" * 72)
print("  FREQUENCY TRACKING  (10 consecutive windows)")
print("=" * 72)

tracker = FrequencyTracker(mac_threshold=0.70, freq_tol=0.15, damping_tol=0.10)

n_windows = 10
window_len = n_samples // n_windows
t_start = datetime(2026, 1, 1, 0, 0, 0)

for w in range(n_windows):
    chunk = acc_data[w * window_len : (w + 1) * window_len]
    f_w, d_w, m_w, _ = perform_ssi_cov(chunk, order=40, max_lag=2000, sampling_freq=fs, xi_max=0.15)
    t_w = t_start + timedelta(minutes=30 * w)
    tracker.update(t_w, f_w, d_w, m_w)

print(f"  Reference modes after tracking: {len(tracker.reference_modes)}")
for mid, mf, md, _ in tracker.reference_modes[:7]:
    idx = np.argmin(np.abs(freq_true - mf))
    print(f"    Mode ID {mid:2d}: {mf:.4f} Hz  (ζ={md*100:.3f}%)  ← true Mode {idx+1}")
print()


# =====================================================================
# 11. MODAL TRACKING ANALYZER — statistics & anomalies
# =====================================================================
print("=" * 72)
print("  MODAL TRACKING ANALYZER")
print("=" * 72)

analyzer = ModalTrackingAnalyzer(tracker)

for mid in list(tracker.history.keys())[:5]:
    stats = analyzer.compute_frequency_statistics(mid)
    if not stats:
        continue
    trend = analyzer.compute_trend(mid)
    anomalies = analyzer.detect_frequency_anomalies(mid, threshold=2.5)
    idx = np.argmin(np.abs(freq_true - stats["mean_freq"]))

    print(f"  Mode ID {mid} → true Mode {idx+1} ({freq_true[idx]:.2f} Hz)")
    print(f"    mean={stats['mean_freq']:.4f}  std={stats['std_freq']:.4f}  "
          f"range={stats['freq_range']:.4f}  n={stats['n_observations']}")
    if trend:
        print(f"    trend slope={trend['slope']:.2e} Hz/s  R²={trend['r_squared']:.4f}")
    print(f"    anomalies (z>2.5): {anomalies}")
    print()


# =====================================================================
# 12. FREEZE / RESET BASELINE
# =====================================================================
print("=" * 72)
print("  BASELINE MANAGEMENT")
print("=" * 72)

ref_before = tracker.reference_modes[0][1]
tracker.freeze_baseline()
# Feed one more window — baseline should NOT change
chunk = acc_data[:window_len]
f_w, d_w, m_w, _ = perform_ssi_cov(chunk, order=40, max_lag=2000, sampling_freq=fs, xi_max=0.15)
tracker.update(t_start + timedelta(hours=10), f_w, d_w, m_w)
ref_after = tracker.reference_modes[0][1]
print(f"  Freeze test: ref freq before={ref_before:.6f}  after={ref_after:.6f}  "
      f"{'PASS (unchanged)' if ref_before == ref_after else 'FAIL'}")

tracker.unfreeze_baseline()
print(f"  Baseline unfrozen ✓")
print()


# =====================================================================
# 13. VISUALISATION  (save to files)
# =====================================================================
print("=" * 72)
print("  GENERATING PLOTS")
print("=" * 72)

# Stabilisation diagram
fig, ax = plot_stabilization_diagram(
    acc_data, order_range=range(20, 70, 5), sampling_freq=fs, xi_max=0.15
)
# Overlay true frequencies
for ft in freq_true:
    ax.axhline(ft, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
ax.legend(["Identified poles", *[f"True f={ft:.1f} Hz" for ft in freq_true]], fontsize=8)
fig.savefig("validation_stabilization.png", dpi=150, bbox_inches="tight")
print("  Saved: validation_stabilization.png")

# PSD with peaks
fig2, axes2 = plot_psd_with_peaks(acc_data, sampling_freq=fs, frequencies=freq_true[:5])
fig2.savefig("validation_psd.png", dpi=150, bbox_inches="tight")
print("  Saved: validation_psd.png")

# Singular values
fig3, ax3 = plot_singular_values(sv)
fig3.savefig("validation_svd.png", dpi=150, bbox_inches="tight")
print("  Saved: validation_svd.png")

# Frequency tracking history
fig4, ax4 = plt.subplots(figsize=(12, 5))
for mid in list(tracker.history.keys())[:5]:
    ts, freqs_hist = tracker.get_frequency_trends(mid)
    if len(ts) > 1:
        x = [(t - ts[0]).total_seconds() / 60 for t in ts]
        ax4.plot(x, freqs_hist, "o-", markersize=4, label=f"Mode {mid}")
for ft in freq_true:
    ax4.axhline(ft, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
ax4.set_xlabel("Time [min]")
ax4.set_ylabel("Frequency [Hz]")
ax4.set_title("Frequency Tracking Over Time")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
fig4.savefig("validation_tracking.png", dpi=150, bbox_inches="tight")
print("  Saved: validation_tracking.png")

plt.close("all")


# =====================================================================
# 14. FINAL ACCURACY SUMMARY
# =====================================================================
print()
print("=" * 72)
print("  FINAL ACCURACY SUMMARY")
print("=" * 72)

matched = 0
for i in range(n_dof):
    if len(freqs) == 0:
        break
    # Find best-matching identified mode by MAC (not just frequency proximity)
    best_mac = -1
    best_j = 0
    for j in range(len(freqs)):
        mv = mac(modes[:, j], mode_shapes_true[:, i])
        if mv > best_mac:
            best_mac = mv
            best_j = j
    ferr = abs(freqs[best_j] - freq_true[i]) / freq_true[i] * 100
    derr = abs(damps[best_j] - zeta_true[i]) * 100

    status = "PASS" if (ferr < 5.0 and best_mac > 0.85) else "FAIL"
    if ferr < 5.0 and best_mac > 0.85:
        matched += 1

    print(f"  True Mode {i+1}: f={freq_true[i]:.4f} Hz  zeta={zeta_true[i]*100:.3f}%")
    print(f"    Identified : f={freqs[best_j]:.4f} Hz  zeta={damps[best_j]*100:.3f}%")
    print(f"    Freq error : {ferr:.3f}%   Damp error: {derr:.3f} pp   MAC: {best_mac:.4f}  {status}")
    print()

print(f"  RESULT: {matched}/{n_dof} modes correctly identified "
      f"(freq err < 5%, MAC > 0.85)")
print("=" * 72)
