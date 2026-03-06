"""
Frequency tracking module for continuous SHM systems.

This module maintains a baseline mode set and tracks identified modes
over time using the Modal Assurance Criterion (MAC) and frequency proximity.
Supports baseline freezing/resetting for long-term monitoring.

References
----------
[1] Magalhaes, F., Cunha, A., & Caetano, E. (2009). Online automatic
    identification of modal parameters. MSSP, 23(2), 316-329.
[2] Cheynet, E. (2020). Operational modal analysis with automated SSI-COV
    algorithm (MATLAB Central File Exchange).
"""

import numpy as np
from datetime import datetime
from .automation import mac


class FrequencyTracker:
    """
    Track modal frequencies and mode shapes over time.

    Maintains a baseline set of modes and matches new mode observations
    against the baseline using MAC and frequency proximity criteria.

    Parameters
    ----------
    mac_threshold : float, default 0.85
        Minimum MAC value to consider a mode matched.
    freq_tol : float, default 0.1
        Maximum relative frequency difference to match modes (10 %).
    damping_tol : float, default 0.1
        Maximum absolute damping difference to match modes.
    adaptation_rate : float, default 0.05
        Exponential moving average rate for baseline updates (0-1).
        Set to 0 to freeze the baseline completely.
    """

    def __init__(
        self,
        mac_threshold=0.85,
        freq_tol=0.1,
        damping_tol=0.1,
        adaptation_rate=0.05,
    ):
        self.mac_threshold = mac_threshold
        self.freq_tol = freq_tol
        self.damping_tol = damping_tol
        self.adaptation_rate = adaptation_rate

        # mode_id -> list of (timestamp, frequency, damping)
        self.history = {}

        # List of (mode_id, baseline_freq, baseline_damp, baseline_shape)
        self.reference_modes = []

        self.next_mode_id = 1
        self.initialized = False
        self._baseline_frozen = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, timestamp, frequencies, damping_ratios, mode_shapes):
        """
        Initialise the tracker with baseline modes.

        Parameters
        ----------
        timestamp : datetime or float
        frequencies : ndarray, shape (n_modes,)
        damping_ratios : ndarray, shape (n_modes,)
        mode_shapes : ndarray, shape (n_sensors, n_modes)
        """
        for i in range(len(frequencies)):
            mode_id = self.next_mode_id
            f = frequencies[i]
            d = damping_ratios[i]
            mode_shape = mode_shapes[:, i].copy()

            self.reference_modes.append((mode_id, f, d, mode_shape))
            self.history[mode_id] = [(timestamp, f, d)]
            self.next_mode_id += 1

        self.initialized = True

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, timestamp, frequencies, damping_ratios, mode_shapes):
        """
        Update tracker with new modes from the latest time window.

        Parameters
        ----------
        timestamp : datetime or float
        frequencies : ndarray, shape (n_modes,)
        damping_ratios : ndarray, shape (n_modes,)
        mode_shapes : ndarray, shape (n_sensors, n_modes)
        """
        if not self.initialized:
            self.initialize(timestamp, frequencies, damping_ratios, mode_shapes)
            return

        matched_refs = set()

        for new_idx in range(len(frequencies)):
            new_f = frequencies[new_idx]
            new_d = damping_ratios[new_idx]
            new_mode = mode_shapes[:, new_idx]

            best_match_id = None
            best_match_score = -1.0

            for ref_id, ref_f, ref_d, ref_mode in self.reference_modes:
                if ref_id in matched_refs:
                    continue

                freq_diff = np.abs(new_f - ref_f) / (ref_f + 1e-12)
                if freq_diff > self.freq_tol:
                    continue

                damp_diff = np.abs(new_d - ref_d)
                if damp_diff > self.damping_tol:
                    continue

                mac_val = mac(new_mode, ref_mode)
                if mac_val >= self.mac_threshold and mac_val > best_match_score:
                    best_match_score = mac_val
                    best_match_id = ref_id

            if best_match_id is not None:
                matched_refs.add(best_match_id)
                self.history[best_match_id].append((timestamp, new_f, new_d))

                # Adapt baseline (unless frozen)
                if not self._baseline_frozen and self.adaptation_rate > 0:
                    ref_idx = next(
                        i
                        for i, (m_id, _, _, _) in enumerate(self.reference_modes)
                        if m_id == best_match_id
                    )
                    _, ref_f, ref_d, ref_m = self.reference_modes[ref_idx]
                    a = self.adaptation_rate

                    updated_freq = (1 - a) * ref_f + a * new_f
                    updated_damp = (1 - a) * ref_d + a * new_d
                    updated_mode = (1 - a) * ref_m + a * new_mode
                    norm = np.linalg.norm(updated_mode)
                    if norm > 1e-10:
                        updated_mode /= norm

                    self.reference_modes[ref_idx] = (
                        best_match_id,
                        updated_freq,
                        updated_damp,
                        updated_mode,
                    )
            else:
                # New mode — not matched to any reference
                mode_id = self.next_mode_id
                self.reference_modes.append(
                    (mode_id, new_f, new_d, new_mode.copy())
                )
                self.history[mode_id] = [(timestamp, new_f, new_d)]
                self.next_mode_id += 1

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def freeze_baseline(self):
        """Freeze the baseline so that adaptation_rate has no effect."""
        self._baseline_frozen = True

    def unfreeze_baseline(self):
        """Re-enable baseline adaptation."""
        self._baseline_frozen = False

    def reset_baseline(self, timestamp, frequencies, damping_ratios, mode_shapes):
        """
        Discard all reference modes and re-initialise with new baseline data.

        History is preserved; only the reference set is replaced.

        Parameters
        ----------
        timestamp : datetime or float
        frequencies : ndarray, shape (n_modes,)
        damping_ratios : ndarray, shape (n_modes,)
        mode_shapes : ndarray, shape (n_sensors, n_modes)
        """
        self.reference_modes = []
        for i in range(len(frequencies)):
            mode_id = self.next_mode_id
            self.reference_modes.append(
                (mode_id, frequencies[i], damping_ratios[i], mode_shapes[:, i].copy())
            )
            self.history[mode_id] = [(timestamp, frequencies[i], damping_ratios[i])]
            self.next_mode_id += 1

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_history(self, mode_id):
        """Get the time history for a specific mode."""
        return self.history.get(mode_id, [])

    def get_all_history(self):
        """Get the complete history dictionary."""
        return self.history.copy()

    def get_current_modes(self):
        """
        Get current baseline modes.

        Returns
        -------
        modes : list of tuples
            ``(mode_id, frequency, damping, mode_shape)``
        """
        return self.reference_modes.copy()

    def get_frequency_trends(self, mode_id):
        """
        Extract frequency trend for a mode.

        Returns
        -------
        timestamps : ndarray
        frequencies : ndarray
        """
        if mode_id not in self.history:
            return np.array([]), np.array([])

        data = self.history[mode_id]
        timestamps = np.array([h[0] for h in data])
        frequencies = np.array([h[1] for h in data])
        return timestamps, frequencies

    def export_to_dict(self):
        """
        Export tracking data to a dictionary (for serialisation).

        Returns
        -------
        export_dict : dict
        """
        return {
            "history": self.history,
            "reference_modes": [
                {
                    "mode_id": m_id,
                    "baseline_freq": float(m_f),
                    "baseline_damp": float(m_d),
                }
                for m_id, m_f, m_d, _ in self.reference_modes
            ],
            "metadata": {
                "mac_threshold": self.mac_threshold,
                "freq_tol": self.freq_tol,
                "damping_tol": self.damping_tol,
                "adaptation_rate": self.adaptation_rate,
                "baseline_frozen": self._baseline_frozen,
            },
        }


# ======================================================================
# Modal Tracking Analyzer
# ======================================================================

class ModalTrackingAnalyzer:
    """
    Analyse tracked modal data for SHM applications.

    Computes statistical summaries, detects anomalies, and generates
    reports on structural health based on frequency and damping trends.
    """

    def __init__(self, tracker):
        """
        Parameters
        ----------
        tracker : FrequencyTracker
        """
        self.tracker = tracker

    def compute_frequency_statistics(self, mode_id):
        """
        Compute statistics for a mode's frequency over time.

        Returns
        -------
        stats : dict
        """
        timestamps, frequencies = self.tracker.get_frequency_trends(mode_id)
        if len(frequencies) == 0:
            return {}

        return {
            "mode_id": mode_id,
            "mean_freq": float(np.mean(frequencies)),
            "std_freq": float(np.std(frequencies)),
            "min_freq": float(np.min(frequencies)),
            "max_freq": float(np.max(frequencies)),
            "freq_range": float(np.max(frequencies) - np.min(frequencies)),
            "n_observations": len(frequencies),
        }

    def detect_frequency_anomalies(self, mode_id, threshold=3.0):
        """
        Detect anomalous frequency measurements using z-score.

        Parameters
        ----------
        mode_id : int
        threshold : float, default 3.0
            Number of standard deviations to flag as anomaly.

        Returns
        -------
        anomalies : list of int
            Indices of anomalous observations.
        """
        timestamps, frequencies = self.tracker.get_frequency_trends(mode_id)
        if len(frequencies) < 2:
            return []

        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)

        if std_freq < 1e-12:
            return []

        z_scores = np.abs((frequencies - mean_freq) / std_freq)
        return np.where(z_scores > threshold)[0].tolist()

    def compute_trend(self, mode_id):
        """
        Compute linear trend in frequency over time.

        Returns
        -------
        trend_info : dict
            Contains 'slope', 'intercept', 'r_squared', 'unit'.
        """
        timestamps, frequencies = self.tracker.get_frequency_trends(mode_id)
        if len(frequencies) < 2:
            return {}

        # Convert timestamps to numeric (seconds from start)
        if isinstance(timestamps[0], (int, float, np.integer, np.floating)):
            t_numeric = np.array(timestamps, dtype=float)
        else:
            t_start = timestamps[0]
            t_numeric = np.array(
                [(t - t_start).total_seconds() for t in timestamps]
            )

        coeffs = np.polyfit(t_numeric, frequencies, 1)
        slope, intercept = coeffs

        y_pred = np.polyval(coeffs, t_numeric)
        ss_res = np.sum((frequencies - y_pred) ** 2)
        ss_tot = np.sum((frequencies - np.mean(frequencies)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mode_id": mode_id,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "unit": "Hz per second",
        }
