"""
Frequency tracking module for continuous SHM systems.

This module maintains a baseline mode set and tracks identified modes
over time using the Modal Assurance Criterion (MAC) and frequency proximity.
"""

import numpy as np
from datetime import datetime
from .automation import mac


class FrequencyTracker:
    """
    Track modal frequencies and mode shapes over time.
    
    Maintains a baseline set of modes and matches new mode observations
    against the baseline using MAC and frequency proximity criteria.
    """
    
    def __init__(self, mac_threshold=0.85, freq_tol=0.1, 
                 damping_tol=0.1, adaptation_rate=0.05):
        """
        Initialize the FrequencyTracker.
        
        Parameters
        ----------
        mac_threshold : float, default=0.85
            Minimum MAC value to consider a mode matched.
        freq_tol : float, default=0.1
            Maximum relative frequency difference to match modes (10%).
        damping_tol : float, default=0.1
            Maximum absolute damping difference to match modes.
        adaptation_rate : float, default=0.05
            Exponential moving average rate for baseline updates (0-1).
        """
        self.mac_threshold = mac_threshold
        self.freq_tol = freq_tol
        self.damping_tol = damping_tol
        self.adaptation_rate = adaptation_rate
        
        # Dictionary: mode_id -> list of (timestamp, frequency, damping, mode_shape)
        self.history = {}
        
        # List of reference modes: (mode_id, baseline_freq, baseline_damp, baseline_shape)
        self.reference_modes = []
        
        # Counter for assigning new mode IDs
        self.next_mode_id = 1
        
        # Initialization flag
        self.initialized = False
    
    def initialize(self, timestamp, frequencies, damping_ratios, mode_shapes):
        """
        Initialize the tracker with baseline modes.
        
        Parameters
        ----------
        timestamp : datetime or float
            Timestamp of the baseline measurement.
        frequencies : ndarray, shape (n_modes,)
            Natural frequencies of baseline modes.
        damping_ratios : ndarray, shape (n_modes,)
            Damping ratios of baseline modes.
        mode_shapes : ndarray, shape (n_sensors, n_modes)
            Mode shape matrix (columns are mode vectors).
        """
        for i, (f, d) in enumerate(zip(frequencies, damping_ratios)):
            mode_id = self.next_mode_id
            mode_shape = mode_shapes[:, i]
            
            self.reference_modes.append((mode_id, f, d, mode_shape.copy()))
            self.history[mode_id] = [(timestamp, f, d)]
            self.next_mode_id += 1
        
        self.initialized = True
    
    def update(self, timestamp, frequencies, damping_ratios, mode_shapes):
        """
        Update tracker with new modes from the latest time window.
        
        Parameters
        ----------
        timestamp : datetime or float
            Timestamp of the new measurement.
        frequencies : ndarray, shape (n_modes,)
            Natural frequencies from the latest window.
        damping_ratios : ndarray, shape (n_modes,)
            Damping ratios from the latest window.
        mode_shapes : ndarray, shape (n_sensors, n_modes)
            Mode shape matrix from the latest window.
        """
        if not self.initialized:
            self.initialize(timestamp, frequencies, damping_ratios, mode_shapes)
            return
        
        # Track which reference modes were matched
        matched_refs = set()
        
        # Try to match new modes to reference modes
        for new_idx, (new_f, new_d, new_mode) in enumerate(
            zip(frequencies, damping_ratios, mode_shapes.T)):
            
            best_match_id = None
            best_match_score = -1
            
            for ref_id, ref_f, ref_d, ref_mode in self.reference_modes:
                # Skip if already matched in this update
                if ref_id in matched_refs:
                    continue
                
                # Check frequency criterion
                freq_diff = np.abs(new_f - ref_f) / (ref_f + 1e-8)
                if freq_diff > self.freq_tol:
                    continue
                
                # Check damping criterion
                damp_diff = np.abs(new_d - ref_d)
                if damp_diff > self.damping_tol:
                    continue
                
                # Check MAC criterion
                mac_val = mac(new_mode, ref_mode)
                if mac_val >= self.mac_threshold:
                    # Score based on MAC value (higher is better)
                    if mac_val > best_match_score:
                        best_match_score = mac_val
                        best_match_id = ref_id
            
            if best_match_id is not None:
                # Match found! Track it.
                matched_refs.add(best_match_id)
                
                # Add to history
                self.history[best_match_id].append((timestamp, new_f, new_d))
                
                # Update reference mode with exponential moving average
                # (adapt to slow variations like temperature effects)
                ref_idx = next(i for i, (m_id, _, _, _) in enumerate(self.reference_modes)
                              if m_id == best_match_id)
                ref_id, ref_f, ref_d, ref_mode = self.reference_modes[ref_idx]
                
                updated_freq = (1 - self.adaptation_rate) * ref_f + self.adaptation_rate * new_f
                updated_damp = (1 - self.adaptation_rate) * ref_d + self.adaptation_rate * new_d
                updated_mode = ((1 - self.adaptation_rate) * ref_mode + 
                               self.adaptation_rate * new_mode)
                updated_mode /= np.linalg.norm(updated_mode)
                
                self.reference_modes[ref_idx] = (ref_id, updated_freq, updated_damp, updated_mode)
            else:
                # No match found - this could be a newly excited mode
                # Add it as a new reference mode
                mode_id = self.next_mode_id
                self.reference_modes.append((mode_id, new_f, new_d, new_mode.copy()))
                self.history[mode_id] = [(timestamp, new_f, new_d)]
                self.next_mode_id += 1
    
    def get_history(self, mode_id):
        """
        Get the time history for a specific mode.
        
        Parameters
        ----------
        mode_id : int
            The mode ID to retrieve.
        
        Returns
        -------
        history : list of tuples
            List of (timestamp, frequency, damping) for the mode.
        """
        return self.history.get(mode_id, [])
    
    def get_all_history(self):
        """
        Get the complete history dictionary.
        
        Returns
        -------
        history : dict
            Dictionary mapping mode_id to list of (timestamp, frequency, damping).
        """
        return self.history.copy()
    
    def get_current_modes(self):
        """
        Get current baseline modes.
        
        Returns
        -------
        modes : list of tuples
            List of (mode_id, frequency, damping, mode_shape).
        """
        return self.reference_modes.copy()
    
    def get_frequency_trends(self, mode_id):
        """
        Extract frequency trend for a mode.
        
        Parameters
        ----------
        mode_id : int
            The mode ID to analyze.
        
        Returns
        -------
        timestamps : ndarray
            Array of timestamps.
        frequencies : ndarray
            Array of frequencies at each timestamp.
        """
        if mode_id not in self.history:
            return np.array([]), np.array([])
        
        history_data = self.history[mode_id]
        timestamps = np.array([h[0] for h in history_data])
        frequencies = np.array([h[1] for h in history_data])
        
        return timestamps, frequencies
    
    def export_to_dict(self):
        """
        Export tracking data to a dictionary (for serialization).
        
        Returns
        -------
        export_dict : dict
            Dictionary containing history and metadata.
        """
        return {
            'history': self.history,
            'reference_modes': [
                {
                    'mode_id': m_id,
                    'baseline_freq': float(m_f),
                    'baseline_damp': float(m_d),
                    # mode_shape excluded due to serialization complexity
                }
                for m_id, m_f, m_d, _ in self.reference_modes
            ],
            'metadata': {
                'mac_threshold': self.mac_threshold,
                'freq_tol': self.freq_tol,
                'damping_tol': self.damping_tol,
                'adaptation_rate': self.adaptation_rate,
            }
        }


class ModalTrackingAnalyzer:
    """
    Analyze tracked modal data for SHM applications.
    
    Computes statistical summaries, detects anomalies, and generates
    reports on structural health based on frequency and damping trends.
    """
    
    def __init__(self, tracker):
        """
        Initialize the analyzer with a FrequencyTracker instance.
        
        Parameters
        ----------
        tracker : FrequencyTracker
            The tracker instance containing mode history.
        """
        self.tracker = tracker
    
    def compute_frequency_statistics(self, mode_id):
        """
        Compute statistics for a mode's frequency over time.
        
        Parameters
        ----------
        mode_id : int
            The mode ID to analyze.
        
        Returns
        -------
        stats : dict
            Dictionary containing mean, std, min, max frequencies.
        """
        timestamps, frequencies = self.tracker.get_frequency_trends(mode_id)
        
        if len(frequencies) == 0:
            return {}
        
        return {
            'mode_id': mode_id,
            'mean_freq': np.mean(frequencies),
            'std_freq': np.std(frequencies),
            'min_freq': np.min(frequencies),
            'max_freq': np.max(frequencies),
            'freq_range': np.max(frequencies) - np.min(frequencies),
            'n_observations': len(frequencies),
        }
    
    def detect_frequency_anomalies(self, mode_id, threshold=3.0):
        """
        Detect anomalous frequency measurements using z-score.
        
        Parameters
        ----------
        mode_id : int
            The mode ID to analyze.
        threshold : float, default=3.0
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
        
        if std_freq == 0:
            return []
        
        z_scores = np.abs((frequencies - mean_freq) / std_freq)
        anomalies = np.where(z_scores > threshold)[0].tolist()
        
        return anomalies
    
    def compute_trend(self, mode_id):
        """
        Compute linear trend in frequency over time.
        
        Parameters
        ----------
        mode_id : int
            The mode ID to analyze.
        
        Returns
        -------
        trend_info : dict
            Dictionary containing slope, intercept, and R² value.
        """
        timestamps, frequencies = self.tracker.get_frequency_trends(mode_id)
        
        if len(frequencies) < 2:
            return {}
        
        # Convert timestamps to numeric (seconds from start)
        if isinstance(timestamps[0], (int, float)):
            t_numeric = np.array(timestamps)
        else:
            # Assume datetime objects
            t_start = timestamps[0]
            t_numeric = np.array([(t - t_start).total_seconds() for t in timestamps])
        
        # Linear regression
        coeffs = np.polyfit(t_numeric, frequencies, 1)
        slope, intercept = coeffs
        
        # Compute R²
        y_pred = np.polyval(coeffs, t_numeric)
        ss_res = np.sum((frequencies - y_pred) ** 2)
        ss_tot = np.sum((frequencies - np.mean(frequencies)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mode_id': mode_id,
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'unit': 'Hz per second'
        }
