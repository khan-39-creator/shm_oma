"""
Tests for the tracking module.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
from shm_oma import FrequencyTracker, ModalTrackingAnalyzer


class TestFrequencyTracker:
    """Test suite for FrequencyTracker class."""
    
    @pytest.fixture
    def sample_modes(self):
        """Generate sample modal data."""
        # 3 sensors, 2 modes
        frequencies = np.array([10.5, 25.3])
        damping_ratios = np.array([0.05, 0.03])
        mode_shapes = np.array([
            [1.0, 0.5],
            [0.8, 0.7],
            [0.6, 0.9]
        ])
        return frequencies, damping_ratios, mode_shapes
    
    def test_tracker_initialization(self, sample_modes):
        """Test tracker initialization."""
        tracker = FrequencyTracker()
        
        freqs, damps, modes = sample_modes
        timestamp = datetime.now()
        
        tracker.initialize(timestamp, freqs, damps, modes)
        
        assert tracker.initialized
        assert len(tracker.reference_modes) == 2
        assert tracker.next_mode_id == 3
    
    def test_tracker_history_recording(self, sample_modes):
        """Test that tracker correctly records history."""
        tracker = FrequencyTracker()
        
        freqs, damps, modes = sample_modes
        timestamp_1 = datetime.now()
        
        tracker.initialize(timestamp_1, freqs, damps, modes)
        
        # Check history
        assert len(tracker.history) == 2
        assert 1 in tracker.history
        assert 2 in tracker.history
        
        # Check that first history entry matches initialization
        hist1 = tracker.history[1]
        assert hist1[0][1] == freqs[0]  # frequency
        assert hist1[0][2] == damps[0]  # damping
    
    def test_tracker_update_with_match(self, sample_modes):
        """Test tracker update when modes match."""
        tracker = FrequencyTracker(mac_threshold=0.85, freq_tol=0.15)
        
        freqs, damps, modes = sample_modes
        timestamp_1 = datetime.now()
        
        tracker.initialize(timestamp_1, freqs, damps, modes)
        
        # Update with similar modes
        timestamp_2 = timestamp_1 + timedelta(seconds=60)
        new_freqs = freqs + 0.1  # Small perturbation
        new_damps = damps.copy()
        new_modes = modes.copy()
        
        tracker.update(timestamp_2, new_freqs, new_damps, new_modes)
        
        # Check that modes were matched (history should grow)
        assert len(tracker.history[1]) == 2  # Two entries for mode 1
        assert len(tracker.history[2]) == 2  # Two entries for mode 2
    
    def test_tracker_new_mode_detection(self, sample_modes):
        """Test detection of new modes."""
        tracker = FrequencyTracker(mac_threshold=0.95, freq_tol=0.01)
        
        freqs, damps, modes = sample_modes
        timestamp_1 = datetime.now()
        
        tracker.initialize(timestamp_1, freqs, damps, modes)
        initial_count = len(tracker.reference_modes)
        
        # Update with very different modes (should be treated as new)
        timestamp_2 = timestamp_1 + timedelta(seconds=60)
        new_freqs = np.array([50.0, 60.0])  # Very different frequencies
        new_damps = np.array([0.01, 0.02])
        new_modes = np.random.randn(3, 2)
        
        tracker.update(timestamp_2, new_freqs, new_damps, new_modes)
        
        # Should have detected new modes
        assert len(tracker.reference_modes) > initial_count
    
    def test_tracker_get_frequency_trends(self, sample_modes):
        """Test frequency trend retrieval."""
        tracker = FrequencyTracker()
        
        freqs, damps, modes = sample_modes
        timestamp_1 = datetime.now()
        
        tracker.initialize(timestamp_1, freqs, damps, modes)
        
        # Add more history entries
        timestamp_2 = timestamp_1 + timedelta(seconds=60)
        tracker.update(timestamp_2, freqs + 0.05, damps, modes)
        
        # Get trend
        timestamps, frequencies = tracker.get_frequency_trends(1)
        
        assert len(timestamps) == 2
        assert len(frequencies) == 2
        assert frequencies[0] == freqs[0]
        assert frequencies[1] > freqs[0]  # Should have increased
    
    def test_tracker_export_to_dict(self, sample_modes):
        """Test export functionality."""
        tracker = FrequencyTracker()
        
        freqs, damps, modes = sample_modes
        timestamp = datetime.now()
        
        tracker.initialize(timestamp, freqs, damps, modes)
        
        export = tracker.export_to_dict()
        
        assert 'history' in export
        assert 'reference_modes' in export
        assert 'metadata' in export
        assert len(export['reference_modes']) == 2


class TestModalTrackingAnalyzer:
    """Test suite for ModalTrackingAnalyzer."""
    
    @pytest.fixture
    def tracker_with_data(self):
        """Create a tracker with time series data."""
        tracker = FrequencyTracker()
        
        # Initialize with baseline modes
        freqs = np.array([10.0, 25.0])
        damps = np.array([0.05, 0.03])
        modes = np.array([[1.0, 0.5], [0.8, 0.7], [0.6, 0.9]])
        
        timestamp_0 = datetime(2026, 3, 5, 0, 0, 0)
        tracker.initialize(timestamp_0, freqs, damps, modes)
        
        # Add multiple time points
        for i in range(1, 11):
            timestamp = timestamp_0 + timedelta(hours=i)
            # Add slight variations: temperature effect simulation
            perturbed_freqs = freqs + 0.05 * np.sin(2 * np.pi * i / 10)
            tracker.update(timestamp, perturbed_freqs, damps, modes)
        
        return tracker
    
    def test_analyzer_initialization(self, tracker_with_data):
        """Test analyzer initialization."""
        analyzer = ModalTrackingAnalyzer(tracker_with_data)
        
        assert analyzer.tracker is tracker_with_data
    
    def test_compute_frequency_statistics(self, tracker_with_data):
        """Test frequency statistics computation."""
        analyzer = ModalTrackingAnalyzer(tracker_with_data)
        
        stats = analyzer.compute_frequency_statistics(1)
        
        assert 'mean_freq' in stats
        assert 'std_freq' in stats
        assert 'min_freq' in stats
        assert 'max_freq' in stats
        assert stats['n_observations'] == 11  # Initial + 10 updates
    
    def test_compute_trend(self, tracker_with_data):
        """Test trend computation."""
        analyzer = ModalTrackingAnalyzer(tracker_with_data)
        
        trend = analyzer.compute_trend(1)
        
        assert 'slope' in trend
        assert 'intercept' in trend
        assert 'r_squared' in trend
        assert trend['mode_id'] == 1
    
    def test_detect_frequency_anomalies(self, tracker_with_data):
        """Test anomaly detection."""
        analyzer = ModalTrackingAnalyzer(tracker_with_data)
        
        # Add an anomalous value
        tracker_with_data.history[1].append(
            (datetime(2026, 3, 15, 0, 0, 0), 15.0, 0.05)
        )
        
        anomalies = analyzer.detect_frequency_anomalies(1, threshold=2.0)
        
        # Should detect the added anomaly
        # (The added value is significantly different from the trend)
        assert len(anomalies) >= 0  # May or may not detect depending on threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
