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
        frequencies = np.array([10.5, 25.3])
        damping_ratios = np.array([0.05, 0.03])
        mode_shapes = np.array([
            [1.0, 0.5],
            [0.8, 0.7],
            [0.6, 0.9],
        ])
        return frequencies, damping_ratios, mode_shapes

    def test_tracker_initialization(self, sample_modes):
        tracker = FrequencyTracker()
        freqs, damps, modes = sample_modes
        tracker.initialize(datetime.now(), freqs, damps, modes)

        assert tracker.initialized
        assert len(tracker.reference_modes) == 2
        assert tracker.next_mode_id == 3

    def test_tracker_history_recording(self, sample_modes):
        tracker = FrequencyTracker()
        freqs, damps, modes = sample_modes
        tracker.initialize(datetime.now(), freqs, damps, modes)

        assert len(tracker.history) == 2
        assert 1 in tracker.history
        assert 2 in tracker.history
        assert tracker.history[1][0][1] == freqs[0]
        assert tracker.history[1][0][2] == damps[0]

    def test_tracker_update_with_match(self, sample_modes):
        tracker = FrequencyTracker(mac_threshold=0.85, freq_tol=0.15)
        freqs, damps, modes = sample_modes
        t1 = datetime.now()
        tracker.initialize(t1, freqs, damps, modes)

        t2 = t1 + timedelta(seconds=60)
        tracker.update(t2, freqs + 0.1, damps.copy(), modes.copy())

        assert len(tracker.history[1]) == 2
        assert len(tracker.history[2]) == 2

    def test_tracker_new_mode_detection(self, sample_modes):
        tracker = FrequencyTracker(mac_threshold=0.95, freq_tol=0.01)
        freqs, damps, modes = sample_modes
        t1 = datetime.now()
        tracker.initialize(t1, freqs, damps, modes)
        initial_count = len(tracker.reference_modes)

        t2 = t1 + timedelta(seconds=60)
        new_freqs = np.array([50.0, 60.0])
        new_damps = np.array([0.01, 0.02])
        new_modes = np.random.randn(3, 2)
        tracker.update(t2, new_freqs, new_damps, new_modes)

        assert len(tracker.reference_modes) > initial_count

    def test_tracker_get_frequency_trends(self, sample_modes):
        tracker = FrequencyTracker()
        freqs, damps, modes = sample_modes
        t1 = datetime.now()
        tracker.initialize(t1, freqs, damps, modes)

        t2 = t1 + timedelta(seconds=60)
        tracker.update(t2, freqs + 0.05, damps, modes)

        timestamps, frequencies = tracker.get_frequency_trends(1)
        assert len(timestamps) == 2
        assert len(frequencies) == 2
        assert frequencies[0] == freqs[0]
        assert frequencies[1] > freqs[0]

    def test_tracker_export_to_dict(self, sample_modes):
        tracker = FrequencyTracker()
        freqs, damps, modes = sample_modes
        tracker.initialize(datetime.now(), freqs, damps, modes)

        export = tracker.export_to_dict()
        assert "history" in export
        assert "reference_modes" in export
        assert "metadata" in export
        assert len(export["reference_modes"]) == 2

    def test_freeze_baseline(self, sample_modes):
        """When baseline is frozen, reference modes should not change."""
        tracker = FrequencyTracker(mac_threshold=0.5, freq_tol=0.5)
        freqs, damps, modes = sample_modes
        t1 = datetime.now()
        tracker.initialize(t1, freqs, damps, modes)

        tracker.freeze_baseline()
        original_ref_freq = tracker.reference_modes[0][1]

        t2 = t1 + timedelta(seconds=60)
        tracker.update(t2, freqs + 0.5, damps, modes)

        # Frequency in the reference should NOT have changed
        assert tracker.reference_modes[0][1] == original_ref_freq

    def test_unfreeze_baseline(self, sample_modes):
        tracker = FrequencyTracker(mac_threshold=0.5, freq_tol=0.5)
        freqs, damps, modes = sample_modes
        t1 = datetime.now()
        tracker.initialize(t1, freqs, damps, modes)

        tracker.freeze_baseline()
        tracker.unfreeze_baseline()

        original_ref_freq = tracker.reference_modes[0][1]
        t2 = t1 + timedelta(seconds=60)
        tracker.update(t2, freqs + 0.5, damps, modes)

        # Now it SHOULD have adapted
        assert tracker.reference_modes[0][1] != original_ref_freq

    def test_reset_baseline(self, sample_modes):
        tracker = FrequencyTracker()
        freqs, damps, modes = sample_modes
        t1 = datetime.now()
        tracker.initialize(t1, freqs, damps, modes)

        new_freqs = np.array([40.0])
        new_damps = np.array([0.02])
        new_modes = np.random.randn(3, 1)
        tracker.reset_baseline(t1, new_freqs, new_damps, new_modes)

        assert len(tracker.reference_modes) == 1
        assert tracker.reference_modes[0][1] == 40.0


class TestModalTrackingAnalyzer:
    """Test suite for ModalTrackingAnalyzer."""

    @pytest.fixture
    def tracker_with_data(self):
        tracker = FrequencyTracker()
        freqs = np.array([10.0, 25.0])
        damps = np.array([0.05, 0.03])
        modes = np.array([[1.0, 0.5], [0.8, 0.7], [0.6, 0.9]])

        t0 = datetime(2026, 3, 5, 0, 0, 0)
        tracker.initialize(t0, freqs, damps, modes)

        for i in range(1, 11):
            t = t0 + timedelta(hours=i)
            perturbed = freqs + 0.05 * np.sin(2 * np.pi * i / 10)
            tracker.update(t, perturbed, damps, modes)

        return tracker

    def test_compute_frequency_statistics(self, tracker_with_data):
        analyzer = ModalTrackingAnalyzer(tracker_with_data)
        stats = analyzer.compute_frequency_statistics(1)

        assert "mean_freq" in stats
        assert "std_freq" in stats
        assert stats["n_observations"] == 11

    def test_compute_trend(self, tracker_with_data):
        analyzer = ModalTrackingAnalyzer(tracker_with_data)
        trend = analyzer.compute_trend(1)

        assert "slope" in trend
        assert "r_squared" in trend
        assert trend["mode_id"] == 1

    def test_detect_frequency_anomalies_finds_outlier(self, tracker_with_data):
        """Injecting a clear outlier should be detected."""
        analyzer = ModalTrackingAnalyzer(tracker_with_data)

        # Inject a large outlier
        tracker_with_data.history[1].append(
            (datetime(2026, 3, 15, 0, 0, 0), 50.0, 0.05)
        )

        anomalies = analyzer.detect_frequency_anomalies(1, threshold=2.0)
        assert len(anomalies) > 0, "Injected outlier at 50 Hz should be detected"

    def test_detect_frequency_anomalies_no_false_positives(self, tracker_with_data):
        """Normal data should have no anomalies at a high threshold."""
        analyzer = ModalTrackingAnalyzer(tracker_with_data)
        anomalies = analyzer.detect_frequency_anomalies(1, threshold=10.0)
        assert len(anomalies) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
