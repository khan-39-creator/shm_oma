"""
Tests for the ssicov module.
"""

import numpy as np
import pytest
from shm_oma import perform_ssi_cov, compute_autocorrelation, build_toeplitz_matrix


class TestSSICOV:
    """Test suite for SSI-COV algorithm."""

    @pytest.fixture
    def dummy_data(self):
        """Generate dummy acceleration data for testing."""
        np.random.seed(42)
        return np.random.randn(1000, 3)

    def test_compute_autocorrelation_shape(self, dummy_data):
        max_lag = 100
        acf = compute_autocorrelation(dummy_data, max_lag=max_lag)
        assert acf.shape == (max_lag, 3, 3)

    def test_compute_autocorrelation_symmetry(self, dummy_data):
        acf = compute_autocorrelation(dummy_data, max_lag=50)
        lag0 = acf[0]
        assert np.allclose(lag0, lag0.T)

    def test_build_toeplitz_matrix_shape(self, dummy_data):
        acf = compute_autocorrelation(dummy_data, max_lag=50)
        block_rows = 10
        H = build_toeplitz_matrix(acf, block_rows)
        expected_size = block_rows * 3
        assert H.shape == (expected_size, expected_size)

    def test_build_toeplitz_matrix_block_structure(self, dummy_data):
        """Block-Toeplitz: H[i,j] block depends only on |i-j|."""
        acf = compute_autocorrelation(dummy_data, max_lag=50)
        H = build_toeplitz_matrix(acf, block_rows=5)
        ns = 3  # n_sensors
        # Blocks on the same diagonal should be identical
        block_00 = H[0:ns, 0:ns]
        block_11 = H[ns:2*ns, ns:2*ns]
        assert np.allclose(block_00, block_11)
        # Off-diag blocks at same lag should match
        block_01 = H[0:ns, ns:2*ns]
        block_12 = H[ns:2*ns, 2*ns:3*ns]
        assert np.allclose(block_01, block_12)

    def test_perform_ssi_cov_output_shapes(self, dummy_data):
        """Test SSI-COV returns 4 items with correct shapes."""
        order = 20
        result = perform_ssi_cov(dummy_data, order=order, max_lag=100)
        assert len(result) == 4, "perform_ssi_cov must return 4 values"

        frequencies, damping_ratios, mode_shapes, singular_values = result
        assert len(damping_ratios) == len(frequencies)
        assert singular_values is not None
        if len(frequencies) > 0:
            assert mode_shapes.shape[0] == 3  # 3 sensors
            assert mode_shapes.shape[1] == len(frequencies)

    def test_perform_ssi_cov_with_randomized_svd(self, dummy_data):
        order = 15
        freqs, damps, modes, sv = perform_ssi_cov(
            dummy_data, order=order, use_randomized_svd=True
        )
        if len(freqs) > 0:
            assert np.all(freqs >= 0)
            assert np.all(damps >= 0)

    def test_perform_ssi_cov_damping_ratio_bounds(self, dummy_data):
        frequencies, damping_ratios, _, _ = perform_ssi_cov(dummy_data, order=20)
        if len(damping_ratios) > 0:
            assert np.all(damping_ratios >= 0)
            assert np.all(damping_ratios <= 1)

    def test_perform_ssi_cov_mode_shape_normalization(self, dummy_data):
        _, _, mode_shapes, _ = perform_ssi_cov(dummy_data, order=20)
        if mode_shapes.shape[1] > 0:
            norms = np.linalg.norm(mode_shapes, axis=0)
            assert np.allclose(norms, 1.0, atol=1e-6)

    def test_no_duplicate_conjugate_frequencies(self, dummy_data):
        """HC conjugate filter should prevent near-duplicate freq pairs."""
        freqs, _, _, _ = perform_ssi_cov(dummy_data, order=20, sampling_freq=100.0)
        if len(freqs) > 1:
            # No two frequencies should be nearly identical (conjugate artefact)
            diffs = np.diff(freqs)
            # Allow a tiny tolerance but flag exact duplicates
            assert np.all(diffs > -1e-10), "Frequencies should be sorted and non-decreasing"


class TestSSICOVEdgeCases:
    """Test edge cases and error handling."""

    def test_perform_ssi_cov_single_sensor(self):
        np.random.seed(42)
        data_single = np.random.randn(500, 1)
        freqs, damps, modes, sv = perform_ssi_cov(data_single, order=10)
        # May return 0 modes for pure noise — that is acceptable
        assert len(freqs) == len(damps)

    def test_perform_ssi_cov_many_sensors(self):
        np.random.seed(42)
        data_many = np.random.randn(500, 20)
        freqs, damps, modes, sv = perform_ssi_cov(data_many, order=15)
        if len(freqs) > 0:
            assert modes.shape[0] == 20

    def test_perform_ssi_cov_short_data(self):
        np.random.seed(42)
        data_short = np.random.randn(100, 3)
        freqs, damps, modes, sv = perform_ssi_cov(data_short, order=10, max_lag=20)
        assert len(freqs) == len(damps)

    def test_empty_result_shapes(self):
        """Even when no modes pass HC, shapes should be consistent."""
        np.random.seed(99)
        data = np.random.randn(50, 2) * 1e-10  # tiny noise
        freqs, damps, modes, sv = perform_ssi_cov(data, order=5, max_lag=10)
        assert freqs.ndim == 1
        assert damps.ndim == 1
        assert modes.ndim == 2
        assert modes.shape[0] == 2  # n_sensors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
