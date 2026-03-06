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
        # 1000 samples, 3 sensors
        return np.random.randn(1000, 3)
    
    def test_compute_autocorrelation_shape(self, dummy_data):
        """Test autocorrelation computation output shape."""
        max_lag = 100
        acf = compute_autocorrelation(dummy_data, max_lag=max_lag)
        
        assert acf.shape == (max_lag, 3, 3)
    
    def test_compute_autocorrelation_symmetry(self, dummy_data):
        """Test that autocorrelation matrix is symmetric."""
        acf = compute_autocorrelation(dummy_data, max_lag=50)
        
        # Check symmetry at lag 0
        lag0 = acf[0]
        assert np.allclose(lag0, lag0.T)
    
    def test_build_toeplitz_matrix_shape(self, dummy_data):
        """Test block-Toeplitz matrix dimensions."""
        acf = compute_autocorrelation(dummy_data, max_lag=50)
        block_rows = 10
        H = build_toeplitz_matrix(acf, block_rows)
        
        expected_size = block_rows * 3  # 3 sensors
        assert H.shape == (expected_size, expected_size)
    
    def test_build_toeplitz_matrix_symmetry(self, dummy_data):
        """Test that block-Toeplitz matrix is symmetric."""
        acf = compute_autocorrelation(dummy_data, max_lag=50)
        H = build_toeplitz_matrix(acf, block_rows=5)
        
        assert np.allclose(H, H.T)
    
    def test_perform_ssi_cov_output_shapes(self, dummy_data):
        """Test SSI-COV function output shapes."""
        order = 20
        frequencies, damping_ratios, mode_shapes = perform_ssi_cov(
            dummy_data, order=order, max_lag=100
        )
        
        assert len(frequencies) > 0
        assert len(damping_ratios) == len(frequencies)
        assert mode_shapes.shape[0] == 3  # 3 sensors
        assert mode_shapes.shape[1] == len(frequencies)
    
    def test_perform_ssi_cov_with_randomized_svd(self, dummy_data):
        """Test SSI-COV with randomized SVD."""
        order = 15
        freqs, damps, modes = perform_ssi_cov(
            dummy_data, order=order, use_randomized_svd=True
        )
        
        assert len(freqs) > 0
        assert np.all(freqs >= 0)  # Frequencies should be non-negative
        assert np.all(damps >= 0)  # Damping should be non-negative
    
    def test_perform_ssi_cov_damping_ratio_bounds(self, dummy_data):
        """Test that damping ratios are physically reasonable."""
        frequencies, damping_ratios, _ = perform_ssi_cov(
            dummy_data, order=20
        )
        
        # Damping ratios should be between 0 and 1 for stable systems
        assert np.all(damping_ratios >= 0)
        assert np.all(damping_ratios <= 1)
    
    def test_perform_ssi_cov_mode_shape_normalization(self, dummy_data):
        """Test that mode shapes are normalized."""
        _, _, mode_shapes = perform_ssi_cov(dummy_data, order=20)
        
        # Check that each column (mode shape) has unit norm or close to it
        norms = np.linalg.norm(mode_shapes, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-6)


class TestSSICOVEdgeCases:
    """Test edge cases and error handling."""
    
    def test_perform_ssi_cov_single_sensor(self):
        """Test SSI-COV with single sensor data."""
        np.random.seed(42)
        data_single = np.random.randn(500, 1)
        
        frequencies, damping, modes = perform_ssi_cov(
            data_single, order=10
        )
        
        assert len(frequencies) > 0
        assert modes.shape[0] == 1
    
    def test_perform_ssi_cov_many_sensors(self):
        """Test SSI-COV with many sensors."""
        np.random.seed(42)
        data_many = np.random.randn(500, 20)
        
        frequencies, damping, modes = perform_ssi_cov(
            data_many, order=15
        )
        
        assert len(frequencies) > 0
        assert modes.shape[0] == 20
    
    def test_perform_ssi_cov_short_data(self):
        """Test SSI-COV with short data."""
        np.random.seed(42)
        data_short = np.random.randn(100, 3)
        
        frequencies, damping, modes = perform_ssi_cov(
            data_short, order=10, max_lag=20
        )
        
        # Should still return results
        assert len(frequencies) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
