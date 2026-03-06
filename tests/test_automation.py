"""
Tests for the automation module.
"""

import numpy as np
import pytest
from shm_oma import mac, vectorized_mac, automate_poles_dbscan, filter_spurious_poles


class TestMAC:
    """Tests for Modal Assurance Criterion."""

    def test_mac_identical_modes(self):
        phi = np.array([1.0, 0.5, 0.3])
        assert np.isclose(mac(phi, phi), 1.0)

    def test_mac_orthogonal_modes(self):
        phi1 = np.array([1.0, 0.0, 0.0])
        phi2 = np.array([0.0, 1.0, 0.0])
        assert np.isclose(mac(phi1, phi2), 0.0)

    def test_mac_complex_modes(self):
        """MAC should work correctly with complex mode shapes."""
        phi1 = np.array([1.0 + 0.5j, 0.3 - 0.2j])
        phi2 = phi1.copy()
        assert np.isclose(mac(phi1, phi2), 1.0)

    def test_mac_range(self):
        np.random.seed(42)
        phi1 = np.random.randn(5)
        phi2 = np.random.randn(5)
        val = mac(phi1, phi2)
        assert 0.0 <= val <= 1.0


class TestVectorizedMAC:
    """Tests for vectorised MAC computation."""

    def test_diagonal_ones(self):
        modes = np.eye(3)
        mac_mat = vectorized_mac(modes)
        assert np.allclose(np.diag(mac_mat), 1.0)

    def test_matches_scalar_mac(self):
        np.random.seed(42)
        modes = np.random.randn(4, 3)
        mac_mat = vectorized_mac(modes)
        for i in range(3):
            for j in range(3):
                expected = mac(modes[:, i], modes[:, j])
                assert np.isclose(mac_mat[i, j], expected, atol=1e-8)

    def test_complex_modes_consistency(self):
        """vectorized_mac should match scalar mac for complex modes."""
        np.random.seed(42)
        modes = np.random.randn(4, 3) + 1j * np.random.randn(4, 3)
        mac_mat = vectorized_mac(modes)
        for i in range(3):
            for j in range(3):
                expected = mac(modes[:, i], modes[:, j])
                assert np.isclose(mac_mat[i, j], expected, atol=1e-8)


class TestDBSCAN:
    """Tests for DBSCAN pole clustering."""

    def test_basic_clustering(self):
        freqs = np.array([10.0, 10.01, 10.02, 20.0, 20.01, 20.02])
        damps = np.array([0.05, 0.05, 0.05, 0.03, 0.03, 0.03])
        modes = np.random.randn(3, 6)
        result = automate_poles_dbscan(freqs, damps, modes, eps_freq=0.1, min_samples=2)
        assert len(result) == 2  # two clusters

    def test_empty_input(self):
        result = automate_poles_dbscan(
            np.array([]), np.array([]), np.empty((3, 0))
        )
        assert result == []


class TestFilterSpuriousPoles:
    """Tests for multi-order stability filtering."""

    def test_stable_poles_pass_through(self):
        """Identical modes across orders should all be stable."""
        freqs = np.array([10.0, 20.0])
        damps = np.array([0.05, 0.03])
        modes = np.array([[1.0, 0.5], [0.8, 0.7], [0.6, 0.9]])

        all_f = [freqs, freqs + 0.001, freqs + 0.002]
        all_d = [damps, damps, damps]
        all_m = [modes, modes, modes]

        sf, sd, sm = filter_spurious_poles(all_f, all_d, all_m, freq_tol=0.01)
        assert len(sf) == 2

    def test_unstable_poles_removed(self):
        """Modes that jump significantly should be removed."""
        freqs1 = np.array([10.0, 20.0])
        freqs2 = np.array([10.0, 55.0])  # second mode jumps
        damps = np.array([0.05, 0.03])
        modes = np.array([[1.0, 0.5], [0.8, 0.7], [0.6, 0.9]])

        sf, sd, sm = filter_spurious_poles(
            [freqs1, freqs2], [damps, damps], [modes, modes], freq_tol=0.01
        )
        assert len(sf) == 1  # only first mode survives
        assert np.isclose(sf[0], 10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
