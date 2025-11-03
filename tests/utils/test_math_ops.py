"""Tests for mathematical operations utilities."""

import numpy as np
import pandas as pd
import pytest

from itau_quant.utils.math_ops import (
    clip_with_tolerance,
    expm1_safe,
    normalize_vector,
    project_to_simplex,
    soft_threshold,
    stable_inverse,
    weighted_norm,
)


class TestProjectToSimplex:
    """Tests for project_to_simplex function."""

    def test_projects_positive_vector_to_simplex(self):
        """Test projecting a positive vector to simplex."""
        vector = np.array([0.5, 0.3, 0.2])
        result = project_to_simplex(vector, sum_to=1.0)

        assert result.sum() == pytest.approx(1.0)
        assert np.all(result >= 0)

    def test_projects_arbitrary_vector_to_simplex(self):
        """Test projecting an arbitrary vector to simplex."""
        vector = np.array([1.0, 2.0, -1.0, 0.5])
        result = project_to_simplex(vector, sum_to=1.0)

        assert result.sum() == pytest.approx(1.0)
        assert np.all(result >= 0)

    def test_custom_sum_target(self):
        """Test projecting to simplex with custom sum."""
        vector = np.array([1.0, 2.0, 3.0])
        result = project_to_simplex(vector, sum_to=2.0)

        assert result.sum() == pytest.approx(2.0)
        assert np.all(result >= 0)

    def test_already_on_simplex(self):
        """Test that a vector already on simplex is unchanged."""
        vector = np.array([0.3, 0.5, 0.2])
        result = project_to_simplex(vector, sum_to=1.0)

        np.testing.assert_allclose(result, vector, atol=1e-8)

    def test_raises_on_2d_input(self):
        """Test that 2D input raises ValueError."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="1D"):
            project_to_simplex(matrix)


class TestSoftThreshold:
    """Tests for soft_threshold function."""

    def test_applies_soft_threshold_numpy(self):
        """Test soft thresholding on numpy array."""
        data = np.array([3.0, -2.0, 0.5, -0.3])
        lam = 1.0

        result = soft_threshold(data, lam)

        expected = np.array([2.0, -1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_applies_soft_threshold_series(self):
        """Test soft thresholding on pandas Series."""
        data = pd.Series([3.0, -2.0, 0.5], index=["a", "b", "c"])
        lam = 1.0

        result = soft_threshold(data, lam)

        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b", "c"]
        expected_values = np.array([2.0, -1.0, 0.0])
        np.testing.assert_allclose(result.values, expected_values)

    def test_applies_soft_threshold_dataframe(self):
        """Test soft thresholding on DataFrame."""
        data = pd.DataFrame({"A": [3.0, -2.0], "B": [1.5, -1.5]})
        lam = 1.0

        result = soft_threshold(data, lam)

        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame({"A": [2.0, -1.0], "B": [0.5, -0.5]})
        pd.testing.assert_frame_equal(result, expected)

    def test_zero_lambda(self):
        """Test that lambda=0 returns unchanged data."""
        data = np.array([1.0, -2.0, 3.0])
        result = soft_threshold(data, lam=0.0)

        np.testing.assert_allclose(result, data)

    def test_raises_on_negative_lambda(self):
        """Test that negative lambda raises ValueError."""
        data = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="não-negativo"):
            soft_threshold(data, lam=-1.0)


class TestNormalizeVector:
    """Tests for normalize_vector function."""

    def test_normalizes_with_l2_norm(self):
        """Test L2 normalization."""
        vector = np.array([3.0, 4.0])
        result = normalize_vector(vector, norm="l2")

        expected = np.array([0.6, 0.8])  # 3/5, 4/5
        np.testing.assert_allclose(result, expected)

    def test_normalizes_with_l1_norm(self):
        """Test L1 normalization."""
        vector = np.array([2.0, 3.0, 5.0])
        result = normalize_vector(vector, norm="l1")

        expected = np.array([0.2, 0.3, 0.5])  # divided by 10
        np.testing.assert_allclose(result, expected)

    def test_normalizes_with_max_norm(self):
        """Test max normalization."""
        vector = np.array([1.0, 2.0, -3.0])
        result = normalize_vector(vector, norm="max")

        expected = np.array([1/3, 2/3, -1.0])
        np.testing.assert_allclose(result, expected)

    def test_normalizes_series(self):
        """Test normalization of pandas Series."""
        vector = pd.Series([3.0, 4.0], index=["a", "b"])
        result = normalize_vector(vector, norm="l2")

        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b"]
        np.testing.assert_allclose(result.values, [0.6, 0.8])

    def test_normalizes_dataframe_columns(self):
        """Test normalization of DataFrame columns."""
        df = pd.DataFrame({"A": [3.0, 4.0], "B": [1.0, 0.0]})
        result = normalize_vector(df, norm="l2")

        assert isinstance(result, pd.DataFrame)
        # Column A: [3/5, 4/5], Column B: [1, 0]
        expected = pd.DataFrame({"A": [0.6, 0.8], "B": [1.0, 0.0]})
        pd.testing.assert_frame_equal(result, expected, atol=1e-10)

    def test_handles_zero_vector(self):
        """Test that zero vectors don't cause division by zero."""
        vector = np.array([0.0, 0.0])
        result = normalize_vector(vector, norm="l2")

        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_raises_on_invalid_norm(self):
        """Test that invalid norm type raises ValueError."""
        vector = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Norma deve ser"):
            normalize_vector(vector, norm="l3")


class TestWeightedNorm:
    """Tests for weighted_norm function."""

    def test_computes_weighted_l2_norm(self):
        """Test weighted L2 norm."""
        vector = np.array([3.0, 4.0])
        weights = np.array([1.0, 2.0])

        result = weighted_norm(vector, weights, order=2)

        # sqrt(1*3^2 + 2*4^2) = sqrt(9 + 32) = sqrt(41)
        expected = np.sqrt(41)
        assert result == pytest.approx(expected)

    def test_computes_weighted_l1_norm(self):
        """Test weighted L1 norm."""
        vector = np.array([2.0, 3.0])
        weights = np.array([1.0, 2.0])

        result = weighted_norm(vector, weights, order=1)

        # (1*2 + 2*3)^1 = 8
        expected = 8.0
        assert result == pytest.approx(expected)

    def test_works_with_series(self):
        """Test with pandas Series."""
        vector = pd.Series([1.0, 2.0, 3.0])
        weights = pd.Series([0.5, 1.0, 1.5])

        result = weighted_norm(vector, weights, order=2)

        # sqrt(0.5*1 + 1.0*4 + 1.5*9) = sqrt(0.5 + 4 + 13.5) = sqrt(18)
        expected = np.sqrt(18)
        assert result == pytest.approx(expected)

    def test_raises_on_shape_mismatch(self):
        """Test that mismatched shapes raise ValueError."""
        vector = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="mesmo formato"):
            weighted_norm(vector, weights, order=2)


class TestClipWithTolerance:
    """Tests for clip_with_tolerance function."""

    def test_clips_values_outside_bounds(self):
        """Test clipping values outside bounds."""
        vector = np.array([-1.0, 0.5, 2.0])
        result = clip_with_tolerance(vector, lower=0.0, upper=1.0)

        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_handles_values_within_tolerance(self):
        """Test that values within tolerance are adjusted."""
        vector = np.array([1e-10, 0.5, 1.0 - 1e-10])
        result = clip_with_tolerance(vector, lower=0.0, upper=1.0, tol=1e-9)

        assert result[0] == 0.0
        assert result[1] == pytest.approx(0.5)
        assert result[2] == 1.0

    def test_works_with_series(self):
        """Test clipping with pandas Series."""
        vector = pd.Series([-0.5, 0.5, 1.5], index=["a", "b", "c"])
        result = clip_with_tolerance(vector, lower=0.0, upper=1.0)

        assert isinstance(result, pd.Series)
        expected = pd.Series([0.0, 0.5, 1.0], index=["a", "b", "c"])
        pd.testing.assert_series_equal(result, expected)

    def test_works_with_dataframe(self):
        """Test clipping with DataFrame."""
        df = pd.DataFrame({"A": [-1.0, 0.5], "B": [0.3, 2.0]})
        result = clip_with_tolerance(df, lower=0.0, upper=1.0)

        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame({"A": [0.0, 0.5], "B": [0.3, 1.0]})
        pd.testing.assert_frame_equal(result, expected)


class TestStableInverse:
    """Tests for stable_inverse function."""

    def test_inverts_well_conditioned_matrix(self):
        """Test inversion of well-conditioned matrix."""
        matrix = np.array([[4.0, 2.0], [2.0, 3.0]])
        result = stable_inverse(matrix, ridge=1e-8)

        # Verify A * A^-1 ≈ I
        identity = np.dot(matrix, result)
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-6)

    def test_stabilizes_near_singular_matrix(self):
        """Test that ridge helps with near-singular matrices."""
        # Create nearly singular matrix
        matrix = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]])

        result = stable_inverse(matrix, ridge=1e-3)

        assert result.shape == (2, 2)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_raises_on_non_square_matrix(self):
        """Test that non-square matrix raises ValueError."""
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(ValueError, match="quadrada"):
            stable_inverse(matrix)

    def test_ridge_effect(self):
        """Test that larger ridge produces more stable inverse."""
        matrix = np.array([[1.0, 0.9], [0.9, 1.0]])

        inv_small = stable_inverse(matrix, ridge=1e-10)
        inv_large = stable_inverse(matrix, ridge=0.1)

        # Larger ridge should produce smaller condition number
        cond_small = np.linalg.cond(inv_small)
        cond_large = np.linalg.cond(inv_large)

        assert cond_large < cond_small


class TestExpm1Safe:
    """Tests for expm1_safe function."""

    def test_computes_expm1_numpy(self):
        """Test expm1 on numpy array."""
        data = np.array([0.0, 1.0, -1.0])
        result = expm1_safe(data)

        expected = np.expm1(data)
        np.testing.assert_allclose(result, expected)

    def test_computes_expm1_series(self):
        """Test expm1 on pandas Series."""
        data = pd.Series([0.0, 0.1, -0.1], index=["a", "b", "c"])
        result = expm1_safe(data)

        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b", "c"]
        expected = np.expm1(data.values)
        np.testing.assert_allclose(result.values, expected)

    def test_computes_expm1_dataframe(self):
        """Test expm1 on DataFrame."""
        data = pd.DataFrame({"A": [0.0, 1.0], "B": [-0.5, 0.5]})
        result = expm1_safe(data)

        assert isinstance(result, pd.DataFrame)
        expected = np.expm1(data.values)
        np.testing.assert_allclose(result.values, expected)

    def test_handles_small_values_accurately(self):
        """Test that small values are handled accurately."""
        # expm1 is more accurate than exp(x)-1 for small x
        data = np.array([1e-10, 1e-8, 1e-6])
        result = expm1_safe(data)

        # For small x, exp(x) - 1 ≈ x
        np.testing.assert_allclose(result, data, rtol=1e-6)

    def test_handles_large_negative_values(self):
        """Test handling of large negative values."""
        data = np.array([-10.0, -5.0, -1.0])
        result = expm1_safe(data)

        # exp(-10) - 1 ≈ -1 for large negative values
        assert np.all(result < 0)
        assert np.all(result >= -1)
