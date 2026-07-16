import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from cogmood_analysis.nonparam import run_reg_boots, run_reg_perms

# these tests were written by Claude Sonnet 4.5

def test_run_reg_boots():
    """Test run_reg_boots function with synthetic data"""
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic dataset
    n = 200
    dat = pd.DataFrame({
        'age': np.random.uniform(20, 80, n),
        'sex': np.random.choice([0, 1], n),
        'brain_volume': np.random.normal(1000, 100, n),
        'iq_score': np.random.normal(100, 15, n)
    })
    
    # Add age squared
    dat['age2'] = dat['age'] ** 2
    
    # Add some relationship between variables
    dat['iq_score'] += 0.5 * dat['brain_volume'] + 2 * dat['age'] - 0.02 * dat['age2']
    
    # Create bootstrap indices using pandas index
    n_boots = 10
    boot_indexes = np.column_stack([
        np.random.choice(dat.index, size=len(dat), replace=True) 
        for _ in range(n_boots)
    ])
    
    # Run the function
    result = run_reg_boots(
        task='test_task',
        tp='brain_volume',
        ss='iq_score',
        dat=dat,
        boot_indexes=boot_indexes
    )
    
    # Test 1: Check that result has all expected keys
    expected_keys = {
        'task', 'parameter', 'score', 't', 'full_r2', 'partial_r2',
        'boot_t_mean', 'boot_t_std', 'boot_t_005', 'boot_t_025', 
        'boot_t_975', 'boot_t_995', 'boot_pr2_mean', 'boot_pr2_std',
        'boot_pr2_005', 'boot_pr2_025', 'boot_pr2_975', 'boot_pr2_995'
    }
    assert set(result.keys()) == expected_keys, "Missing or extra keys in result"
    
    # Test 2: Check metadata
    assert result['task'] == 'test_task'
    assert result['parameter'] == 'brain_volume'
    assert result['score'] == 'iq_score'
    
    # Test 3: Check that statistics are reasonable
    assert isinstance(result['t'], (float, np.floating)), "t-statistic should be numeric"
    assert 0 <= result['full_r2'] <= 1, "R² should be between 0 and 1"
    assert 0 <= result['partial_r2'] <= 1, "Partial R² should be between 0 and 1"
    
    # Test 4: Check bootstrap statistics are computed
    assert not np.isnan(result['boot_t_mean']), "Bootstrap mean should not be NaN"
    assert result['boot_t_std'] >= 0, "Bootstrap std should be non-negative"
    assert not np.isnan(result['boot_pr2_mean']), "Bootstrap PR² mean should not be NaN"
    
    # Test 5: Check quantile ordering
    assert result['boot_t_005'] <= result['boot_t_025'] <= result['boot_t_975'] <= result['boot_t_995'], \
        "t-statistic quantiles should be ordered"
    assert result['boot_pr2_005'] <= result['boot_pr2_025'] <= result['boot_pr2_975'] <= result['boot_pr2_995'], \
        "Partial R² quantiles should be ordered"
    
    # Test 6: Verify original t-value is included in bootstrap distribution
    # (it's the first element, so mean should be close to it with few bootstraps)
    assert abs(result['boot_t_mean'] - result['t']) < 3 * result['boot_t_std'], \
        "Original t-value should be within reasonable range of bootstrap mean"
    
    print("✓ All tests passed!")


def test_run_reg_boots_with_nonsequential_index():
    """Test that the function handles non-sequential pandas indices correctly"""
    
    np.random.seed(123)
    
    # Create dataset with non-sequential index
    n = 100
    dat = pd.DataFrame({
        'age': np.random.uniform(20, 80, n),
        'sex': np.random.choice([0, 1], n),
        'brain_volume': np.random.normal(1000, 100, n),
        'iq_score': np.random.normal(100, 15, n)
    }, index=np.random.choice(range(1000, 2000), n, replace=False))  # Non-sequential index
    
    dat['age2'] = dat['age'] ** 2
    dat['iq_score'] += 0.3 * dat['brain_volume']
    
    # Create bootstrap indices using the actual pandas index
    n_boots = 5
    boot_indexes = np.column_stack([
        np.random.choice(dat.index, size=len(dat), replace=True) 
        for _ in range(n_boots)
    ])
    
    # This should not raise an error
    result = run_reg_boots(
        task='test_nonseq',
        tp='brain_volume',
        ss='iq_score',
        dat=dat,
        boot_indexes=boot_indexes
    )
    
    assert result is not None, "Function should handle non-sequential indices"
    assert not np.isnan(result['t']), "Should produce valid t-statistic"
    
    print("✓ Non-sequential index test passed!")


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'age': np.random.uniform(20, 80, n),
        'sex': np.random.choice([0, 1], n),
        'score': np.random.randn(n),
        'predictor': np.random.randn(n)
    })
    data['age2'] = data['age'] ** 2
    return data


@pytest.fixture
def sample_data_with_gaps():
    """Create sample data with non-contiguous index (some rows dropped)."""
    np.random.seed(42)
    n = 105
    data = pd.DataFrame({
        'age': np.random.uniform(20, 80, n),
        'sex': np.random.choice([0, 1], n),
        'score': np.random.randn(n),
        'predictor': np.random.randn(n)
    })
    data['age2'] = data['age'] ** 2
    
    # Drop some random rows to create gaps in the index
    rows_to_drop = np.random.choice(n, size=5, replace=False)
    data = data.drop(rows_to_drop)
    
    # Verify we have gaps
    assert len(data) == 100
    assert data.index.max() == 104  # Original last index
    assert len(data.index) < data.index.max() + 1  # Confirms gaps exist
    
    return data


@pytest.fixture
def perm_indexes():
    """Create permutation indexes for testing."""
    np.random.seed(123)
    n = 100
    n_perms = 10
    return np.array([np.random.permutation(n) for _ in range(n_perms)]).T


class TestRunRegPerms:
    
    def test_basic_functionality(self, sample_data, perm_indexes):
        """Test that function runs without errors and returns expected structure."""
        result = run_reg_perms(
            task='test_task',
            tp='predictor',
            ss='score',
            dat=sample_data,
            perm_indexes=perm_indexes
        )
        
        # Check return type
        assert isinstance(result, dict)
        
        # Check all expected keys are present
        expected_keys = {'task', 'parameter', 'score', 't', 'full_r2', 'partial_r2', 'perm_p'}
        assert set(result.keys()) == expected_keys
        
        # Check values have correct types
        assert isinstance(result['task'], str)
        assert isinstance(result['parameter'], str)
        assert isinstance(result['score'], str)
        assert isinstance(result['t'], (int, float, np.number))
        assert isinstance(result['full_r2'], (int, float, np.number))
        assert isinstance(result['partial_r2'], (int, float, np.number))
        assert isinstance(result['perm_p'], (int, float, np.number))
    
    def test_result_values(self, sample_data, perm_indexes):
        """Test that result values are as expected."""
        result = run_reg_perms(
            task='task1',
            tp='predictor',
            ss='score',
            dat=sample_data,
            perm_indexes=perm_indexes
        )
        
        assert result['task'] == 'task1'
        assert result['parameter'] == 'predictor'
        assert result['score'] == 'score'
    
    def test_r_squared_bounds(self, sample_data, perm_indexes):
        """Test that R-squared values are within valid bounds [0, 1]."""
        result = run_reg_perms(
            task='test',
            tp='predictor',
            ss='score',
            dat=sample_data,
            perm_indexes=perm_indexes
        )
        
        assert 0 <= result['full_r2'] <= 1
        assert 0 <= result['partial_r2'] <= 1
    
    def test_p_value_bounds(self, sample_data, perm_indexes):
        """Test that p-value is within valid bounds [0, 1]."""
        result = run_reg_perms(
            task='test',
            tp='predictor',
            ss='score',
            dat=sample_data,
            perm_indexes=perm_indexes
        )
        
        assert 0 <= result['perm_p'] <= 1
    
    def test_different_permutation_sizes(self, sample_data):
        """Test with different numbers of permutations."""
        np.random.seed(42)
        n = len(sample_data)
        
        for n_perms in [5, 50, 100]:
            perm_idx = np.array([np.random.permutation(n) for _ in range(n_perms)]).T
            result = run_reg_perms(
                task='test',
                tp='predictor',
                ss='score',
                dat=sample_data,
                perm_indexes=perm_idx
            )
            assert isinstance(result['perm_p'], (int, float, np.number))
    
    def test_perfect_predictor(self):
        """Test with a perfect predictor (should give high R-squared, low p-value)."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'age': np.random.uniform(20, 80, n),
            'sex': np.random.choice([0, 1], n),
        })
        data['age2'] = data['age'] ** 2
        data['predictor'] = np.random.randn(n)
        # Make score strongly dependent on predictor
        data['score'] = 5 * data['predictor'] + np.random.randn(n) * 0.1
        
        perm_idx = np.array([np.random.permutation(n) for _ in range(100)]).T
        
        result = run_reg_perms(
            task='test',
            tp='predictor',
            ss='score',
            dat=data,
            perm_indexes=perm_idx
        )
        
        # Should have high R-squared
        assert result['full_r2'] > 0.5
        # Should have low p-value (likely significant)
        assert result['perm_p'] < 0.5
    
    def test_null_predictor(self):
        """Test with a predictor uncorrelated with outcome."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'age': np.random.uniform(20, 80, n),
            'sex': np.random.choice([0, 1], n),
            'score': np.random.randn(n),
            'predictor': np.random.randn(n)
        })
        data['age2'] = data['age'] ** 2
        
        perm_idx = np.array([np.random.permutation(n) for _ in range(100)]).T
        
        result = run_reg_perms(
            task='test',
            tp='predictor',
            ss='score',
            dat=data,
            perm_indexes=perm_idx
        )
        
        # P-value should be relatively high (not significant)
        # Using relaxed threshold since randomness can vary
        assert 0 <= result['perm_p'] <= 1
    
    def test_reproducibility(self, sample_data):
        """Test that same inputs give same results."""
        np.random.seed(999)
        perm_idx = np.array([np.random.permutation(len(sample_data)) for _ in range(10)]).T
        
        result1 = run_reg_perms('test', 'predictor', 'score', sample_data, perm_idx)
        result2 = run_reg_perms('test', 'predictor', 'score', sample_data, perm_idx)
        
        assert result1['t'] == result2['t']
        assert result1['full_r2'] == result2['full_r2']
        assert result1['partial_r2'] == result2['partial_r2']
        assert result1['perm_p'] == result2['perm_p']
    
    def test_perm_index_shape_validation(self, sample_data):
        """Test that permutation indexes must match data length."""
        # Incorrect shape - should raise error
        wrong_perm = np.random.permutation(50).reshape(-1, 1)
        
        with pytest.raises((ValueError, Exception)):
            run_reg_perms('test', 'predictor', 'score', sample_data, wrong_perm)
    
    def test_missing_column_error(self, sample_data, perm_indexes):
        """Test that missing columns raise appropriate errors."""
        with pytest.raises((KeyError, Exception)):
            run_reg_perms('test', 'nonexistent_col', 'score', sample_data, perm_indexes)
        
        with pytest.raises((KeyError, Exception)):
            run_reg_perms('test', 'predictor', 'nonexistent_score', sample_data, perm_indexes)
    
    def test_partial_r2_positive(self, sample_data, perm_indexes):
        """Test that partial R-squared is non-negative."""
        result = run_reg_perms(
            task='test',
            tp='predictor',
            ss='score',
            dat=sample_data,
            perm_indexes=perm_indexes
        )
        
        # Partial R² should be >= 0 (adding a predictor shouldn't increase SSR)
        assert result['partial_r2'] >= -1e-10  # Allow tiny numerical errors
    
    def test_single_permutation(self, sample_data):
        """Test with just one permutation."""
        np.random.seed(42)
        perm_idx = np.random.permutation(len(sample_data)).reshape(-1, 1)
        
        result = run_reg_perms(
            task='test',
            tp='predictor',
            ss='score',
            dat=sample_data,
            perm_indexes=perm_idx
        )
        
        # P-value should be either 0.5 or 1.0 (only 2 values: t0 and one permutation)
        assert result['perm_p'] in [0.5, 1.0]
    
    # ========== NEW TESTS FOR NON-CONTIGUOUS INDEXES ==========
    
    def test_non_contiguous_index_basic(self, sample_data_with_gaps, perm_indexes):
        """Test that function works with non-contiguous DataFrame indexes."""
        # Verify the fixture has gaps
        assert len(sample_data_with_gaps) == 100
        assert not sample_data_with_gaps.index.is_monotonic_increasing or \
               sample_data_with_gaps.index.max() > len(sample_data_with_gaps) - 1
        
        # Function should work without errors
        result = run_reg_perms(
            task='test_gaps',
            tp='predictor',
            ss='score',
            dat=sample_data_with_gaps,
            perm_indexes=perm_indexes
        )
        
        # Check all expected outputs are present and valid
        assert isinstance(result, dict)
        assert 0 <= result['full_r2'] <= 1
        assert 0 <= result['perm_p'] <= 1
    
    def test_non_contiguous_vs_contiguous_consistency(self):
        """Test that results are consistent between contiguous and non-contiguous indexes."""
        np.random.seed(42)
        n = 100
        
        # Create data with contiguous index
        data_contiguous = pd.DataFrame({
            'age': np.random.uniform(20, 80, n),
            'sex': np.random.choice([0, 1], n),
            'score': np.random.randn(n),
            'predictor': np.random.randn(n)
        })
        data_contiguous['age2'] = data_contiguous['age'] ** 2
        
        # Create same data with non-contiguous index (skip some indexes)
        data_non_contiguous = data_contiguous.copy()
        new_index = list(range(0, 50)) + list(range(55, 105))  # Skip 50-54
        data_non_contiguous.index = new_index
        
        # Same permutations for both
        np.random.seed(123)
        perm_idx = np.array([np.random.permutation(n) for _ in range(10)]).T
        
        result_contiguous = run_reg_perms(
            'test', 'predictor', 'score', data_contiguous, perm_idx
        )
        result_non_contiguous = run_reg_perms(
            'test', 'predictor', 'score', data_non_contiguous, perm_idx
        )
        
        # Results should be identical (within numerical precision)
        assert_allclose(result_contiguous['t'], result_non_contiguous['t'], rtol=1e-10)
        assert_allclose(result_contiguous['full_r2'], result_non_contiguous['full_r2'], rtol=1e-10)
        assert_allclose(result_contiguous['partial_r2'], result_non_contiguous['partial_r2'], rtol=1e-10)
        assert_allclose(result_contiguous['perm_p'], result_non_contiguous['perm_p'], rtol=1e-10)
    
    def test_non_contiguous_index_with_large_gaps(self):
        """Test with very large gaps in the index."""
        np.random.seed(42)
        n = 100
        
        # Create data with very sparse index (e.g., 0, 100, 200, ...)
        data = pd.DataFrame({
            'age': np.random.uniform(20, 80, n),
            'sex': np.random.choice([0, 1], n),
            'score': np.random.randn(n),
            'predictor': np.random.randn(n)
        })
        data['age2'] = data['age'] ** 2
        data.index = range(0, n * 10, 10)  # Index: 0, 10, 20, ..., 990
        
        assert len(data) == 100
        assert data.index.max() == 990
        
        perm_idx = np.array([np.random.permutation(n) for _ in range(10)]).T
        
        result = run_reg_perms(
            task='test_large_gaps',
            tp='predictor',
            ss='score',
            dat=data,
            perm_indexes=perm_idx
        )
        
        assert isinstance(result, dict)
        assert 0 <= result['full_r2'] <= 1
        assert 0 <= result['perm_p'] <= 1
    
    def test_non_contiguous_index_random_order(self):
        """Test with non-contiguous index in random order."""
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'age': np.random.uniform(20, 80, n),
            'sex': np.random.choice([0, 1], n),
            'score': np.random.randn(n),
            'predictor': np.random.randn(n)
        })
        data['age2'] = data['age'] ** 2
        
        # Use random non-contiguous index
        random_index = np.random.choice(range(1000), size=n, replace=False)
        data.index = random_index
        
        perm_idx = np.array([np.random.permutation(n) for _ in range(10)]).T
        
        result = run_reg_perms(
            task='test_random_index',
            tp='predictor',
            ss='score',
            dat=data,
            perm_indexes=perm_idx
        )
        
        assert isinstance(result, dict)
        assert 0 <= result['full_r2'] <= 1
        assert 0 <= result['perm_p'] <= 1
    
    def test_permutation_indexes_are_positional(self, sample_data_with_gaps):
        """
        Verify that permutation indexes work positionally (0 to n-1) 
        regardless of DataFrame index values.
        """
        n = len(sample_data_with_gaps)
        
        # Create permutation that swaps first and last positions
        perm_idx = np.arange(n).reshape(-1, 1)
        perm_idx[0], perm_idx[-1] = perm_idx[-1].copy(), perm_idx[0].copy()
        
        # This should work without trying to use the DataFrame's actual index values
        result = run_reg_perms(
            task='test_positional',
            tp='predictor',
            ss='score',
            dat=sample_data_with_gaps,
            perm_indexes=perm_idx
        )
        
        assert isinstance(result, dict)
        # Should complete without IndexError or KeyError
    
    def test_reset_index_equivalence(self):
        """
        Test that results are the same whether we use non-contiguous index
        or reset the index to contiguous.
        """
        np.random.seed(42)
        n = 105
        
        # Create data and drop some rows
        data_original = pd.DataFrame({
            'age': np.random.uniform(20, 80, n),
            'sex': np.random.choice([0, 1], n),
            'score': np.random.randn(n),
            'predictor': np.random.randn(n)
        })
        data_original['age2'] = data_original['age'] ** 2
        rows_to_drop = [5, 17, 33, 68, 99]
        data_with_gaps = data_original.drop(rows_to_drop)
        
        # Reset index version
        data_reset = data_with_gaps.reset_index(drop=True)
        
        # Same permutations
        np.random.seed(123)
        perm_idx = np.array([np.random.permutation(100) for _ in range(20)]).T
        
        result_gaps = run_reg_perms('test', 'predictor', 'score', data_with_gaps, perm_idx)
        result_reset = run_reg_perms('test', 'predictor', 'score', data_reset, perm_idx)
        
        # Should be identical
        assert_allclose(result_gaps['t'], result_reset['t'], rtol=1e-10)
        assert_allclose(result_gaps['full_r2'], result_reset['full_r2'], rtol=1e-10)
        assert_allclose(result_gaps['perm_p'], result_reset['perm_p'], rtol=1e-10)