"""
Tests for Curve Fitting Engine - Arps DCA model fitting.
"""
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.fitting import (
    fit_arps_model,
    fit_all_models,
    get_best_model,
    calculate_aic,
    calculate_bic,
    add_information_criteria
)
from src.models import (
    exponential_decline,
    hyperbolic_decline,
    harmonic_decline
)

class TestFitting:
    def setup_method(self):
        """Create test data for fitting."""
        self.time = np.array([0, 1, 2, 3, 4, 5])
        self.rate = np.array([1000, 900, 810, 729, 656, 590])

    def test_fit_exponential_model(self):
        """Test fitting exponential decline model."""
        result = fit_arps_model('exponential', self.time, self.rate)

        assert result['success']
        assert result['model_name'] == 'exponential'
        assert 'Qi' in result['parameters']
        assert 'Di' in result['parameters']
        assert 'r_squared' in result
        assert 'rmse' in result
        assert 'residuals' in result
        assert 'fitted_rate' in result

        # Check parameter values
        assert result['parameters']['Qi'] > 0
        assert result['parameters']['Di'] > 0
        assert result['r_squared'] > 0.9

    def test_fit_hyperbolic_model(self):
        """Test fitting hyperbolic decline model."""
        result = fit_arps_model('hyperbolic', self.time, self.rate)

        assert result['success']
        assert result['model_name'] == 'hyperbolic'
        assert 'Qi' in result['parameters']
        assert 'Di' in result['parameters']
        assert 'b' in result['parameters']
        assert 'r_squared' in result
        assert 'rmse' in result
        assert 'residuals' in result
        assert 'fitted_rate' in result

        # Check parameter values
        assert result['parameters']['Qi'] > 0
        assert result['parameters']['Di'] > 0
        assert 0 < result['parameters']['b'] < 1
        assert result['r_squared'] > 0.9

    def test_fit_harmonic_model(self):
        """Test fitting harmonic decline model."""
        result = fit_arps_model('harmonic', self.time, self.rate)

        assert result['success']
        assert result['model_name'] == 'harmonic'
        assert 'Qi' in result['parameters']
        assert 'Di' in result['parameters']
        assert 'r_squared' in result
        assert 'rmse' in result
        assert 'residuals' in result
        assert 'fitted_rate' in result

        # Check parameter values
        assert result['parameters']['Qi'] > 0
        assert result['parameters']['Di'] > 0
        assert result['r_squared'] > 0.9

    def test_fit_all_models(self):
        """Test fitting all models."""
        results = fit_all_models(self.time, self.rate)

        assert isinstance(results, dict)
        assert len(results) == 3
        assert 'exponential' in results
        assert 'hyperbolic' in results
        assert 'harmonic' in results

        for model_name, result in results.items():
            assert 'success' in result
            assert 'model_name' in result
            assert 'parameters' in result
            assert 'r_squared' in result
            assert 'rmse' in result

    def test_get_best_model(self):
        """Test getting the best fitting model."""
        results = {
            'exponential': {'r_squared': 0.95, 'success': True},
            'hyperbolic': {'r_squared': 0.98, 'success': True},
            'harmonic': {'r_squared': 0.92, 'success': True}
        }

        best_model, best_result = get_best_model(results)

        assert best_model == 'hyperbolic'
        assert best_result['r_squared'] == 0.98

    def test_get_best_model_with_failed_fits(self):
        """Test getting best model when some fits fail."""
        results = {
            'exponential': {'r_squared': 0.95, 'success': True},
            'hyperbolic': {'r_squared': 0.0, 'success': False},
            'harmonic': {'r_squared': 0.92, 'success': True}
        }

        best_model, best_result = get_best_model(results)

        assert best_model == 'exponential'
        assert best_result['r_squared'] == 0.95

    def test_calculate_aic(self):
        """Test AIC calculation."""
        n = 100  # Number of data points
        k = 2    # Number of parameters
        rmse = 10.0

        aic = calculate_aic(n, k, rmse)

        assert isinstance(aic, float)
        assert aic < 0  # Should be negative for good fit

    def test_calculate_bic(self):
        """Test BIC calculation."""
        n = 100
        k = 2
        rmse = 10.0

        bic = calculate_bic(n, k, rmse)

        assert isinstance(bic, float)
        assert bic < 0

    def test_add_information_criteria(self):
        """Test adding AIC and BIC to results."""
        results = {
            'exponential': {'rmse': 10.0, 'success': True},
            'hyperbolic': {'rmse': 8.0, 'success': True},
            'harmonic': {'rmse': 12.0, 'success': True}
        }
        n = 100

        updated_results = add_information_criteria(results, n)

        for model_name, result in updated_results.items():
            assert 'aic' in result
            assert 'bic' in result
            assert isinstance(result['aic'], float)
            assert isinstance(result['bic'], float)

    def test_fit_with_initial_guess(self):
        """Test fitting with initial parameter guesses."""
        initial_guess = {'Qi': 1000, 'Di': 0.1, 'b': 0.5}
        result = fit_arps_model('hyperbolic', self.time, self.rate, initial_guess)

        assert result['success']
        assert np.isclose(result['parameters']['Qi'], initial_guess['Qi'], rtol=0.1)
        assert np.isclose(result['parameters']['Di'], initial_guess['Di'], rtol=0.1)
        assert np.isclose(result['parameters']['b'], initial_guess['b'], rtol=0.1)

    def test_fit_with_insufficient_data(self):
        """Test fitting with insufficient data points."""
        small_time = np.array([0, 1])
        small_rate = np.array([1000, 900])

        result = fit_arps_model('exponential', small_time, small_rate)

        assert not result['success']
        assert 'error' in result['message']

    def test_fit_with_zero_rates(self):
        """Test fitting with zero rates."""
        zero_rate = np.array([1000, 900, 0, 0, 0, 0])

        result = fit_arps_model('exponential', self.time, zero_rate)

        assert not result['success']
        assert 'error' in result['message']

    def test_fit_with_negative_rates(self):
        """Test fitting with negative rates."""
        negative_rate = np.array([1000, 900, -100, -200, -300, -400])

        result = fit_arps_model('exponential', self.time, negative_rate)

        assert not result['success']
        assert 'error' in result['message']

    def test_fit_with_constant_rate(self):
        """Test fitting with constant rate (no decline)."""
        constant_rate = np.array([1000, 1000, 1000, 1000, 1000, 1000])

        result = fit_arps_model('exponential', self.time, constant_rate)

        assert result['success']
        assert result['parameters']['Di'] == 0.0
        assert result['r_squared'] == 1.0

    def test_fit_with_increasing_rate(self):
        """Test fitting with increasing rate (unphysical)."""
        increasing_rate = np.array([500, 600, 700, 800, 900, 1000])

        result = fit_arps_model('exponential', self.time, increasing_rate)

        assert not result['success']
        assert 'error' in result['message']

    def test_fit_with_noise(self):
        """Test fitting with noisy data."""
        np.random.seed(42)
        noisy_rate = self.rate * (1 + np.random.normal(0, 0.1, size=self.rate.shape))

        result = fit_arps_model('exponential', self.time, noisy_rate)

        assert result['success']
        assert result['r_squared'] > 0.8  # Should still fit reasonably well

    def test_fit_with_different_time_units(self):
        """Test fitting with different time units."""
        # Time in years instead of months
        time_years = self.time / 12.0
        rate = np.array([1000, 950, 903, 858, 816, 776])

        result = fit_arps_model('exponential', time_years, rate)

        assert result['success']
        # Di should be adjusted for time units
        assert result['parameters']['Di'] > 0