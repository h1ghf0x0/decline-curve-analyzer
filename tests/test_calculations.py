"""
Tests for Reserves Calculations - EUR and reserves table generation.
"""
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.calculations import (
    calculate_eur,
    monte_carlo_eur_simulation,
    calculate_confidence_intervals,
    sensitivity_analysis,
    calculate_statistical_summary,
    calculate_statistical_decline_summary,
    calculate_decline_curve_summary,
    calculate_cumulative_production,
    calculate_remaining_reserves,
    generate_reserves_table,
    calculate_decline_metrics,
    calculate_recovery_factor
)
from src.models import (
    exponential_decline,
    hyperbolic_decline,
    harmonic_decline
)

class TestCalculations:
    def setup_method(self):
        """Create test parameters for calculations."""
        self.params = {
            'Qi': 1000.0,
            'Di': 0.1,
            'b': 0.5
        }
        self.q_abandon = 10.0
        self.time_unit = 'months'

    def test_calculate_eur_exponential(self):
        """Test EUR calculation for exponential decline."""
        params = {'Qi': 1000.0, 'Di': 0.1, 'b': 0.0}
        eur = calculate_eur(params['Qi'], params['Di'], params['b'],
                           self.q_abandon, self.time_unit)

        # For exponential: EUR = (Qi - q_abandon) / Di
        expected_eur = (1000.0 - 10.0) / 0.1
        assert np.isclose(eur, expected_eur)

    def test_calculate_eur_hyperbolic(self):
        """Test EUR calculation for hyperbolic decline."""
        eur = calculate_eur(self.params['Qi'], self.params['Di'], self.params['b'],
                           self.q_abandon, self.time_unit)

        # For hyperbolic: EUR = (Qi / (Di * (1-b))) * (1 - (q_abandon/Qi)^(1-b))
        expected_eur = (1000.0 / (0.1 * (1.0 - 0.5))) * (1.0 - (10.0 / 1000.0) ** (1.0 - 0.5))
        assert np.isclose(eur, expected_eur)

    def test_calculate_eur_harmonic(self):
        """Test EUR calculation for harmonic decline."""
        params = {'Qi': 1000.0, 'Di': 0.1, 'b': 1.0}
        eur = calculate_eur(params['Qi'], params['Di'], params['b'],
                           self.q_abandon, self.time_unit)

        # For harmonic: EUR = (Qi / Di) * ln(1 + Di * t_abandon)
        t_abandon = (1000.0 / 10.0 - 1.0) / 0.1
        expected_eur = (1000.0 / 0.1) * np.log(1.0 + 0.1 * t_abandon)
        assert np.isclose(eur, expected_eur)

    def test_monte_carlo_eur_simulation(self):
        """Test Monte Carlo EUR simulation."""
        params = {
            'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5},
            'perr': {'Qi': 50.0, 'Di': 0.01, 'b': 0.05}
        }

        results = monte_carlo_eur_simulation(params, self.q_abandon, self.time_unit, n_samples=100)

        assert 'mean' in results
        assert 'std' in results
        assert 'ci_95_lower' in results
        assert 'ci_95_upper' in results
        assert 'samples' in results
        assert 'n_samples' in results

        assert results['n_samples'] > 0
        assert results['mean'] > 0
        assert results['std'] > 0
        assert results['ci_95_lower'] < results['ci_95_upper']

    def test_calculate_confidence_intervals(self):
        """Test confidence interval calculation."""
        params = {
            'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5},
            'perr': {'Qi': 50.0, 'Di': 0.01, 'b': 0.05}
        }

        ci = calculate_confidence_intervals(params, self.q_abandon, self.time_unit)

        assert 'std' in ci
        assert 'ci_lower' in ci
        assert 'ci_upper' in ci
        assert 'confidence_level' in ci

        assert ci['std'] > 0
        assert ci['ci_lower'] >= 0
        assert ci['ci_upper'] > ci['ci_lower']

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        params = {
            'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}
        }

        sens = sensitivity_analysis(params, self.q_abandon, self.time_unit, n_samples=50)

        assert 'S1' in sens
        assert 'S2' in sens
        assert 'ST' in sens
        assert 'problem' in sens
        assert 'n_valid_samples' in sens

        assert len(sens['S1']) == 3
        assert len(sens['S2']) == 3
        assert len(sens['ST']) == 3
        assert sens['n_valid_samples'] > 0

        # Sensitivity indices should be between 0 and 1
        assert all(0 <= s <= 1 for s in sens['S1'])
        assert all(0 <= s <= 1 for s in sens['ST'])

    def test_calculate_statistical_summary(self):
        """Test statistical summary calculation."""
        results = {
            'exponential': {
                'success': True,
                'parameters': {'Qi': 1000.0, 'Di': 0.1},
                'perr': {'Qi': 50.0, 'Di': 0.01},
                'r_squared': 0.95,
                'rmse': 10.0
            },
            'hyperbolic': {
                'success': True,
                'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5},
                'perr': {'Qi': 50.0, 'Di': 0.01, 'b': 0.05},
                'r_squared': 0.98,
                'rmse': 8.0
            },
            'harmonic': {
                'success': True,
                'parameters': {'Qi': 1000.0, 'Di': 0.1},
                'perr': {'Qi': 50.0, 'Di': 0.01},
                'r_squared': 0.92,
                'rmse': 12.0
            }
        }

        summary = calculate_statistical_summary(results, self.q_abandon, self.time_unit)

        assert isinstance(summary, dict)
        assert 'exponential' in summary
        assert 'hyperbolic' in summary
        assert 'harmonic' in summary

        for model_name, stats in summary.items():
            assert 'eur' in stats
            assert 'ci_lower' in stats
            assert 'ci_upper' in stats
            assert 'mc_mean' in stats
            assert 'mc_std' in stats
            assert 'sensitivity_S1' in stats
            assert 'sensitivity_ST' in stats

            assert stats['eur'] > 0
            assert stats['ci_lower'] >= 0
            assert stats['mc_mean'] > 0
            assert len(stats['sensitivity_S1']) == 3
            assert len(stats['sensitivity_ST']) == 3

    def test_calculate_statistical_decline_summary(self):
        """Test statistical decline summary calculation."""
        result = {
            'success': True,
            'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5},
            'perr': {'Qi': 50.0, 'Di': 0.01, 'b': 0.05},
            'r_squared': 0.98,
            'rmse': 8.0
        }

        summary = calculate_statistical_decline_summary(result, self.q_abandon, self.time_unit)

        assert isinstance(summary, dict)
        assert 'Qi' in summary
        assert 'Di' in summary
        assert 'b' in summary
        assert 'eur' in summary
        assert 'time_to_abandonment' in summary
        assert 'effective_decline_rate' in summary
        assert 'mc_mean' in summary
        assert 'mc_std' in summary
        assert 'sensitivity_S1' in summary
        assert 'sensitivity_ST' in summary

        assert summary['eur'] > 0
        assert summary['time_to_abandonment'] > 0
        assert summary['effective_decline_rate'] > 0
        assert summary['mc_mean'] > 0
        assert len(summary['sensitivity_S1']) == 3
        assert len(summary['sensitivity_ST']) == 3

    def test_calculate_decline_curve_summary(self):
        """Test decline curve summary calculation."""
        result = {
            'success': True,
            'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5},
            'r_squared': 0.98,
            'rmse': 8.0
        }

        summary = calculate_decline_curve_summary(result, self.q_abandon, self.time_unit)

        assert isinstance(summary, dict)
        assert 'Qi' in summary
        assert 'Di' in summary
        assert 'b' in summary
        assert 'eur' in summary
        assert 'time_to_abandonment' in summary
        assert 'effective_decline_rate' in summary
        assert 'r_squared' in summary
        assert 'rmse' in summary
        assert 'model_name' in summary

        assert summary['eur'] > 0
        assert summary['time_to_abandonment'] > 0
        assert summary['effective_decline_rate'] > 0

    def test_calculate_cumulative_production_exponential(self):
        """Test cumulative production calculation for exponential decline."""
        t = np.array([0, 1, 2, 3, 4, 5])
        qi = 1000.0
        di = 0.1
        b = 0.0

        cum_prod = calculate_cumulative_production(qi, di, b, t)

        # For exponential: Np = (qi / Di) * (1 - exp(-Di * t))
        expected_cum = (qi / di) * (1.0 - np.exp(-di * t))
        assert np.allclose(cum_prod, expected_cum)

    def test_calculate_cumulative_production_hyperbolic(self):
        """Test cumulative production calculation for hyperbolic decline."""
        t = np.array([0, 1, 2, 3, 4, 5])
        qi = 1000.0
        di = 0.1
        b = 0.5

        cum_prod = calculate_cumulative_production(qi, di, b, t)

        # For hyperbolic: Np = (qi^b / (Di * (1-b))) * (qi^(1-b) - q^(1-b))
        q_t = hyperbolic_decline(t, qi, di, b)
        expected_cum = (qi ** b / (di * (1.0 - b))) * (qi ** (1.0 - b) - q_t ** (1.0 - b))
        assert np.allclose(cum_prod, expected_cum)

    def test_calculate_cumulative_production_harmonic(self):
        """Test cumulative production calculation for harmonic decline."""
        t = np.array([0, 1, 2, 3, 4, 5])
        qi = 1000.0
        di = 0.1
        b = 1.0

        cum_prod = calculate_cumulative_production(qi, di, b, t)

        # For harmonic: Np = (qi / Di) * ln(1 + Di * t)
        expected_cum = (qi / di) * np.log(1.0 + di * t)
        assert np.allclose(cum_prod, expected_cum)

    def test_calculate_remaining_reserves(self):
        """Test remaining reserves calculation."""
        t = np.array([0, 1, 2, 3, 4, 5])
        qi = 1000.0
        di = 0.1
        b = 0.5
        eur = calculate_eur(qi, di, b, self.q_abandon, self.time_unit)

        remaining = calculate_remaining_reserves(qi, di, b, self.q_abandon, t)

        # Remaining = EUR - Cumulative production
        cum_prod = calculate_cumulative_production(qi, di, b, t)
        expected_remaining = eur - cum_prod
        assert np.allclose(remaining, expected_remaining)
        assert np.all(remaining >= 0)

    def test_generate_reserves_table(self):
        """Test reserves table generation."""
        params = {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}
        time_range = (0, 60)  # 60 months

        reserves_table = generate_reserves_table(params, time_range, freq='M',
                                               q_abandon=self.q_abandon, time_unit=self.time_unit)

        assert isinstance(reserves_table, pd.DataFrame)
        assert 'time' in reserves_table.columns
        assert 'rate' in reserves_table.columns
        assert 'cumulative_production' in reserves_table.columns
        assert 'remaining_reserves' in reserves_table.columns
        assert len(reserves_table) > 0

        # Check that rates decrease over time
        assert np.all(np.diff(reserves_table['rate']) <= 0)

    def test_calculate_decline_metrics(self):
        """Test decline metrics calculation."""
        params = {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}

        metrics = calculate_decline_metrics(params, self.q_abandon, self.time_unit)

        assert isinstance(metrics, dict)
        assert 'Qi' in metrics
        assert 'Di' in metrics
        assert 'b' in metrics
        assert 'q_abandon' in metrics
        assert 'eur' in metrics
        assert 'time_to_abandonment' in metrics
        assert 'effective_decline_rate' in metrics
        assert 'initial_period_production' in metrics
        assert 'time_unit' in metrics

        assert metrics['eur'] > 0
        assert metrics['time_to_abandonment'] > 0
        assert metrics['effective_decline_rate'] > 0
        assert metrics['initial_period_production'] > 0

    def test_calculate_recovery_factor(self):
        """Test recovery factor calculation."""
        eur = 100000.0
        oip = 1000000.0

        rf = calculate_recovery_factor(eur, oip)

        assert 0 <= rf <= 1
        assert np.isclose(rf, eur / oip)

        # Test edge cases
        assert calculate_recovery_factor(0, 1000000) == 0.0
        assert calculate_recovery_factor(1000000, 0) == 0.0
        assert calculate_recovery_factor(1500000, 1000000) == 1.0

    def test_calculate_eur_with_zero_abandonment_rate(self):
        """Test EUR calculation with zero abandonment rate."""
        params = {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}
        q_abandon = 0.0

        eur = calculate_eur(params['Qi'], params['Di'], params['b'],
                           q_abandon, self.time_unit)

        # With zero abandonment rate, EUR should be very large (theoretically infinite)
        assert eur > 1e6  # Should be very large

    def test_calculate_eur_with_high_abandonment_rate(self):
        """Test EUR calculation with high abandonment rate."""
        params = {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}
        q_abandon = 900.0  # Close to initial rate

        eur = calculate_eur(params['Qi'], params['Di'], params['b'],
                           q_abandon, self.time_unit)

        # With high abandonment rate, EUR should be small
        assert eur < 100.0

    def test_calculate_eur_with_negative_parameters(self):
        """Test EUR calculation with negative parameters (should handle gracefully)."""
        params = {'Qi': -1000.0, 'Di': -0.1, 'b': -0.5}

        eur = calculate_eur(params['Qi'], params['Di'], params['b'],
                           self.q_abandon, self.time_unit)

        # Should return 0 for invalid parameters
        assert eur == 0.0

    def test_monte_carlo_with_zero_uncertainty(self):
        """Test Monte Carlo with zero parameter uncertainty."""
        params = {
            'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5},
            'perr': {'Qi': 0.0, 'Di': 0.0, 'b': 0.0}
        }

        results = monte_carlo_eur_simulation(params, self.q_abandon, self.time_unit, n_samples=10)

        # With zero uncertainty, all samples should be the same
        assert np.allclose(results['samples'], results['mean'])
        assert results['std'] == 0.0
        assert results['ci_95_lower'] == results['ci_95_upper']

    def test_sensitivity_analysis_with_constant_eur(self):
        """Test sensitivity analysis when EUR is constant across parameter range."""
        params = {
            'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}
        }

        # Create a problem where EUR doesn't change with parameters
        sens = sensitivity_analysis(params, self.q_abandon, self.time_unit, n_samples=10)

        # All sensitivity indices should be zero
        assert np.allclose(sens['S1'], 0.0)
        assert np.allclose(sens['ST'], 0.0)

    def test_generate_reserves_table_with_daily_frequency(self):
        """Test reserves table generation with daily frequency."""
        params = {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}
        time_range = (0, 30)  # 30 days

        reserves_table = generate_reserves_table(params, time_range, freq='D',
                                               q_abandon=self.q_abandon, time_unit=self.time_unit)

        assert len(reserves_table) == 30
        assert 'period' in reserves_table.columns
        assert reserves_table['period'].iloc[0] == 'Day 0'

    def test_generate_reserves_table_with_yearly_frequency(self):
        """Test reserves table generation with yearly frequency."""
        params = {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}
        time_range = (0, 5)  # 5 years

        reserves_table = generate_reserves_table(params, time_range, freq='Y',
                                               q_abandon=self.q_abandon, time_unit=self.time_unit)

        assert len(reserves_table) == 5
        assert 'period' in reserves_table.columns
        assert reserves_table['period'].iloc[0] == 'Year 0'

    def test_calculate_decline_metrics_with_zero_parameters(self):
        """Test decline metrics calculation with zero parameters."""
        params = {'Qi': 0.0, 'Di': 0.0, 'b': 0.0}

        metrics = calculate_decline_metrics(params, self.q_abandon, self.time_unit)

        assert metrics['eur'] == 0.0
        assert metrics['time_to_abandonment'] == 0.0
        assert metrics['effective_decline_rate'] == 0.0
        assert metrics['initial_period_production'] == 0.0