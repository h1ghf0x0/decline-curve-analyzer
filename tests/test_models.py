"""
Tests for Arps DCA Model Definitions - Exponential, Hyperbolic, and Harmonic decline models.
"""
import numpy as np
from src.models import (
    exponential_decline,
    hyperbolic_decline,
    harmonic_decline,
    calculate_decline_rate,
    calculate_effective_decline_rate,
    calculate_time_to_abandonment
)

class TestModels:
    def setup_method(self):
        """Create test data for model tests."""
        self.time = np.array([0, 1, 2, 3, 4, 5])
        self.Qi = 1000.0
        self.Di = 0.1
        self.b = 0.5

    def test_exponential_decline_basic_functionality(self):
        """Test basic exponential decline functionality."""
        result = exponential_decline(self.time, self.Qi, self.Di)

        assert len(result) == len(self.time)
        assert result[0] == self.Qi  # At t=0, rate should equal Qi
        assert result[1] < result[0]  # Rate should decrease over time
        assert result[2] < result[1]
        assert result[3] < result[2]

        # Check that rates don't go negative
        assert np.all(result >= 0)

    def test_exponential_decline_zero_time(self):
        """Test that rate equals Qi at t=0."""
        t = np.array([0])
        Qi = 500.0
        Di = 0.05

        result = exponential_decline(t, Qi, Di)
        assert result[0] == Qi

    def test_exponential_decline_large_time(self):
        """Test exponential decline with large time values."""
        t = np.array([0, 100, 200])  # Very large time
        Qi = 100.0
        Di = 1.0

        result = exponential_decline(t, Qi, Di)

        # Should approach zero but not go negative
        assert np.all(result >= 0)
        assert result[-1] < 1.0  # Should be very small

    def test_hyperbolic_decline_basic_functionality(self):
        """Test basic hyperbolic decline functionality."""
        result = hyperbolic_decline(self.time, self.Qi, self.Di, self.b)

        assert len(result) == len(self.time)
        assert result[0] == self.Qi  # At t=0, rate should equal Qi
        assert result[1] < result[0]  # Rate should decrease over time
        assert result[2] < result[1]
        assert result[3] < result[2]

        # Check that rates don't go negative
        assert np.all(result >= 0)

    def test_hyperbolic_decline_b_parameter_effects(self):
        """Test that different b values produce different curves."""
        t = np.array([0, 1, 2, 3])
        Qi = 1000.0
        Di = 0.1

        # Different b values
        result_b0 = hyperbolic_decline(t, Qi, Di, 0.0)  # Should be exponential
        result_b05 = hyperbolic_decline(t, Qi, Di, 0.5)
        result_b09 = hyperbolic_decline(t, Qi, Di, 0.9)

        # At t=0, all should equal Qi
        assert result_b0[0] == Qi
        assert result_b05[0] == Qi
        assert result_b09[0] == Qi

        # Different b values should give different results at t>0
        assert result_b0[1] != result_b05[1]
        assert result_b05[1] != result_b09[1]

        # Check that b=0 gives exponential decline
        expected_exp = exponential_decline(t, Qi, Di)
        assert np.allclose(result_b0, expected_exp, rtol=1e-10)

    def test_hyperbolic_decline_boundary_conditions(self):
        """Test boundary conditions for b parameter."""
        t = np.array([1, 2, 3])
        Qi = 1000.0
        Di = 0.1

        # b approaching 0 should approach exponential
        result_b_small = hyperbolic_decline(t, Qi, Di, 0.001)
        result_exp = exponential_decline(t, Qi, Di)

        # Should be very close
        assert np.allclose(result_b_small, result_exp, rtol=1e-3)

        # b=1 should give harmonic decline
        result_b1 = hyperbolic_decline(t, Qi, Di, 1.0)
        result_harm = harmonic_decline(t, Qi, Di)
        assert np.allclose(result_b1, result_harm)

    def test_hyperbolic_decline_special_cases(self):
        """Test special cases for hyperbolic decline."""
        t = np.array([0, 1, 2, 3])
        Qi = 1000.0
        Di = 0.1

        # Test b=0 (should be exponential)
        result_b0 = hyperbolic_decline(t, Qi, Di, 0.0)
        expected_exp = exponential_decline(t, Qi, Di)
        assert np.allclose(result_b0, expected_exp)

        # Test b=1 (should be harmonic)
        result_b1 = hyperbolic_decline(t, Qi, Di, 1.0)
        expected_harm = harmonic_decline(t, Qi, Di)
        assert np.allclose(result_b1, expected_harm)

    def test_harmonic_decline_basic_functionality(self):
        """Test basic harmonic decline functionality."""
        result = harmonic_decline(self.time, self.Qi, self.Di)

        assert len(result) == len(self.time)
        assert result[0] == self.Qi  # At t=0, rate should equal Qi
        assert result[1] < result[0]  # Rate should decrease over time
        assert result[2] < result[1]
        assert result[3] < result[2]

        # Check that rates don't go negative
        assert np.all(result >= 0)

    def test_harmonic_decline_special_case(self):
        """Test that harmonic is special case of hyperbolic with b=1."""
        t = np.array([1, 2, 3])
        Qi = 1000.0
        Di = 0.1

        result_harmonic = harmonic_decline(t, Qi, Di)
        result_hyperbolic = hyperbolic_decline(t, Qi, Di, 1.0)

        assert np.allclose(result_harmonic, result_hyperbolic)

    def test_calculate_decline_rate(self):
        """Test decline rate calculation."""
        t = np.array([0, 1, 2, 3])
        Qi = 1000.0
        Di = 0.1
        b = 0.5

        decline_rates = calculate_decline_rate(Qi, Di, b, t)

        # Should be positive and decreasing
        assert np.all(decline_rates > 0)
        assert decline_rates[0] > decline_rates[-1]

        # Check special case b=0 (exponential)
        decline_rates_exp = calculate_decline_rate(Qi, Di, 0.0, t)
        assert np.allclose(decline_rates_exp, Di)

        # Check special case b=1 (harmonic)
        decline_rates_harm = calculate_decline_rate(Qi, Di, 1.0, t)
        expected_harm = Di / (1.0 + Di * t)
        assert np.allclose(decline_rates_harm, expected_harm)

    def test_calculate_effective_decline_rate(self):
        """Test effective decline rate calculation."""
        Di = 0.1
        b = 0.5
        period = 1.0  # 1 year

        effective_rate = calculate_effective_decline_rate(Di, b, period)

        # Should be between 0 and 1
        assert 0 < effective_rate < 1

        # Should be different for different b values
        effective_rate_exp = calculate_effective_decline_rate(Di, 0.0, period)
        effective_rate_harm = calculate_effective_decline_rate(Di, 1.0, period)

        assert effective_rate_exp != effective_rate_harm
        assert effective_rate_exp < effective_rate < effective_rate_harm

        # Check edge cases
        assert calculate_effective_decline_rate(0.0, b, period) == 0.0
        assert calculate_effective_decline_rate(Di, 0.0, period) == (1.0 - np.exp(-Di * period))

    def test_calculate_time_to_abandonment(self):
        """Test time to abandonment calculation."""
        Qi = 1000.0
        Di = 0.1
        b = 0.5
        q_abandon = 10.0

        time_to_abandon = calculate_time_to_abandonment(Qi, Di, b, q_abandon)

        # Should be positive
        assert time_to_abandon > 0

        # Should be finite
        assert np.isfinite(time_to_abandon)

        # Test edge cases
        time_zero = calculate_time_to_abandonment(Qi, Di, b, Qi)  # q_abandon = Qi
        assert time_zero == 0.0

        time_infinite = calculate_time_to_abandonment(Qi, Di, b, 0.0)  # q_abandon = 0
        assert np.isinf(time_infinite) or time_infinite > 1e6

    def test_exponential_decline_with_zero_initial_rate(self):
        """Test exponential decline with zero initial rate."""
        t = np.array([0, 1, 2, 3])
        Qi = 0.0
        Di = 0.1

        result = exponential_decline(t, Qi, Di)

        # Should return all zeros
        assert np.all(result == 0.0)

    def test_exponential_decline_with_zero_decline_rate(self):
        """Test exponential decline with zero decline rate."""
        t = np.array([0, 1, 2, 3])
        Qi = 1000.0
        Di = 0.0

        result = exponential_decline(t, Qi, Di)

        # Should return constant rate equal to Qi
        assert np.allclose(result, Qi)

    def test_hyperbolic_decline_with_zero_initial_rate(self):
        """Test hyperbolic decline with zero initial rate."""
        t = np.array([0, 1, 2, 3])
        Qi = 0.0
        Di = 0.1
        b = 0.5

        result = hyperbolic_decline(t, Qi, Di, b)

        # Should return all zeros
        assert np.all(result == 0.0)

    def test_hyperbolic_decline_with_zero_decline_rate(self):
        """Test hyperbolic decline with zero decline rate."""
        t = np.array([0, 1, 2, 3])
        Qi = 1000.0
        Di = 0.0
        b = 0.5

        result = hyperbolic_decline(t, Qi, Di, b)

        # Should return constant rate equal to Qi
        assert np.allclose(result, Qi)

    def test_harmonic_decline_with_zero_initial_rate(self):
        """Test harmonic decline with zero initial rate."""
        t = np.array([0, 1, 2, 3])
        Qi = 0.0
        Di = 0.1

        result = harmonic_decline(t, Qi, Di)

        # Should return all zeros
        assert np.all(result == 0.0)

    def test_harmonic_decline_with_zero_decline_rate(self):
        """Test harmonic decline with zero decline rate."""
        t = np.array([0, 1, 2, 3])
        Qi = 1000.0
        Di = 0.0

        result = harmonic_decline(t, Qi, Di)

        # Should return constant rate equal to Qi
        assert np.allclose(result, Qi)

    def test_calculate_decline_rate_with_zero_initial_rate(self):
        """Test decline rate calculation with zero initial rate."""
        Qi = 0.0
        Di = 0.1
        b = 0.5
        t = np.array([0, 1, 2, 3])

        decline_rates = calculate_decline_rate(Qi, Di, b, t)

        # Should return zeros
        assert np.all(decline_rates == 0.0)

    def test_calculate_decline_rate_with_zero_decline_rate(self):
        """Test decline rate calculation with zero decline rate."""
        Qi = 1000.0
        Di = 0.0
        b = 0.5
        t = np.array([0, 1, 2, 3])

        decline_rates = calculate_decline_rate(Qi, Di, b, t)

        # Should return zeros
        assert np.all(decline_rates == 0.0)

    def test_calculate_effective_decline_rate_with_zero_decline_rate(self):
        """Test effective decline rate calculation with zero decline rate."""
        Di = 0.0
        b = 0.5
        period = 1.0

        effective_rate = calculate_effective_decline_rate(Di, b, period)

        # Should return 0
        assert effective_rate == 0.0

    def test_calculate_time_to_abandonment_with_zero_initial_rate(self):
        """Test time to abandonment with zero initial rate."""
        Qi = 0.0
        Di = 0.1
        b = 0.5
        q_abandon = 10.0

        time_to_abandon = calculate_time_to_abandonment(Qi, Di, b, q_abandon)

        # Should return 0 (already at or below abandonment rate)
        assert time_to_abandon == 0.0

    def test_calculate_time_to_abandonment_with_zero_abandonment_rate(self):
        """Test time to abandonment with zero abandonment rate."""
        Qi = 1000.0
        Di = 0.1
        b = 0.5
        q_abandon = 0.0

        time_to_abandon = calculate_time_to_abandonment(Qi, Di, b, q_abandon)

        # Should return infinity (never reaches zero rate)
        assert np.isinf(time_to_abandon) or time_to_abandon > 1e6

    def test_calculate_time_to_abandonment_with_high_abandonment_rate(self):
        """Test time to abandonment with high abandonment rate."""
        Qi = 1000.0
        Di = 0.1
        b = 0.5
        q_abandon = 900.0  # Close to initial rate

        time_to_abandon = calculate_time_to_abandonment(Qi, Di, b, q_abandon)

        # Should be small positive value
        assert 0 < time_to_abandon < 1.0

    def test_calculate_time_to_abandonment_with_negative_parameters(self):
        """Test time to abandonment with negative parameters."""
        Qi = -1000.0
        Di = -0.1
        b = -0.5
        q_abandon = 10.0

        time_to_abandon = calculate_time_to_abandonment(Qi, Di, b, q_abandon)

        # Should handle gracefully (return 0 or positive value)
        assert time_to_abandon >= 0

    def test_model_consistency(self):
        """Test consistency between different models."""
        t = np.array([0, 1, 2, 3, 4, 5])
        Qi = 1000.0
        Di = 0.1

        # Exponential should be special case of hyperbolic with b=0
        exp_result = exponential_decline(t, Qi, Di)
        hyp_result_b0 = hyperbolic_decline(t, Qi, Di, 0.0)
        assert np.allclose(exp_result, hyp_result_b0)

        # Harmonic should be special case of hyperbolic with b=1
        harm_result = harmonic_decline(t, Qi, Di)
        hyp_result_b1 = hyperbolic_decline(t, Qi, Di, 1.0)
        assert np.allclose(harm_result, hyp_result_b1)

        # All models should return Qi at t=0
        assert exponential_decline(np.array([0]), Qi, Di)[0] == Qi
        assert hyperbolic_decline(np.array([0]), Qi, Di, 0.5)[0] == Qi
        assert harmonic_decline(np.array([0]), Qi, Di)[0] == Qi

    def test_model_behavior_at_boundaries(self):
        """Test model behavior at boundary conditions."""
        t = np.array([0, 1, 2, 3, 4, 5])
        Qi = 1000.0
        Di = 0.1

        # Test very small b values
        result_b_small = hyperbolic_decline(t, Qi, Di, 1e-10)
        expected_exp = exponential_decline(t, Qi, Di)
        assert np.allclose(result_b_small, expected_exp, rtol=1e-5)

        # Test very large b values (should approach harmonic)
        result_b_large = hyperbolic_decline(t, Qi, Di, 0.99)
        expected_harm = harmonic_decline(t, Qi, Di)
        # Should be close to harmonic but not exactly
        assert np.mean(np.abs(result_b_large - expected_harm)) < 100

        # Test zero parameters
        assert np.all(exponential_decline(t, 0.0, Di) == 0.0)
        assert np.all(hyperbolic_decline(t, 0.0, Di, b) == 0.0)
        assert np.all(harmonic_decline(t, 0.0, Di) == 0.0)

    def test_model_performance_with_large_datasets(self):
        """Test model performance with large time arrays."""
        t = np.linspace(0, 1000, 10000)  # 10,000 time points
        Qi = 1000.0
        Di = 0.1
        b = 0.5

        # Test all models with large dataset
        exp_result = exponential_decline(t, Qi, Di)
        hyp_result = hyperbolic_decline(t, Qi, Di, b)
        harm_result = harmonic_decline(t, Qi, Di)

        # Should all complete without errors
        assert len(exp_result) == len(t)
        assert len(hyp_result) == len(t)
        assert len(harm_result) == len(t)

        # Should all return finite values
        assert np.all(np.isfinite(exp_result))
        assert np.all(np.isfinite(hyp_result))
        assert np.all(np.isfinite(harm_result))

    def test_model_compatibility_with_fitting(self):
        """Test that model functions are compatible with scipy curve_fit."""
        from scipy.optimize import curve_fit

        # Test that models can be used with curve_fit
        try:
            # Test exponential
            popt, pcov = curve_fit(exponential_decline, self.time, self.rate,
                                  p0=[self.Qi, self.Di])
            assert len(popt) == 2

            # Test hyperbolic
            popt, pcov = curve_fit(hyperbolic_decline, self.time, self.rate,
                                  p0=[self.Qi, self.Di, self.b])
            assert len(popt) == 3

            # Test harmonic
            popt, pcov = curve_fit(harmonic_decline, self.time, self.rate,
                                  p0=[self.Qi, self.Di])
            assert len(popt) == 2

            # All should complete without errors
            curve_fit_success = True
        except:
            curve_fit_success = False

        assert curve_fit_success