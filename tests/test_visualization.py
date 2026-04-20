"""
Tests for Visualization Module - Plotly chart generation.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import patch
from src.visualization import (
    create_rate_time_chart,
    create_log_chart,
    create_residuals_chart,
    create_model_comparison_chart,
    create_cumulative_production_chart,
    create_decline_rate_chart
)

class TestVisualization:
    def setup_method(self):
        """Create test data for visualization."""
        self.time = np.array([0, 1, 2, 3, 4, 5])
        self.rate = np.array([1000, 900, 810, 729, 656, 590])
        self.actual_data = pd.DataFrame({
            'time': self.time,
            'oil_rate': self.rate
        })

        self.fitted_curves = {
            'exponential': {
                'success': True,
                'fitted_rate': np.array([1000, 890, 792, 705, 628, 559]),
                'residuals': self.rate - np.array([1000, 890, 792, 705, 628, 559])
            },
            'hyperbolic': {
                'success': True,
                'fitted_rate': np.array([1000, 895, 801, 718, 644, 578]),
                'residuals': self.rate - np.array([1000, 895, 801, 718, 644, 578])
            },
            'harmonic': {
                'success': True,
                'fitted_rate': np.array([1000, 888, 788, 700, 623, 555]),
                'residuals': self.rate - np.array([1000, 888, 788, 700, 623, 555])
            }
        }

    def test_create_rate_time_chart(self):
        """Test rate vs time chart creation."""
        fig = create_rate_time_chart(
            self.actual_data,
            self.fitted_curves,
            selected_model='hyperbolic',
            rate_column='oil_rate',
            unit='STB/D'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # 1 actual + 3 fitted models

        # Check actual data trace
        actual_trace = fig.data[0]
        assert actual_trace.mode == 'markers'
        assert actual_trace.name == 'Actual Data'
        assert np.array_equal(actual_trace.x, self.time)
        assert np.array_equal(actual_trace.y, self.rate)

        # Check fitted traces
        for i, model_name in enumerate(['exponential', 'hyperbolic', 'harmonic']):
            trace = fig.data[i + 1]
            assert trace.mode == 'lines'
            assert trace.name == model_name.title()
            assert np.array_equal(trace.x, self.time)
            assert np.array_equal(trace.y, self.fitted_curves[model_name]['fitted_rate'])

    def test_create_log_chart(self):
        """Test log scale chart creation."""
        fig = create_log_chart(
            self.actual_data,
            self.fitted_curves,
            selected_model='hyperbolic',
            rate_column='oil_rate',
            unit='STB/D'
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.yaxis_type == 'log'

        # Check actual data (should filter out zeros for log scale)
        actual_trace = fig.data[0]
        assert actual_trace.mode == 'markers'
        assert actual_trace.name == 'Actual Data'
        assert np.array_equal(actual_trace.x, self.time)
        assert np.array_equal(actual_trace.y, self.rate)

        # Check fitted traces
        for i, model_name in enumerate(['exponential', 'hyperbolic', 'harmonic']):
            trace = fig.data[i + 1]
            assert trace.mode == 'lines'
            assert trace.name == model_name.title()
            assert np.array_equal(trace.x, self.time)
            assert np.array_equal(trace.y, self.fitted_curves[model_name]['fitted_rate'])

    def test_create_residuals_chart(self):
        """Test residuals chart creation."""
        fig = create_residuals_chart(self.fitted_curves)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 6  # 3 residuals traces + 3 histogram traces

        # Check residuals vs time traces
        for i, model_name in enumerate(['exponential', 'hyperbolic', 'harmonic']):
            trace = fig.data[i]
            assert trace.mode == 'markers'
            assert trace.name == model_name.title()
            assert np.array_equal(trace.x, np.arange(len(self.rate)))
            assert np.array_equal(trace.y, self.fitted_curves[model_name]['residuals'])

        # Check residuals distribution traces
        for i, model_name in enumerate(['exponential', 'hyperbolic', 'harmonic']):
            hist_trace = fig.data[i + 3]
            assert hist_trace.type == 'histogram'
            assert hist_trace.name == f'{model_name.title()} (dist)'

        # Check zero line
        assert len(fig.data) == 6
        assert fig.data[3].y == 0  # Zero line should be at y=0

    def test_create_model_comparison_chart(self):
        """Test model comparison chart creation."""
        fig = create_model_comparison_chart(self.fitted_curves)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # 2 bar charts with 3 bars each

        # Check R-squared bar chart
        r2_bars = fig.data[0:3]
        for i, model_name in enumerate(['exponential', 'hyperbolic', 'harmonic']):
            bar = r2_bars[i]
            assert bar.type == 'bar'
            assert bar.name == 'R-squared'
            assert bar.x[0] == model_name.title()
            assert bar.y[0] == self.fitted_curves[model_name]['r_squared']

        # Check RMSE bar chart
        rmse_bars = fig.data[3:6]
        for i, model_name in enumerate(['exponential', 'hyperbolic', 'harmonic']):
            bar = rmse_bars[i]
            assert bar.type == 'bar'
            assert bar.name == 'RMSE'
            assert bar.x[0] == model_name.title()
            assert bar.y[0] == self.fitted_curves[model_name]['rmse']

    def test_create_cumulative_production_chart(self):
        """Test cumulative production chart creation."""
        fitted_result = self.fitted_curves['hyperbolic']

        fig = create_cumulative_production_chart(
            self.actual_data,
            fitted_result,
            rate_column='oil_rate',
            unit='STB'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Actual + Fitted cumulative

        # Check actual cumulative trace
        actual_trace = fig.data[0]
        assert actual_trace.mode == 'lines'
        assert actual_trace.name == 'Actual Cumulative'
        assert np.array_equal(actual_trace.x, self.time)

        # Calculate actual cumulative for comparison
        actual_cum = np.cumsum(self.rate)
        assert np.allclose(actual_trace.y, actual_cum)

        # Check fitted cumulative trace
        fitted_trace = fig.data[1]
        assert fitted_trace.mode == 'lines'
        assert fitted_trace.name == 'Fitted Cumulative'
        assert fitted_trace.line.dash == 'dash'
        assert np.array_equal(fitted_trace.x, self.time)

        # Calculate fitted cumulative for comparison
        fitted_cum = np.cumsum(fitted_result['fitted_rate'])
        assert np.allclose(fitted_trace.y, fitted_cum)

    def test_create_decline_rate_chart(self):
        """Test decline rate chart creation."""
        fitted_result = self.fitted_curves['hyperbolic']
        fitted_result['parameters'] = {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}

        fig = create_decline_rate_chart(
            fitted_result,
            time_range=(0, 5),
            title='Decline Rate vs Time'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

        # Check decline rate trace
        trace = fig.data[0]
        assert trace.mode == 'lines'
        assert trace.name == 'Decline Rate'
        assert trace.line.color == 'blue'
        assert trace.line.width == 2

        # Check decline rate values
        time = np.linspace(0, 5, 100)
        di = fitted_result['parameters']['Di']
        b = fitted_result['parameters']['b']
        expected_decline_rate = di / (1.0 + b * di * time)
        assert np.allclose(trace.y, expected_decline_rate)

    def test_create_rate_time_chart_with_empty_data(self):
        """Test rate vs time chart with empty data."""
        empty_data = pd.DataFrame({'time': [], 'oil_rate': []})
        empty_fitted_curves = {}

        fig = create_rate_time_chart(
            empty_data,
            empty_fitted_curves,
            rate_column='oil_rate',
            unit='STB/D'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # Should have no traces

    def test_create_log_chart_with_zero_rates(self):
        """Test log chart with zero rates."""
        log_data = pd.DataFrame({
            'time': self.time,
            'oil_rate': [0, 0, 0, 0, 0, 0]
        })

        fig = create_log_chart(
            log_data,
            self.fitted_curves,
            rate_column='oil_rate',
            unit='STB/D'
        )

        assert isinstance(fig, go.Figure)
        # Should handle zero rates gracefully (no actual data trace)

    def test_create_residuals_chart_with_failed_fits(self):
        """Test residuals chart with failed fits."""
        failed_fitted_curves = {
            'exponential': {'success': False},
            'hyperbolic': {'success': False},
            'harmonic': {'success': False}
        }

        fig = create_residuals_chart(failed_fitted_curves)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # No traces for failed fits

    def test_create_model_comparison_chart_with_single_model(self):
        """Test model comparison chart with single model."""
        single_model_curves = {
            'exponential': {'success': True, 'r_squared': 0.95, 'rmse': 10.0}
        }

        fig = create_model_comparison_chart(single_model_curves)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # 1 R-squared bar + 1 RMSE bar

    def test_create_cumulative_production_chart_with_no_fitted_data(self):
        """Test cumulative production chart with no fitted data."""
        no_fitted_result = {'success': False}

        fig = create_cumulative_production_chart(
            self.actual_data,
            no_fitted_result,
            rate_column='oil_rate',
            unit='STB'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Only actual cumulative

    def test_create_decline_rate_chart_with_exponential_model(self):
        """Test decline rate chart with exponential model."""
        fitted_result = {
            'success': True,
            'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.0}
        }

        fig = create_decline_rate_chart(
            fitted_result,
            time_range=(0, 5),
            title='Decline Rate vs Time'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

        # For exponential, decline rate should be constant
        trace = fig.data[0]
        assert np.allclose(trace.y, fitted_result['parameters']['Di'])

    def test_create_decline_rate_chart_with_harmonic_model(self):
        """Test decline rate chart with harmonic model."""
        fitted_result = {
            'success': True,
            'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 1.0}
        }

        fig = create_decline_rate_chart(
            fitted_result,
            time_range=(0, 5),
            title='Decline Rate vs Time'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

        # For harmonic, decline rate should decrease over time
        trace = fig.data[0]
        time = np.linspace(0, 5, 100)
        di = fitted_result['parameters']['Di']
        expected_decline_rate = di / (1.0 + di * time)
        assert np.allclose(trace.y, expected_decline_rate)
        assert trace.y[0] > trace.y[-1]  # Should decrease

    def test_chart_creation_with_invalid_parameters(self):
        """Test chart creation with invalid parameters."""
        # Test with None values
        fig = create_rate_time_chart(
            None,
            None,
            rate_column='oil_rate',
            unit='STB/D'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        fig = create_rate_time_chart(
            empty_df,
            self.fitted_curves,
            rate_column='oil_rate',
            unit='STB/D'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

        # Test with missing rate column
        missing_col_df = pd.DataFrame({'time': self.time})
        fig = create_rate_time_chart(
            missing_col_df,
            self.fitted_curves,
            rate_column='oil_rate',
            unit='STB/D'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0