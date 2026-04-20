"""
Tests for Export Module - CSV export functionality.
"""
import pandas as pd
import numpy as np
from io import BytesIO
from unittest.mock import patch
from src.exports import (
    export_fitting_results_to_csv,
    export_reserves_table_to_csv,
    export_complete_analysis_to_csv,
    generate_summary_report,
    export_summary_report_to_text
)

class TestExports:
    def setup_method(self):
        """Create test data for export tests."""
        self.results = {
            'exponential': {
                'success': True,
                'parameters': {'Qi': 1000.0, 'Di': 0.1},
                'perr': {'Qi': 50.0, 'Di': 0.01},
                'r_squared': 0.95,
                'rmse': 10.0,
                'aic': 100.0,
                'bic': 105.0,
                'residuals': np.array([10, -5, 3, -2, 8]),
                'fitted_rate': np.array([990, 905, 807, 731, 648])
            },
            'hyperbolic': {
                'success': True,
                'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5},
                'perr': {'Qi': 50.0, 'Di': 0.01, 'b': 0.05},
                'r_squared': 0.98,
                'rmse': 8.0,
                'aic': 90.0,
                'bic': 95.0,
                'residuals': np.array([8, -3, 2, -1, 6]),
                'fitted_rate': np.array([992, 903, 808, 730, 650])
            },
            'harmonic': {
                'success': True,
                'parameters': {'Qi': 1000.0, 'Di': 0.1},
                'perr': {'Qi': 50.0, 'Di': 0.01},
                'r_squared': 0.92,
                'rmse': 12.0,
                'aic': 110.0,
                'bic': 115.0,
                'residuals': np.array([12, -7, 4, -3, 10]),
                'fitted_rate': np.array([988, 907, 806, 732, 646])
            }
        }

        self.metrics = {
            'Qi': 1000.0,
            'Di': 0.1,
            'b': 0.5,
            'q_abandon': 10.0,
            'eur': 95000.0,
            'time_to_abandonment': 45.0,
            'effective_decline_rate': 0.25,
            'initial_period_production': 1000.0,
            'time_unit': 'months',
            'r_squared': 0.98,
            'rmse': 8.0
        }

        self.reserves_table = pd.DataFrame({
            'time': [0, 12, 24, 36, 48, 60],
            'rate': [1000, 850, 723, 614, 522, 444],
            'cumulative_production': [0, 10000, 21500, 32750, 43500, 54000],
            'remaining_reserves': [95000, 85000, 73500, 62250, 51500, 41000]
        })

    def test_export_fitting_results_to_csv(self):
        """Test exporting fitting results to CSV."""
        csv_data = export_fitting_results_to_csv(self.results, 'hyperbolic')

        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Read the CSV data
        buffer = BytesIO(csv_data)
        with pd.ExcelFile(buffer) as xls:
            # Check sheets
            assert 'Summary' in xls.sheet_names
            assert 'Model Comparison' in xls.sheet_names
            assert 'Fitted Data' in xls.sheet_names

            # Check summary sheet
            summary_df = pd.read_excel(xls, 'Summary')
            assert len(summary_df) > 0
            assert 'Parameter' in summary_df.columns
            assert 'Value' in summary_df.columns
            assert 'Error' in summary_df.columns

            # Check model comparison sheet
            comparison_df = pd.read_excel(xls, 'Model Comparison')
            assert len(comparison_df) == 4  # Header + 3 models
            assert 'Model' in comparison_df.columns
            assert 'R-squared' in comparison_df.columns
            assert 'RMSE' in comparison_df.columns
            assert 'AIC' in comparison_df.columns
            assert 'BIC' in comparison_df.columns
            assert 'Success' in comparison_df.columns

            # Check fitted data sheet
            fitted_df = pd.read_excel(xls, 'Fitted Data')
            assert len(fitted_df) > 0
            assert 'Time' in fitted_df.columns
            assert 'Actual Rate' in fitted_df.columns
            assert 'Fitted Rate' in fitted_df.columns
            assert 'Residual' in fitted_df.columns

    def test_export_reserves_table_to_csv(self):
        """Test exporting reserves table to CSV."""
        csv_data = export_reserves_table_to_csv(self.reserves_table, self.metrics)

        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Read the CSV data
        buffer = BytesIO(csv_data)
        with pd.ExcelFile(buffer) as xls:
            # Check sheets
            assert 'Summary' in xls.sheet_names
            assert 'Reserves Table' in xls.sheet_names

            # Check summary sheet
            summary_df = pd.read_excel(xls, 'Summary')
            assert len(summary_df) > 0
            assert 'Metric' in summary_df.columns
            assert 'Value' in summary_df.columns
            assert 'Unit' in summary_df.columns

            # Check reserves table sheet
            reserves_df = pd.read_excel(xls, 'Reserves Table')
            assert len(reserves_df) == len(self.reserves_table)
            assert 'time' in reserves_df.columns
            assert 'rate' in reserves_df.columns
            assert 'cumulative_production' in reserves_df.columns
            assert 'remaining_reserves' in reserves_df.columns

    def test_export_complete_analysis_to_csv(self):
        """Test exporting complete analysis to CSV."""
        csv_data = export_complete_analysis_to_csv(
            self.results,
            self.reserves_table,
            self.metrics,
            'hyperbolic'
        )

        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Read the CSV data
        buffer = BytesIO(csv_data)
        with pd.ExcelFile(buffer) as xls:
            # Check sheets
            assert 'Summary' in xls.sheet_names
            assert 'Model Comparison' in xls.sheet_names
            assert 'Fitted Data' in xls.sheet_names
            assert 'Reserves Table' in xls.sheet_names

            # Check summary sheet
            summary_df = pd.read_excel(xls, 'Summary')
            assert len(summary_df) > 0
            assert 'Analysis Summary' in summary_df.columns
            assert 'Parameters' in summary_df.columns
            assert 'Metrics' in summary_df.columns

            # Check model comparison sheet
            comparison_df = pd.read_excel(xls, 'Model Comparison')
            assert len(comparison_df) == 4
            assert 'Model' in comparison_df.columns
            assert 'R-squared' in comparison_df.columns
            assert 'RMSE' in comparison_df.columns
            assert 'AIC' in comparison_df.columns
            assert 'BIC' in comparison_df.columns
            assert 'Success' in comparison_df.columns

            # Check fitted data sheet
            fitted_df = pd.read_excel(xls, 'Fitted Data')
            assert len(fitted_df) > 0
            assert 'Time' in fitted_df.columns
            assert 'Actual Rate' in fitted_df.columns
            assert 'Fitted Rate' in fitted_df.columns
            assert 'Residual' in fitted_df.columns

            # Check reserves table sheet
            reserves_df = pd.read_excel(xls, 'Reserves Table')
            assert len(reserves_df) == len(self.reserves_table)
            assert 'time' in reserves_df.columns
            assert 'rate' in reserves_df.columns
            assert 'cumulative_production' in reserves_df.columns
            assert 'remaining_reserves' in reserves_df.columns

    def test_generate_summary_report(self):
        """Test generating summary report."""
        report = generate_summary_report(self.results, self.metrics, 'hyperbolic')

        assert isinstance(report, str)
        assert len(report) > 0

        # Check report sections
        assert "DECLINE CURVE ANALYSIS SUMMARY REPORT" in report
        assert "ANALYSIS INFORMATION:" in report
        assert "MODEL PARAMETERS:" in report
        assert "KEY METRICS:" in report
        assert "MODEL COMPARISON:" in report
        assert "STATISTICAL ANALYSIS:" in report
        assert "ANALYSIS COMPLETE" in report

        # Check specific values
        assert "Selected Model: hyperbolic" in report
        assert "EUR (Estimated Ultimate Recovery): 95000" in report
        assert "Time to Abandonment: 45.0 months" in report
        assert "Effective Decline Rate: 25.00 %/year" in report
        assert "R-squared: 0.98" in report

        # Check model comparison table
        assert "exponential" in report
        assert "hyperbolic" in report
        assert "harmonic" in report

        # Check statistical analysis section
        assert "Monte Carlo Simulation:" in report
        assert "Mean EUR:" in report
        assert "Std Dev:" in report
        assert "95% CI Lower:" in report
        assert "95% CI Upper:" in report
        assert "Standard Error:" in report
        assert "Samples:" in report
        assert "Sensitivity Analysis:" in report

    def test_export_summary_report_to_text(self):
        """Test exporting summary report to text."""
        text_data = export_summary_report_to_text(self.results, self.metrics, 'hyperbolic')

        assert isinstance(text_data, bytes)
        assert len(text_data) > 0

        # Convert to string and check
        report = text_data.decode('utf-8')
        assert "DECLINE CURVE ANALYSIS SUMMARY REPORT" in report
        assert "ANALYSIS INFORMATION:" in report
        assert "MODEL PARAMETERS:" in report
        assert "KEY METRICS:" in report
        assert "MODEL COMPARISON:" in report
        assert "STATISTICAL ANALYSIS:" in report
        assert "ANALYSIS COMPLETE" in report

    def test_export_fitting_results_with_failed_model(self):
        """Test exporting fitting results with failed model."""
        failed_results = {
            'exponential': {'success': False, 'message': 'Fit failed'},
            'hyperbolic': {'success': True, 'parameters': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5}},
            'harmonic': {'success': False, 'message': 'Fit failed'}
        }

        csv_data = export_fitting_results_to_csv(failed_results, 'hyperbolic')

        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Read the CSV data
        buffer = BytesIO(csv_data)
        with pd.ExcelFile(buffer) as xls:
            # Check model comparison sheet
            comparison_df = pd.read_excel(xls, 'Model Comparison')
            assert comparison_df.iloc[1]['Success'] == True  # Hyperbolic should be successful
            assert comparison_df.iloc[0]['Success'] == False  # Exponential should be failed
            assert comparison_df.iloc[2]['Success'] == False  # Harmonic should be failed

    def test_export_with_empty_results(self):
        """Test exporting with empty results."""
        empty_results = {}
        empty_metrics = {}
        empty_reserves_table = pd.DataFrame()

        # Test fitting results export
        csv_data = export_fitting_results_to_csv(empty_results, 'exponential')
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Test reserves table export
        csv_data = export_reserves_table_to_csv(empty_reserves_table, empty_metrics)
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Test complete analysis export
        csv_data = export_complete_analysis_to_csv(empty_results, empty_reserves_table, empty_metrics, 'exponential')
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Test summary report
        report = generate_summary_report(empty_results, empty_metrics, 'exponential')
        assert isinstance(report, str)
        assert len(report) > 0

    def test_export_with_missing_parameters(self):
        """Test exporting with missing parameters."""
        incomplete_results = {
            'exponential': {
                'success': True,
                'parameters': {'Qi': 1000.0},  # Missing Di
                'perr': {'Qi': 50.0},
                'r_squared': 0.95,
                'rmse': 10.0
            }
        }

        csv_data = export_fitting_results_to_csv(incomplete_results, 'exponential')

        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Read the CSV data
        buffer = BytesIO(csv_data)
        with pd.ExcelFile(buffer) as xls:
            summary_df = pd.read_excel(xls, 'Summary')
            # Should handle missing parameters gracefully
            assert 'Di (Decline Rate)' in summary_df['Parameter'].values

    def test_generate_summary_report_with_no_statistical_summary(self):
        """Test generating summary report without statistical summary."""
        results_without_stats = {
            'exponential': {
                'success': True,
                'parameters': {'Qi': 1000.0, 'Di': 0.1},
                'perr': {'Qi': 50.0, 'Di': 0.01},
                'r_squared': 0.95,
                'rmse': 10.0
            }
        }

        report = generate_summary_report(results_without_stats, self.metrics, 'exponential')

        assert isinstance(report, str)
        assert "STATISTICAL ANALYSIS:" not in report  # Should skip this section

    def test_export_with_large_datasets(self):
        """Test exporting with large datasets."""
        large_results = {
            'exponential': {
                'success': True,
                'parameters': {'Qi': 1000000.0, 'Di': 0.01},
                'perr': {'Qi': 5000.0, 'Di': 0.001},
                'r_squared': 0.999,
                'rmse': 1.0,
                'residuals': np.random.normal(0, 1, 10000),
                'fitted_rate': np.random.normal(1000, 10, 10000)
            }
        }

        large_metrics = {
            'Qi': 1000000.0,
            'Di': 0.01,
            'b': 0.0,
            'q_abandon': 10.0,
            'eur': 95000000.0,
            'time_to_abandonment': 450.0,
            'effective_decline_rate': 0.25,
            'initial_period_production': 1000000.0,
            'time_unit': 'months',
            'r_squared': 0.999,
            'rmse': 1.0
        }

        large_reserves_table = pd.DataFrame({
            'time': np.arange(0, 120, 1),
            'rate': np.random.normal(1000, 50, 120),
            'cumulative_production': np.cumsum(np.random.normal(1000, 50, 120)),
            'remaining_reserves': np.random.normal(95000, 1000, 120)
        })

        # Test fitting results export
        csv_data = export_fitting_results_to_csv(large_results, 'exponential')
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Test reserves table export
        csv_data = export_reserves_table_to_csv(large_reserves_table, large_metrics)
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Test complete analysis export
        csv_data = export_complete_analysis_to_csv(
            large_results,
            large_reserves_table,
            large_metrics,
            'exponential'
        )
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

        # Test summary report
        report = generate_summary_report(large_results, large_metrics, 'exponential')
        assert isinstance(report, str)
        assert len(report) > 0

    def test_generate_summary_report_with_special_characters(self):
        """Test generating summary report with special characters."""
        special_results = {
            'exponential': {
                'success': True,
                'parameters': {'Qi': 1000.0, 'Di': 0.1},
                'perr': {'Qi': 50.0, 'Di': 0.01},
                'r_squared': 0.95,
                'rmse': 10.0,
                'message': 'Fit completed with warning: ⚠️'
            }
        }

        special_metrics = {
            'Qi': 1000.0,
            'Di': 0.1,
            'b': 0.0,
            'q_abandon': 10.0,
            'eur': 95000.0,
            'time_to_abandonment': 45.0,
            'effective_decline_rate': 0.25,
            'initial_period_production': 1000.0,
            'time_unit': 'months',
            'r_squared': 0.95,
            'rmse': 10.0
        }

        report = generate_summary_report(special_results, special_metrics, 'exponential')

        assert isinstance(report, str)
        assert "⚠️" in report  # Should handle special characters
        assert "Fit completed with warning" in report