"""
Tests for Multi-Well Analysis Module - Handle multiple wells in a single dataset.
"""
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.multi_well import (
    detect_well_column,
    split_by_well,
    analyze_multi_well,
    calculate_multi_well_statistics,
    generate_type_curve,
    create_multi_well_comparison_chart
)

class TestMultiWell:
    def setup_method(self):
        """Create test data for multi-well analysis."""
        # Create sample data with 3 wells
        data = []
        for well in ['Well_A', 'Well_B', 'Well_C']:
            for month in range(1, 13):
                data.append({
                    'well': well,
                    'date': f'2023-{month:02d}-01',
                    'oil_rate': 1000 - (ord(well[-1]) - ord('A')) * 100 - month * 10,
                    'gas_rate': 50000 - (ord(well[-1]) - ord('A')) * 5000 - month * 500,
                    'water_rate': 50 + (ord(well[-1]) - ord('A')) * 5 + month * 2
                })

        self.multi_well_df = pd.DataFrame(data)
        self.multi_well_df['date'] = pd.to_datetime(self.multi_well_df['date'])

        # Create sample data with different well column names
        self.variation_df = self.multi_well_df.copy()
        self.variation_df.rename(columns={'well': 'Well ID'}, inplace=True)

        # Create sample data with no well column
        self.single_well_df = self.multi_well_df[self.multi_well_df['well'] == 'Well_A'].copy()
        self.single_well_df.drop(columns=['well'], inplace=True)

    def test_detect_well_column(self):
        """Test detecting well column in various formats."""
        # Test with standard well column
        well_col = detect_well_column(self.multi_well_df)
        assert well_col == 'well'

        # Test with variation in column name
        well_col = detect_well_column(self.variation_df)
        assert well_col == 'Well ID'

        # Test with multiple possible well columns
        multi_well_df = self.multi_well_df.copy()
        multi_well_df['well_id'] = multi_well_df['well']
        multi_well_df['well_name'] = multi_well_df['well']
        well_col = detect_well_column(multi_well_df)
        assert well_col in ['well', 'well_id', 'well_name']

        # Test with no well column
        no_well_df = self.multi_well_df.copy()
        no_well_df.drop(columns=['well'], inplace=True)
        well_col = detect_well_column(no_well_df)
        assert well_col is None

    def test_split_by_well(self):
        """Test splitting data by well."""
        well_data = split_by_well(self.multi_well_df)

        assert isinstance(well_data, dict)
        assert len(well_data) == 3
        assert 'Well_A' in well_data
        assert 'Well_B' in well_data
        assert 'Well_C' in well_data

        # Check data for each well
        for well_name, well_df in well_data.items():
            assert len(well_df) == 12
            assert well_df['well'].iloc[0] == well_name

        # Test with specified well column
        well_data = split_by_well(self.variation_df, well_column='Well ID')
        assert len(well_data) == 3
        assert 'Well_A' in well_data

        # Test with no well column (should treat as single well)
        well_data = split_by_well(self.single_well_df)
        assert len(well_data) == 1
        assert 'Single_Well' in well_data
        assert len(well_data['Single_Well']) == 12

    def test_analyze_multi_well(self):
        """Test multi-well analysis."""
        results = analyze_multi_well(
            self.multi_well_df,
            rate_column='oil_rate',
            time_unit='months',
            q_abandon=10.0
        )

        assert isinstance(results, dict)
        assert 'well_data' in results
        assert 'well_column' in results
        assert 'individual_results' in results
        assert 'summary_statistics' in results
        assert 'type_curve' in results

        # Check well data
        assert len(results['well_data']) == 3
        assert results['well_column'] == 'well'

        # Check individual results
        assert len(results['individual_results']) == 3
        for well_name, well_result in results['individual_results'].items():
            assert 'success' in well_result
            assert 'fitting_results' in well_result
            assert 'best_model' in well_result
            assert 'metrics' in well_result
            assert 'reserves_table' in well_result
            assert 'processed_data' in well_result
            assert 'time' in well_result
            assert 'rate' in well_result

            # Check success status
            assert well_result['success'] == True

        # Check summary statistics
        assert 'total_wells' in results['summary_statistics']
        assert 'successful_wells' in results['summary_statistics']
        assert 'success_rate' in results['summary_statistics']
        assert 'qi_stats' in results['summary_statistics']
        assert 'di_stats' in results['summary_statistics']
        assert 'b_stats' in results['summary_statistics']
        assert 'eur_stats' in results['summary_statistics']
        assert 'r2_stats' in results['summary_statistics']

        # Check type curve
        assert results['type_curve'] is not None
        assert 'time' in results['type_curve']
        assert 'average_rate' in results['type_curve']
        assert 'p10_rate' in results['type_curve']
        assert 'p50_rate' in results['type_curve']
        assert 'p90_rate' in results['type_curve']
        assert 'std_rate' in results['type_curve']
        assert 'count' in results['type_curve']

    def test_calculate_multi_well_statistics(self):
        """Test calculating multi-well statistics."""
        # Create mock individual results
        individual_results = {
            'Well_A': {
                'success': True,
                'metrics': {
                    'Qi': 1000.0,
                    'Di': 0.1,
                    'b': 0.5,
                    'eur': 95000.0
                },
                'fitting_results': {
                    'hyperbolic': {
                        'success': True,
                        'r_squared': 0.98
                    }
                },
                'best_model': 'hyperbolic'
            },
            'Well_B': {
                'success': True,
                'metrics': {
                    'Qi': 1200.0,
                    'Di': 0.08,
                    'b': 0.6,
                    'eur': 110000.0
                },
                'fitting_results': {
                    'hyperbolic': {
                        'success': True,
                        'r_squared': 0.96
                    }
                },
                'best_model': 'hyperbolic'
            },
            'Well_C': {
                'success': True,
                'metrics': {
                    'Qi': 800.0,
                    'Di': 0.12,
                    'b': 0.4,
                    'eur': 80000.0
                },
                'fitting_results': {
                    'hyperbolic': {
                        'success': True,
                        'r_squared': 0.94
                    }
                },
                'best_model': 'hyperbolic'
            }
        }

        stats = calculate_multi_well_statistics(individual_results)

        assert isinstance(stats, dict)
        assert stats['total_wells'] == 3
        assert stats['successful_wells'] == 3
        assert stats['success_rate'] == 1.0

        # Check Qi statistics
        assert np.isclose(stats['qi_stats']['mean'], (1000 + 1200 + 800) / 3)
        assert np.isclose(stats['qi_stats']['min'], 800)
        assert np.isclose(stats['qi_stats']['max'], 1200)
        assert np.isclose(stats['qi_stats']['p50'], 1000)

        # Check Di statistics
        assert np.isclose(stats['di_stats']['mean'], (0.1 + 0.08 + 0.12) / 3)
        assert np.isclose(stats['di_stats']['min'], 0.08)
        assert np.isclose(stats['di_stats']['max'], 0.12)
        assert np.isclose(stats['di_stats']['p50'], 0.1)

        # Check EUR statistics
        assert np.isclose(stats['eur_stats']['mean'], (95000 + 110000 + 80000) / 3)
        assert np.isclose(stats['eur_stats']['min'], 80000)
        assert np.isclose(stats['eur_stats']['max'], 110000)
        assert np.isclose(stats['eur_stats']['p50'], 95000)

        # Check R² statistics
        assert np.isclose(stats['r2_stats']['mean'], (0.98 + 0.96 + 0.94) / 3)
        assert np.isclose(stats['r2_stats']['min'], 0.94)
        assert np.isclose(stats['r2_stats']['max'], 0.98)

    def test_generate_type_curve(self):
        """Test generating type curve."""
        # Create mock individual results
        individual_results = {
            'Well_A': {
                'success': True,
                'time': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                'rate': np.array([1000, 950, 903, 858, 816, 776, 738, 702, 668, 636, 604, 575]),
                'metrics': {'Qi': 1000.0}
            },
            'Well_B': {
                'success': True,
                'time': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                'rate': np.array([1200, 1140, 1083, 1029, 978, 930, 884, 841, 799, 760, 722, 686]),
                'metrics': {'Qi': 1200.0}
            },
            'Well_C': {
                'success': True,
                'time': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                'rate': np.array([800, 760, 722, 686, 652, 619, 588, 559, 531, 505, 480, 456]),
                'metrics': {'Qi': 800.0}
            }
        }

        type_curve = generate_type_curve(individual_results)

        assert isinstance(type_curve, dict)
        assert len(type_curve['time']) == 100
        assert len(type_curve['average_rate']) == 100
        assert len(type_curve['p10_rate']) == 100
        assert len(type_curve['p50_rate']) == 100
        assert len(type_curve['p90_rate']) == 100
        assert len(type_curve['std_rate']) == 100
        assert type_curve['count'] == 3

        # Check that time is normalized between 0 and 1
        assert type_curve['time'][0] == 0
        assert type_curve['time'][-1] == 1

        # Check that average rate is reasonable
        assert np.mean(type_curve['average_rate']) > 0
        assert np.mean(type_curve['average_rate']) < 1000

    def test_create_multi_well_comparison_chart(self):
        """Test creating multi-well comparison chart."""
        # Create mock individual results
        individual_results = {
            'Well_A': {
                'success': True,
                'time': np.array([0, 1, 2, 3, 4, 5]),
                'rate': np.array([1000, 950, 903, 858, 816, 776]),
                'best_model': 'hyperbolic',
                'metrics': {'eur': 95000.0}
            },
            'Well_B': {
                'success': True,
                'time': np.array([0, 1, 2, 3, 4, 5]),
                'rate': np.array([1200, 1140, 1083, 1029, 978, 930]),
                'best_model': 'hyperbolic',
                'metrics': {'eur': 110000.0}
            },
            'Well_C': {
                'success': True,
                'time': np.array([0, 1, 2, 3, 4, 5]),
                'rate': np.array([800, 760, 722, 686, 652, 619]),
                'best_model': 'hyperbolic',
                'metrics': {'eur': 80000.0}
            }
        }

        chart_data = create_multi_well_comparison_chart(
            individual_results,
            rate_column='oil_rate',
            unit='STB/D'
        )

        assert isinstance(chart_data, dict)
        assert 'figure' in chart_data
        assert 'well_count' in chart_data

        fig = chart_data['figure']
        assert isinstance(fig, dict)  # Plotly figure is a dictionary
        assert chart_data['well_count'] == 3

        # Check that figure has the expected structure
        assert 'data' in fig
        assert 'layout' in fig
        assert len(fig['data']) > 0

    def test_analyze_multi_well_with_single_well(self):
        """Test multi-well analysis with single well data."""
        results = analyze_multi_well(
            self.single_well_df,
            rate_column='oil_rate',
            time_unit='months',
            q_abandon=10.0
        )

        assert isinstance(results, dict)
        assert 'well_data' in results
        assert 'well_column' in results
        assert 'individual_results' in results
        assert 'summary_statistics' in results
        assert 'type_curve' in results

        # Should treat as single well
        assert len(results['well_data']) == 1
        assert results['well_column'] is None
        assert 'Single_Well' in results['well_data']

        # Should have individual results
        assert len(results['individual_results']) == 1
        well_result = list(results['individual_results'].values())[0]
        assert well_result['success'] == True

        # Should have summary statistics
        assert 'total_wells' in results['summary_statistics']
        assert results['summary_statistics']['total_wells'] == 1

        # Should not have type curve (need multiple wells)
        assert results['type_curve'] is None

    def test_analyze_multi_well_with_insufficient_data(self):
        """Test multi-well analysis with insufficient data."""
        # Create data with only 2 points per well
        minimal_data = []
        for well in ['Well_A', 'Well_B']:
            for month in range(1, 3):
                minimal_data.append({
                    'well': well,
                    'date': f'2023-{month:02d}-01',
                    'oil_rate': 1000 - (ord(well[-1]) - ord('A')) * 100 - month * 10,
                    'gas_rate': 50000 - (ord(well[-1]) - ord('A')) * 5000 - month * 500,
                    'water_rate': 50 + (ord(well[-1]) - ord('A')) * 5 + month * 2
                })

        minimal_df = pd.DataFrame(minimal_data)
        minimal_df['date'] = pd.to_datetime(minimal_df['date'])

        results = analyze_multi_well(
            minimal_df,
            rate_column='oil_rate',
            time_unit='months',
            q_abandon=10.0
        )

        assert isinstance(results, dict)
        assert len(results['individual_results']) == 2

        for well_name, well_result in results['individual_results'].items():
            assert well_result['success'] == False
            assert 'error' in well_result
            assert "Insufficient valid data points" in well_result['error']

    def test_analyze_multi_well_with_invalid_data(self):
        """Test multi-well analysis with invalid data."""
        # Create data with negative rates
        invalid_data = []
        for well in ['Well_A', 'Well_B']:
            for month in range(1, 13):
                invalid_data.append({
                    'well': well,
                    'date': f'2023-{month:02d}-01',
                    'oil_rate': -1000 - (ord(well[-1]) - ord('A')) * 100 - month * 10,  # Negative rate
                    'gas_rate': 50000 - (ord(well[-1]) - ord('A')) * 5000 - month * 500,
                    'water_rate': 50 + (ord(well[-1]) - ord('A')) * 5 + month * 2
                })

        invalid_df = pd.DataFrame(invalid_data)
        invalid_df['date'] = pd.to_datetime(invalid_df['date'])

        results = analyze_multi_well(
            invalid_df,
            rate_column='oil_rate',
            time_unit='months',
            q_abandon=10.0
        )

        assert isinstance(results, dict)
        assert len(results['individual_results']) == 2

        for well_name, well_result in results['individual_results'].items():
            assert well_result['success'] == False
            assert 'error' in well_result
            assert "negative values" in well_result['error']

    def test_calculate_multi_well_statistics_with_mixed_results(self):
        """Test calculating statistics with mixed success/failure results."""
        # Create mock individual results with some failures
        individual_results = {
            'Well_A': {
                'success': True,
                'metrics': {'Qi': 1000.0, 'Di': 0.1, 'b': 0.5, 'eur': 95000.0},
                'fitting_results': {'hyperbolic': {'success': True, 'r_squared': 0.98}},
                'best_model': 'hyperbolic'
            },
            'Well_B': {
                'success': False,
                'error': 'Insufficient data points'
            },
            'Well_C': {
                'success': True,
                'metrics': {'Qi': 800.0, 'Di': 0.12, 'b': 0.4, 'eur': 80000.0},
                'fitting_results': {'hyperbolic': {'success': True, 'r_squared': 0.94}},
                'best_model': 'hyperbolic'
            }
        }

        stats = calculate_multi_well_statistics(individual_results)

        assert isinstance(stats, dict)
        assert stats['total_wells'] == 3
        assert stats['successful_wells'] == 2
        assert stats['success_rate'] == 2/3

        # Should only include successful wells in statistics
        assert np.isclose(stats['qi_stats']['mean'], (1000 + 800) / 2)
        assert np.isclose(stats['eur_stats']['mean'], (95000 + 80000) / 2)
        assert np.isclose(stats['r2_stats']['mean'], (0.98 + 0.94) / 2)

    def test_generate_type_curve_with_single_well(self):
        """Test generating type curve with single well."""
        # Create mock individual results with single well
        individual_results = {
            'Well_A': {
                'success': True,
                'time': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                'rate': np.array([1000, 950, 903, 858, 816, 776, 738, 702, 668, 636, 604, 575]),
                'metrics': {'Qi': 1000.0}
            }
        }

        type_curve = generate_type_curve(individual_results)

        assert isinstance(type_curve, dict)
        assert type_curve['error'] == 'Need at least 2 successful wells for type curve'

    def test_create_multi_well_comparison_chart_with_no_successful_wells(self):
        """Test creating comparison chart with no successful wells."""
        # Create mock individual results with all failures
        individual_results = {
            'Well_A': {
                'success': False,
                'error': 'Insufficient data'
            },
            'Well_B': {
                'success': False,
                'error': 'Invalid data'
            }
        }

        chart_data = create_multi_well_comparison_chart(
            individual_results,
            rate_column='oil_rate',
            unit='STB/D'
        )

        assert isinstance(chart_data, dict)
        assert 'error' in chart_data
        assert chart_data['error'] == 'No successful well analyses'

    def test_analyze_multi_well_with_different_fluid_types(self):
        """Test multi-well analysis with different fluid types."""
        # Create data with different fluid types per well
        data = []
        for well, fluid in [('Well_A', 'oil_rate'), ('Well_B', 'gas_rate'), ('Well_C', 'water_rate')]:
            for month in range(1, 13):
                data.append({
                    'well': well,
                    'date': f'2023-{month:02d}-01',
                    fluid: 1000 - (ord(well[-1]) - ord('A')) * 100 - month * 10,
                    'other_fluid': 50000 - (ord(well[-1]) - ord('A')) * 5000 - month * 500
                })

        multi_fluid_df = pd.DataFrame(data)
        multi_fluid_df['date'] = pd.to_datetime(multi_fluid_df['date'])

        # Test with oil rate
        results_oil = analyze_multi_well(
            multi_fluid_df,
            rate_column='oil_rate',
            time_unit='months',
            q_abandon=10.0
        )
        assert len(results['individual_results']) == 3

        # Test with gas rate
        results_gas = analyze_multi_well(
            multi_fluid_df,
            rate_column='gas_rate',
            time_unit='months',
            q_abandon=10.0
        )
        assert len(results['individual_results']) == 3

        # Test with water rate
        results_water = analyze_multi_well(
            multi_fluid_df,
            rate_column='water_rate',
            time_unit='months',
            q_abandon=10.0
        )
        assert len(results['individual_results']) == 3