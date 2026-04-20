"""
Tests for Data Loader Module - CSV/Excel file loading and validation.
"""
import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from io import BytesIO
from unittest.mock import patch
from src.data_loader import (
    load_production_data,
    validate_production_data,
    preprocess_data,
    get_available_rates,
    detect_data_frequency,
    detect_column_type,
    standardize_columns
)

class TestDataLoader:
    def setup_method(self):
        """Create temporary test files for each test."""
        self.test_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.test_xlsx = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        self.test_invalid = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)

        # Create sample CSV data
        csv_data = """date,oil_rate,gas_rate,water_rate
2023-01-01,1000,50000,50
2023-02-01,950,48000,55
2023-03-01,903,46000,60
2023-04-01,858,44000,65
2023-05-01,816,42000,70
"""
        self.test_csv.write(csv_data.encode('utf-8'))
        self.test_csv.close()

        # Create sample Excel data
        excel_data = """date,oil_rate,gas_rate,water_rate
2023-01-01,1200,60000,40
2023-02-01,1140,57000,42
2023-03-01,1083,54000,44
2023-04-01,1029,51000,46
2023-05-01,978,48000,48
"""
        df = pd.read_csv(BytesIO(excel_data.encode('utf-8')))
        with pd.ExcelWriter(self.test_xlsx.name) as writer:
            df.to_excel(writer, index=False)

        # Create invalid data
        invalid_data = """This is not a valid CSV file
with wrong format
"""
        self.test_invalid.write(invalid_data.encode('utf-8'))
        self.test_invalid.close()

    def teardown_method(self):
        """Clean up temporary files."""
        try:
            os.unlink(self.test_csv.name)
        except:
            pass
        try:
            os.unlink(self.test_xlsx.name)
        except:
            pass
        try:
            os.unlink(self.test_invalid.name)
        except:
            pass

    def test_load_csv_file(self):
        """Test loading CSV file."""
        df = load_production_data(self.test_csv.name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'date' in df.columns
        assert 'oil_rate' in df.columns

    def test_load_excel_file(self):
        """Test loading Excel file."""
        df = load_production_data(self.test_xlsx.name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'date' in df.columns
        assert 'oil_rate' in df.columns

    def test_load_file_object(self):
        """Test loading file object."""
        with open(self.test_csv.name, 'rb') as f:
            df = load_production_data(f)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5

    def test_load_invalid_file_format(self):
        """Test loading invalid file format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_production_data(self.test_invalid.name)

    def test_standardize_columns(self):
        """Test column standardization."""
        # Create DataFrame with different column names
        data = {
            'Production Date': ['2023-01-01', '2023-02-01'],
            'Oil (STB/D)': [1000, 950],
            'Gas (MSCF/D)': [50000, 48000],
            'Water (STB/D)': [50, 55]
        }
        df = pd.DataFrame(data)

        standardized_df = standardize_columns(df)

        assert 'date' in standardized_df.columns
        assert 'oil_rate' in standardized_df.columns
        assert 'gas_rate' in standardized_df.columns
        assert 'water_rate' in standardized_df.columns

    def test_validate_valid_data(self):
        """Test validation of valid data."""
        data = {
            'date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'oil_rate': [1000, 950, 903]
        }
        df = pd.DataFrame(data)

        is_valid, issues = validate_production_data(df)
        assert is_valid
        assert len(issues) == 0

    def test_validate_missing_date_column(self):
        """Test validation with missing date column."""
        data = {
            'oil_rate': [1000, 950, 903]
        }
        df = pd.DataFrame(data)

        is_valid, issues = validate_production_data(df)
        assert not is_valid
        assert "Missing 'date' column" in issues

    def test_validate_missing_rate_column(self):
        """Test validation with missing rate column."""
        data = {
            'date': ['2023-01-01', '2023-02-01', '2023-03-01']
        }
        df = pd.DataFrame(data)

        is_valid, issues = validate_production_data(df)
        assert not is_valid
        assert "Missing rate columns" in issues

    def test_validate_insufficient_data_points(self):
        """Test validation with insufficient data points."""
        data = {
            'date': ['2023-01-01'],
            'oil_rate': [1000]
        }
        df = pd.DataFrame(data)

        is_valid, issues = validate_production_data(df)
        assert not is_valid
        assert "Insufficient data points" in issues

    def test_validate_negative_values(self):
        """Test validation with negative values."""
        data = {
            'date': ['2023-01-01', '2023-02-01'],
            'oil_rate': [1000, -50]
        }
        df = pd.DataFrame(data)

        is_valid, issues = validate_production_data(df)
        assert not is_valid
        assert "negative values" in issues[0]

    def test_preprocess_data(self):
        """Test data preprocessing."""
        data = {
            'date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'oil_rate': [1000, 950, 903]
        }
        df = pd.DataFrame(data)

        processed_df = preprocess_data(df, time_unit='months')

        assert 'time' in processed_df.columns
        assert len(processed_df) == 3
        assert processed_df['time'].iloc[0] == 0
        assert processed_df['time'].iloc[1] > 0

    def test_preprocess_data_with_missing_values(self):
        """Test preprocessing with missing values."""
        data = {
            'date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
            'oil_rate': [1000, None, 903, 858]
        }
        df = pd.DataFrame(data)

        processed_df = preprocess_data(df, time_unit='months')

        # Missing values should be filled
        assert processed_df['oil_rate'].notna().all()

    def test_get_available_rates(self):
        """Test getting available rate columns."""
        data = {
            'date': ['2023-01-01', '2023-02-01'],
            'oil_rate': [1000, 950],
            'gas_rate': [50000, 48000]
        }
        df = pd.DataFrame(data)

        available_rates = get_available_rates(df)
        assert 'oil_rate' in available_rates
        assert 'gas_rate' in available_rates
        assert 'water_rate' not in available_rates

    def test_detect_data_frequency_daily(self):
        """Test daily frequency detection."""
        data = {
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'oil_rate': np.random.rand(10) * 1000
        }
        df = pd.DataFrame(data)

        frequency = detect_data_frequency(df)
        assert frequency == 'daily'

    def test_detect_data_frequency_monthly(self):
        """Test monthly frequency detection."""
        data = {
            'date': pd.date_range(start='2023-01-01', periods=10, freq='ME'),
            'oil_rate': np.random.rand(10) * 1000
        }
        df = pd.DataFrame(data)

        frequency = detect_data_frequency(df)
        assert frequency == 'monthly'

    def test_detect_data_frequency_yearly(self):
        """Test yearly frequency detection."""
        data = {
            'date': pd.date_range(start='2023-01-01', periods=10, freq='YE'),
            'oil_rate': np.random.rand(10) * 1000
        }
        df = pd.DataFrame(data)

        frequency = detect_data_frequency(df)
        assert frequency == 'yearly'

    def test_detect_column_type(self):
        """Test column type detection."""
        assert detect_column_type('date') == 'date'
        assert detect_column_type('oil_rate') == 'oil_rate'
        assert detect_column_type('gas_rate') == 'gas_rate'
        assert detect_column_type('water_rate') == 'water_rate'
        assert detect_column_type('oil_cum') == 'oil_cum'
        assert detect_column_type('bhp') == 'bhp'
        assert detect_column_type('unknown_column') is None

    def test_detect_column_type_with_variations(self):
        """Test column type detection with variations."""
        assert detect_column_type('Date') == 'date'
        assert detect_column_type('Production_Date') == 'date'
        assert detect_column_type('Oil') == 'oil_rate'
        assert detect_column_type('Gas Rate') == 'gas_rate'
        assert detect_column_type('Water Rate') == 'water_rate'
        assert detect_column_type('Cum Oil') == 'oil_cum'
        assert detect_column_type('Bottomhole Pressure') == 'bhp'