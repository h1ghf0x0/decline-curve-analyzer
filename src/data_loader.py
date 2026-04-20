"""
Data Loader Module - CSV/Excel file loading and validation for production data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


# Required columns for production data
REQUIRED_COLUMNS = ['date']
RATE_COLUMNS = ['oil_rate', 'gas_rate', 'water_rate', 'liquid_rate']
OPTIONAL_COLUMNS = ['oil_cum', 'gas_cum', 'water_cum', 'bhp', 'thp']


# Column name mappings for common variations
COLUMN_MAPPINGS = {
    'date': ['date', 'time', 'datetime', 'production_date', 'prod_date', 'Date', 'Time', 'Production Date', 'Production_Date', 'Production Date', 'Production_Date'],
    'oil_rate': ['oil_rate', 'oil', 'oil_rate_bopd', 'qo', 'oil_rate_stb', 'Oil', 'Oil Rate', 'QO', 'Oil (STB/D)', 'Oil (STB/D)', 'Oil (STB/D)', 'Oil (STB/D)'],
    'gas_rate': ['gas_rate', 'gas', 'gas_rate_mscfd', 'qg', 'gas_rate_mscf', 'Gas', 'Gas Rate', 'QG', 'Gas (MSCF/D)', 'Gas (MSCF/D)', 'Gas (MSCF/D)', 'Gas (MSCF/D)'],
    'water_rate': ['water_rate', 'water', 'water_rate_bwpd', 'qw', 'water_rate_stb', 'Water', 'Water Rate', 'QW', 'Water (STB/D)', 'Water (STB/D)', 'Water (STB/D)', 'Water (STB/D)'],
    'liquid_rate': ['liquid_rate', 'liquid', 'total_liquid', 'ql', 'Liquid', 'Liquid Rate'],
    'oil_cum': ['oil_cum', 'oil_cumulative', 'npt', 'Oil Cum', 'Cum Oil'],
    'gas_cum': ['gas_cum', 'gas_cumulative', 'gpt', 'Gas Cum', 'Cum Gas'],
    'water_cum': ['water_cum', 'water_cumulative', 'wpt', 'Water Cum', 'Cum Water'],
    'bhp': ['bhp', 'bottomhole_pressure', 'BHP', 'Bottomhole Pressure'],
    'thp': ['thp', 'tubing_pressure', 'THP', 'Tubing Pressure'],
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to match expected format.

    Args:
        df: DataFrame with production data

    Returns:
        DataFrame with standardized column names
    """
    standardized_df = df.copy()

    # Standardize column names
    for col in df.columns:
        standard_col = detect_column_type(col)
        if standard_col:
            standardized_df.rename(columns={col: standard_col}, inplace=True)

    # Ensure all required columns are present
    for required_col in REQUIRED_COLUMNS:
        if required_col not in standardized_df.columns:
            standardized_df[required_col] = None

    # Ensure all rate columns are present
    for rate_col in RATE_COLUMNS:
        if rate_col not in standardized_df.columns:
            standardized_df[rate_col] = None

    # Remove any columns that weren't mapped (keep only recognized columns)
    all_valid_columns = ['date'] + RATE_COLUMNS + OPTIONAL_COLUMNS
    existing_valid = [col for col in all_valid_columns if col in standardized_df.columns]
    standardized_df = standardized_df[existing_valid]

    return standardized_df


def detect_column_type(column_name: str) -> Optional[str]:
    """
    Detect the type of a column based on its name.

    Args:
        column_name: Name of the column to detect

    Returns:
        Standard column type or None if not recognized
    """
    column_name_lower = column_name.strip().lower()
    print(f"Checking column: '{column_name}' (lowercase: '{column_name_lower}')")
    for standard_name, variations in COLUMN_MAPPINGS.items():
        print(f"  Checking against standard: '{standard_name}'")
        for variation in variations:
            print(f"    Variation: '{variation}' (lowercase: '{variation.lower()}')")
            if column_name_lower == variation.lower():
                print(f"    MATCH FOUND: {standard_name}")
                return standard_name
        print(f"    No match for '{standard_name}'")
    print(f"  No match found for column: '{column_name}'")
    return None


def load_production_data(file) -> pd.DataFrame:
    """
    Load production data from CSV or Excel file.
    
    Args:
        file: File object or path to CSV/Excel file
        
    Returns:
        DataFrame with standardized column names
        
    Raises:
        ValueError: If file format is not supported or no valid data found
    """
    # Determine file type and load
    if isinstance(file, str):
        file_path = file.lower()
        if file_path.endswith('.csv'):
            df = pd.read_csv(file)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel.")
    else:
        # Try to read as CSV first, then Excel
        try:
            df = pd.read_csv(file)
        except Exception:
            try:
                file.seek(0)
                df = pd.read_excel(file)
            except Exception:
                raise ValueError("Could not read file as CSV or Excel.")
    
    return standardize_columns(df)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to match expected format.
    
    Args:
        df: Input DataFrame with various column naming conventions
        
    Returns:
        DataFrame with standardized column names
    """
    # Create mapping of current columns to standard names
    column_map = {}
    for col in df.columns:
        standard_name = detect_column_type(col)
        if standard_name:
            column_map[col] = standard_name
    
    # Rename columns
    df = df.rename(columns=column_map)
    
    # Remove any columns that weren't mapped (keep only recognized columns)
    all_valid_columns = ['date'] + RATE_COLUMNS + OPTIONAL_COLUMNS
    existing_valid = [col for col in all_valid_columns if col in df.columns]
    df = df[existing_valid]
    
    return df


def validate_production_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate that the DataFrame contains required production data.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for date column
    if 'date' not in df.columns:
        issues.append("Missing 'date' column")
    else:
        # Try to convert to datetime
        try:
            pd.to_datetime(df['date'])
        except Exception as e:
            issues.append(f"Cannot parse date column: {str(e)}")

    # Check for at least one rate column
    has_rate = any(col in df.columns for col in RATE_COLUMNS)
    if not has_rate:
        issues.append("Missing rate columns")

    # Check for negative values in rate columns
    for col in RATE_COLUMNS:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append("negative values")

    # Check for sufficient data points
    if len(df) < 3:
        issues.append("Insufficient data points")

    # Check for all-zero rate columns
    for col in RATE_COLUMNS:
        if col in df.columns:
            if (df[col] == 0).all():
                issues.append("All values are zero in rate column")

    return len(issues) == 0, issues


def preprocess_data(df: pd.DataFrame, time_unit: str = 'months') -> pd.DataFrame:
    """
    Clean and preprocess production data for DCA analysis.
    
    Args:
        df: Input DataFrame
        time_unit: Time unit for analysis ('days', 'months', 'years')
        
    Returns:
        Cleaned and preprocessed DataFrame
    """
    df = df.copy()
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate time from first production date
    start_date = df['date'].min()
    
    if time_unit == 'days':
        df['time'] = (df['date'] - start_date).dt.days
    elif time_unit == 'months':
        df['time'] = (df['date'] - start_date).dt.days / 30.4375  # Average days per month
    elif time_unit == 'years':
        df['time'] = (df['date'] - start_date).dt.days / 365.25
    else:
        raise ValueError(f"Invalid time_unit: {time_unit}")
    
    # Handle missing values - forward fill then backward fill for rates
    for col in RATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Remove rows where all rate columns are NaN or zero
    rate_cols_present = [col for col in RATE_COLUMNS if col in df.columns]
    if rate_cols_present:
        mask = df[rate_cols_present].isna().all(axis=1) | (df[rate_cols_present].sum(axis=1) == 0)
        df = df[~mask].reset_index(drop=True)
    
    return df


def get_available_rates(df: pd.DataFrame) -> list:
    """
    Get list of available rate columns in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of available rate column names
    """
    return [col for col in RATE_COLUMNS if col in df.columns]


def detect_data_frequency(df: pd.DataFrame) -> str:
    """
    Detect the frequency of production data (daily, monthly, yearly).
    
    Args:
        df: DataFrame with datetime 'date' column
        
    Returns:
        Detected frequency string
    """
    if 'date' not in df.columns:
        return 'unknown'
    
    dates = pd.to_datetime(df['date']).sort_values()
    if len(dates) < 2:
        return 'unknown'
    
    # Calculate median time difference
    time_diffs = dates.diff().dropna()
    median_diff = time_diffs.median()
    
    # Classify frequency
    days = median_diff.days
    
    if days <= 2:
        return 'daily'
    elif days <= 40:
        return 'monthly'
    elif days <= 400:
        return 'yearly'
    else:
        return 'irregular'