"""
Multi-Well Analysis Module - Handle multiple wells in a single dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .data_loader import load_production_data, preprocess_data
from .fitting import fit_all_models, get_best_model, add_information_criteria
from .calculations import generate_reserves_table, calculate_decline_metrics
from .models import MODEL_DISPLAY_NAMES


def detect_well_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the well identifier column in the dataset.
    
    Args:
        df: DataFrame with production data
        
    Returns:
        Name of the well column if found, None otherwise
    """
    well_column_names = [
        'well', 'well_id', 'wellid', 'well_name', 'wellname',
        'api', 'api_number', 'api_num', 'uwi', 'uwi_number',
        'location', 'field', 'pad', 'lease'
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        for well_name in well_column_names:
            if well_name in col_lower:
                return col
    
    return None


def split_by_well(df: pd.DataFrame, well_column: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Split a dataset into individual wells.
    
    Args:
        df: DataFrame with production data
        well_column: Name of the well identifier column
        
    Returns:
        Dictionary mapping well names to their DataFrames
    """
    if well_column is None:
        well_column = detect_well_column(df)
        if well_column is None:
            # If no well column found, treat entire dataset as one well
            return {'Single_Well': df}
    
    if well_column not in df.columns:
        return {'Single_Well': df}
    
    # Group by well and create separate DataFrames
    well_groups = df.groupby(well_column)
    well_data = {}
    
    for well_name, well_df in well_groups:
        well_data[str(well_name)] = well_df.reset_index(drop=True)
    
    return well_data


def analyze_multi_well(df: pd.DataFrame, 
                      rate_column: str = 'oil_rate',
                      time_unit: str = 'months',
                      q_abandon: float = 10.0) -> Dict:
    """
    Perform decline curve analysis on multiple wells.
    
    Args:
        df: DataFrame with production data (may contain multiple wells)
        rate_column: Column name for the rate data
        time_unit: Time unit for analysis
        q_abandon: Economic abandonment rate threshold
        
    Returns:
        Dictionary with analysis results for all wells
    """
    # Split data by well
    well_column = detect_well_column(df)
    well_data = split_by_well(df, well_column)
    
    results = {
        'well_data': well_data,
        'well_column': well_column,
        'individual_results': {},
        'summary_statistics': {},
        'type_curve': None
    }
    
    # Analyze each well individually
    for well_name, well_df in well_data.items():
        try:
            # Preprocess data for this well
            processed_df = preprocess_data(well_df, time_unit)
            
            # Get time and rate arrays
            time = processed_df['time'].values
            rate = processed_df[rate_column].values
            
            # Filter out zeros for fitting
            valid_mask = rate > 0
            if np.sum(valid_mask) < 3:
                results['individual_results'][well_name] = {
                    'success': False,
                    'error': 'Insufficient valid data points'
                }
                continue
            
            time_valid = time[valid_mask]
            rate_valid = rate[valid_mask]
            
            # Fit all models
            fitting_results = fit_all_models(time_valid, rate_valid)
            fitting_results = add_information_criteria(fitting_results, len(time_valid))
            
            # Get best model
            best_model, best_result = get_best_model(fitting_results)
            
            # Calculate metrics
            metrics = calculate_decline_metrics(
                best_result['parameters'], 
                q_abandon=q_abandon,
                time_unit=time_unit
            )
            
            # Generate reserves table
            params = best_result['parameters']
            max_time = time_valid.max()
            reserves_table = generate_reserves_table(
                params,
                (0, max_time * 2),
                freq='M',
                q_abandon=q_abandon,
                time_unit=time_unit
            )
            
            results['individual_results'][well_name] = {
                'success': True,
                'fitting_results': fitting_results,
                'best_model': best_model,
                'metrics': metrics,
                'reserves_table': reserves_table,
                'processed_data': processed_df,
                'time': time_valid,
                'rate': rate_valid
            }
            
        except Exception as e:
            results['individual_results'][well_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Calculate summary statistics
    results['summary_statistics'] = calculate_multi_well_statistics(results['individual_results'])
    
    # Generate type curve if multiple wells
    if len(well_data) > 1:
        results['type_curve'] = generate_type_curve(results['individual_results'])
    
    return results


def calculate_multi_well_statistics(individual_results: Dict) -> Dict:
    """
    Calculate summary statistics across multiple wells.
    
    Args:
        individual_results: Dictionary of individual well results
        
    Returns:
        Dictionary with summary statistics
    """
    successful_wells = [
        result for result in individual_results.values() 
        if result.get('success', False)
    ]
    
    if not successful_wells:
        return {'error': 'No successful well analyses'}
    
    # Extract parameters for statistics
    qi_values = []
    di_values = []
    b_values = []
    eur_values = []
    r2_values = []
    
    for result in successful_wells:
        metrics = result['metrics']
        qi_values.append(metrics['Qi'])
        di_values.append(metrics['Di'])
        b_values.append(metrics['b'])
        eur_values.append(metrics['eur'])
        r2_values.append(result['fitting_results'][result['best_model']]['r_squared'])
    
    statistics = {
        'total_wells': len(individual_results),
        'successful_wells': len(successful_wells),
        'success_rate': len(successful_wells) / len(individual_results),
        
        'qi_stats': {
            'mean': np.mean(qi_values),
            'std': np.std(qi_values),
            'min': np.min(qi_values),
            'max': np.max(qi_values),
            'p10': np.percentile(qi_values, 90),
            'p50': np.percentile(qi_values, 50),
            'p90': np.percentile(qi_values, 10)
        },
        
        'di_stats': {
            'mean': np.mean(di_values),
            'std': np.std(di_values),
            'min': np.min(di_values),
            'max': np.max(di_values),
            'p10': np.percentile(di_values, 90),
            'p50': np.percentile(di_values, 50),
            'p90': np.percentile(di_values, 10)
        },
        
        'b_stats': {
            'mean': np.mean(b_values),
            'std': np.std(b_values),
            'min': np.min(b_values),
            'max': np.max(b_values),
            'p10': np.percentile(b_values, 90),
            'p50': np.percentile(b_values, 50),
            'p90': np.percentile(b_values, 10)
        },
        
        'eur_stats': {
            'mean': np.mean(eur_values),
            'std': np.std(eur_values),
            'min': np.min(eur_values),
            'max': np.max(eur_values),
            'p10': np.percentile(eur_values, 90),
            'p50': np.percentile(eur_values, 50),
            'p90': np.percentile(eur_values, 10)
        },
        
        'r2_stats': {
            'mean': np.mean(r2_values),
            'std': np.std(r2_values),
            'min': np.min(r2_values),
            'max': np.max(r2_values)
        }
    }
    
    return statistics


def generate_type_curve(individual_results: Dict) -> Dict:
    """
    Generate a type curve by averaging normalized decline curves.
    
    Args:
        individual_results: Dictionary of individual well results
        
    Returns:
        Dictionary with type curve data
    """
    successful_wells = [
        result for result in individual_results.values() 
        if result.get('success', False)
    ]
    
    if len(successful_wells) < 2:
        return {'error': 'Need at least 2 successful wells for type curve'}
    
    # Normalize each well to its initial rate and align time
    normalized_curves = []
    
    for result in successful_wells:
        time = result['time']
        rate = result['rate']
        qi = result['metrics']['Qi']
        
        # Normalize rate to Qi
        normalized_rate = rate / qi
        
        # Create normalized time (0 to 1)
        if len(time) > 1:
            normalized_time = (time - time[0]) / (time[-1] - time[0])
        else:
            normalized_time = np.array([0])
        
        normalized_curves.append({
            'time': normalized_time,
            'rate': normalized_rate,
            'qi': qi
        })
    
    # Create common time grid for averaging
    common_time = np.linspace(0, 1, 100)
    
    # Interpolate each curve to common time grid
    interpolated_rates = []
    for curve in normalized_curves:
        if len(curve['time']) > 1:
            interpolated_rate = np.interp(common_time, curve['time'], curve['rate'])
            interpolated_rates.append(interpolated_rate)
    
    if not interpolated_rates:
        return {'error': 'No valid curves for interpolation'}
    
    # Calculate average and percentiles
    rate_matrix = np.array(interpolated_rates)
    
    type_curve_data = {
        'time': common_time,
        'average_rate': np.mean(rate_matrix, axis=0),
        'p10_rate': np.percentile(rate_matrix, 90, axis=0),
        'p50_rate': np.percentile(rate_matrix, 50, axis=0),
        'p90_rate': np.percentile(rate_matrix, 10, axis=0),
        'std_rate': np.std(rate_matrix, axis=0),
        'count': len(interpolated_rates)
    }
    
    return type_curve_data


def create_multi_well_comparison_chart(individual_results: Dict, 
                                     rate_column: str = 'oil_rate',
                                     unit: str = 'STB/D') -> Dict:
    """
    Create comparison charts for multiple wells.
    
    Args:
        individual_results: Dictionary of individual well results
        rate_column: Column name for the rate data
        unit: Rate unit for y-axis label
        
    Returns:
        Dictionary with chart data
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    successful_wells = [
        (name, result) for name, result in individual_results.items() 
        if result.get('success', False)
    ]
    
    if not successful_wells:
        return {'error': 'No successful well analyses'}
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rate vs Time (Linear)', 'Rate vs Time (Log)', 
                       'EUR Comparison', 'R² Comparison'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot individual well curves
    for i, (well_name, result) in enumerate(successful_wells):
        color = colors[i % len(colors)]
        time = result['time']
        rate = result['rate']
        best_model = result['best_model']
        
        # Linear scale
        fig.add_trace(
            go.Scatter(
                x=time,
                y=rate,
                mode='lines+markers',
                name=f'{well_name} ({best_model})',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                legendgroup=well_name
            ),
            row=1, col=1
        )
        
        # Log scale (filter out zeros)
        valid_mask = rate > 0
        if np.any(valid_mask):
            fig.add_trace(
                go.Scatter(
                    x=time[valid_mask],
                    y=rate[valid_mask],
                    mode='lines+markers',
                    name=f'{well_name} (Log)',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=4),
                    legendgroup=well_name,
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # EUR comparison bar chart
    well_names = [name for name, _ in successful_wells]
    eur_values = [result['metrics']['eur'] for _, result in successful_wells]
    
    fig.add_trace(
        go.Bar(
            x=well_names,
            y=eur_values,
            name='EUR',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # R² comparison bar chart
    r2_values = [result['fitting_results'][result['best_model']]['r_squared'] 
                 for _, result in successful_wells]
    
    fig.add_trace(
        go.Bar(
            x=well_names,
            y=r2_values,
            name='R²',
            marker_color='lightgreen'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Multi-Well Decline Curve Analysis',
        height=800,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (months)", row=1, col=1)
    fig.update_xaxes(title_text="Time (months)", row=1, col=2)
    fig.update_xaxes(title_text="Well Name", row=2, col=1)
    fig.update_xaxes(title_text="Well Name", row=2, col=2)
    
    fig.update_yaxes(title_text=f"Rate ({unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Rate ({unit})", type="log", row=1, col=2)
    fig.update_yaxes(title_text="EUR", row=2, col=1)
    fig.update_yaxes(title_text="R²", row=2, col=2)
    
    return {'figure': fig, 'well_count': len(successful_wells)}