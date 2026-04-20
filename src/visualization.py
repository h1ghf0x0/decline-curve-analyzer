"""
Visualization Module - Plotly chart generation for decline curve analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple

from .models import MODEL_DISPLAY_NAMES


def create_rate_time_chart(actual_data: pd.DataFrame, 
                           fitted_curves: Dict,
                           selected_model: str = 'hyperbolic',
                           title: str = 'Production Rate vs Time',
                           rate_column: str = 'oil_rate',
                           unit: str = 'STB/D') -> go.Figure:
    """
    Create a rate vs time chart with actual data and fitted models.
    
    Args:
        actual_data: DataFrame with actual production data (must have 'time' and rate column)
        fitted_curves: Dictionary with fitting results for each model
        selected_model: Name of the selected/best model to highlight
        title: Chart title
        rate_column: Column name for the rate data
        unit: Rate unit for y-axis label
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add actual data
    if rate_column in actual_data.columns:
        fig.add_trace(go.Scatter(
            x=actual_data['time'],
            y=actual_data[rate_column],
            mode='markers',
            name='Actual Data',
            marker=dict(color='black', size=6, symbol='circle'),
            hovertemplate=f'Time: %{{x:.1f}}<br>Rate: %{{y:.1f}} {unit}<extra></extra>'
        ))
    
    # Add fitted curves
    colors = {
        'exponential': '#1f77b4',
        'hyperbolic': '#ff7f0e',
        'harmonic': '#2ca02c'
    }
    
    line_widths = {
        'exponential': 2,
        'hyperbolic': 2,
        'harmonic': 2
    }
    
    # Highlight selected model with dashed line
    line_dash = {
        'exponential': 'solid',
        'hyperbolic': 'solid',
        'harmonic': 'solid'
    }
    line_dash[selected_model] = 'dash'
    line_widths[selected_model] = 3
    
    for model_name, result in fitted_curves.items():
        if result.get('success', False) and 'fitted_rate' in result:
            fig.add_trace(go.Scatter(
                x=actual_data['time'],
                y=result['fitted_rate'],
                mode='lines',
                name=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                line=dict(
                    color=colors.get(model_name, 'gray'),
                    width=line_widths.get(model_name, 2),
                    dash=line_dash.get(model_name, 'solid')
                ),
                hovertemplate=f'Time: %{{x:.1f}}<br>Fitted Rate: %{{y:.1f}} {unit}<extra>{model_name}</extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time (months)',
        yaxis_title=f'Rate ({unit})',
        legend_title='Legend',
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig


def create_log_chart(actual_data: pd.DataFrame,
                     fitted_curves: Dict,
                     selected_model: str = 'hyperbolic',
                     title: str = 'Production Rate vs Time (Log Scale)',
                     rate_column: str = 'oil_rate',
                     unit: str = 'STB/D') -> go.Figure:
    """
    Create a log-scale rate vs time chart.
    
    Args:
        actual_data: DataFrame with actual production data
        fitted_curves: Dictionary with fitting results for each model
        selected_model: Name of the selected/best model to highlight
        title: Chart title
        rate_column: Column name for the rate data
        unit: Rate unit for y-axis label
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add actual data (filter out zeros for log scale)
    if rate_column in actual_data.columns:
        valid_data = actual_data[actual_data[rate_column] > 0]
        fig.add_trace(go.Scatter(
            x=valid_data['time'],
            y=valid_data[rate_column],
            mode='markers',
            name='Actual Data',
            marker=dict(color='black', size=6, symbol='circle'),
            hovertemplate=f'Time: %{{x:.1f}}<br>Rate: %{{y:.1f}} {unit}<extra></extra>'
        ))
    
    # Add fitted curves
    colors = {
        'exponential': '#1f77b4',
        'hyperbolic': '#ff7f0e',
        'harmonic': '#2ca02c'
    }
    
    for model_name, result in fitted_curves.items():
        if result.get('success', False) and 'fitted_rate' in result:
            # Filter out zero/negative values for log scale
            valid_fit = result['fitted_rate'] > 0
            if np.any(valid_fit):
                fig.add_trace(go.Scatter(
                    x=actual_data['time'][valid_fit],
                    y=result['fitted_rate'][valid_fit],
                    mode='lines',
                    name=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    line=dict(
                        color=colors.get(model_name, 'gray'),
                        width=3 if model_name == selected_model else 2,
                        dash='dash' if model_name == selected_model else 'solid'
                    ),
                    hovertemplate=f'Time: %{{x:.1f}}<br>Fitted Rate: %{{y:.1f}} {unit}<extra>{model_name}</extra>'
                ))
    
    # Update layout with log scale
    fig.update_layout(
        title=title,
        xaxis_title='Time (months)',
        yaxis_title=f'Rate ({unit})',
        yaxis_type='log',
        legend_title='Legend',
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig


def create_residuals_chart(fitted_curves: Dict,
                           title: str = 'Residuals Analysis') -> go.Figure:
    """
    Create a residuals plot for model diagnostics.
    
    Args:
        fitted_curves: Dictionary with fitting results for each model
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=('Residuals vs Time', 'Residuals Distribution')
    )
    
    colors = {
        'exponential': '#1f77b4',
        'hyperbolic': '#ff7f0e',
        'harmonic': '#2ca02c'
    }
    
    for model_name, result in fitted_curves.items():
        if result.get('success', False) and 'residuals' in result:
            residuals = result['residuals']
            time = np.arange(len(residuals))
            
            # Residuals vs Time
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=residuals,
                    mode='markers',
                    name=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    marker=dict(color=colors.get(model_name, 'gray'), size=4),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Residuals Distribution (histogram)
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name=f'{MODEL_DISPLAY_NAMES.get(model_name, model_name)} (dist)',
                    marker_color=colors.get(model_name, 'gray'),
                    opacity=0.6,
                    nbinsx=20,
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    fig.update_layout(
        title=title,
        xaxis2_title='Residual Value',
        yaxis1_title='Residual',
        yaxis2_title='Count',
        template='plotly_white',
        barmode='overlay'
    )
    
    return fig


def create_model_comparison_chart(fitted_curves: Dict,
                                   title: str = 'Model Comparison') -> go.Figure:
    """
    Create a chart comparing R-squared values for all models.
    
    Args:
        fitted_curves: Dictionary with fitting results for each model
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    models = []
    r2_values = []
    rmse_values = []
    
    for model_name, result in fitted_curves.items():
        if result.get('success', False):
            models.append(MODEL_DISPLAY_NAMES.get(model_name, model_name))
            r2_values.append(result.get('r_squared', 0))
            rmse_values.append(result.get('rmse', 0))
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('R-squared', 'RMSE'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # R-squared bar chart
    fig.add_trace(
        go.Bar(
            x=models,
            y=r2_values,
            name='R-squared',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)],
            text=[f'{r2:.4f}' for r2 in r2_values],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # RMSE bar chart
    fig.add_trace(
        go.Bar(
            x=models,
            y=rmse_values,
            name='RMSE',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)]
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        yaxis1_title='R-squared',
        yaxis2_title='RMSE',
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text='Model', row=1, col=1)
    fig.update_xaxes(title_text='Model', row=1, col=2)
    
    return fig


def create_cumulative_production_chart(actual_data: pd.DataFrame,
                                        fitted_result: Dict,
                                        title: str = 'Cumulative Production',
                                        rate_column: str = 'oil_rate',
                                        unit: str = 'STB') -> go.Figure:
    """
    Create a cumulative production chart.
    
    Args:
        actual_data: DataFrame with actual production data
        fitted_result: Fitting result dictionary
        title: Chart title
        rate_column: Column name for the rate data
        unit: Cumulative unit for y-axis label
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Calculate actual cumulative production
    if rate_column in actual_data.columns:
        actual_cum = actual_data[rate_column].cumsum()
        fig.add_trace(go.Scatter(
            x=actual_data['time'],
            y=actual_cum,
            mode='lines',
            name='Actual Cumulative',
            line=dict(color='black', width=2),
            hovertemplate=f'Time: %{{x:.1f}}<br>Cumulative: %{{y:.1f}} {unit}<extra></extra>'
        ))
    
    # Calculate fitted cumulative production
    if fitted_result.get('success', False):
        params = fitted_result.get('parameters', {})
        qi = params.get('Qi', 0)
        di = params.get('Di', 0)
        b = params.get('b', 0)
        
        # Use the same cumulative calculation as in calculations.py
        time = actual_data['time'].values
        if b == 0:
            cum_prod = (qi / di) * (1.0 - np.exp(-di * time))
        elif b == 1:
            cum_prod = (qi / di) * np.log(1.0 + di * time)
        else:
            from .models import hyperbolic_decline
            q_t = hyperbolic_decline(time, qi, di, b)
            cum_prod = (qi ** b / (di * (1.0 - b))) * (qi ** (1.0 - b) - q_t ** (1.0 - b))
        
        fig.add_trace(go.Scatter(
            x=time,
            y=cum_prod,
            mode='lines',
            name='Fitted Cumulative',
            line=dict(color='blue', width=2, dash='dash'),
            hovertemplate=f'Time: %{{x:.1f}}<br>Cumulative: %{{y:.1f}} {unit}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (months)',
        yaxis_title=f'Cumulative Production ({unit})',
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig


def create_decline_rate_chart(fitted_result: Dict,
                               time_range: Tuple[float, float],
                               title: str = 'Decline Rate vs Time') -> go.Figure:
    """
    Create a chart showing the decline rate over time.
    
    Args:
        fitted_result: Fitting result dictionary
        time_range: Tuple of (start_time, end_time)
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    if not fitted_result.get('success', False):
        return go.Figure().add_annotation(text="No valid fitting result", showarrow=False)
    
    params = fitted_result.get('parameters', {})
    qi = params.get('Qi', 0)
    di = params.get('Di', 0)
    b = params.get('b', 0)
    
    time = np.linspace(time_range[0], time_range[1], 100)
    
    # Calculate decline rate
    if b == 0:
        decline_rate = np.full_like(time, di)
    else:
        decline_rate = di / (1.0 + b * di * time)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=decline_rate,
        mode='lines',
        name='Decline Rate',
        line=dict(color='blue', width=2),
        hovertemplate='Time: %{{x:.1f}}<br>Decline Rate: %{{y:.4f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (months)',
        yaxis_title='Decline Rate (1/month)',
        template='plotly_white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig