"""
Export Module - CSV export functionality for results and reserves tables.
"""

import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from .calculations import generate_reserves_table, calculate_decline_curve_summary


def export_fitting_results_to_csv(results: Dict,
                                  model_name: str,
                                  filename: Optional[str] = None) -> bytes:
    """
    Export fitting results to CSV format.

    Args:
        results: Dictionary with fitting results for all models
        model_name: Name of the selected model
        filename: Optional filename for the export

    Returns:
        Bytes object containing CSV data
    """
    # Get the selected model result
    selected_result = results.get(model_name, {})

    # Create summary data
    summary_data = []

    # Model parameters
    params = selected_result.get('parameters', {})
    summary_data.append(['Parameter', 'Value', 'Error'])
    summary_data.append(['Qi (Initial Rate)', params.get('Qi', 0), selected_result.get('perr', {}).get('Qi', 0)])
    summary_data.append(['Di (Decline Rate)', params.get('Di', 0), selected_result.get('perr', {}).get('Di', 0)])

    if 'b' in params:
        summary_data.append(['b (Decline Exponent)', params.get('b', 0), selected_result.get('perr', {}).get('b', 0)])

    summary_data.append(['', '', ''])

    # Model statistics
    summary_data.append(['Statistic', 'Value', ''])
    summary_data.append(['R-squared', selected_result.get('r_squared', 0), ''])
    summary_data.append(['RMSE', selected_result.get('rmse', 0), ''])
    summary_data.append(['AIC', selected_result.get('aic', 0), ''])
    summary_data.append(['BIC', selected_result.get('bic', 0), ''])
    summary_data.append(['Success', selected_result.get('success', False), ''])
    summary_data.append(['Message', selected_result.get('message', ''), ''])

    # Create DataFrame
    df_summary = pd.DataFrame(summary_data[1:], columns=summary_data[0])

    # Create comparison data for all models
    comparison_data = []
    comparison_data.append(['Model', 'R-squared', 'RMSE', 'AIC', 'BIC', 'Success'])

    for model, result in results.items():
        comparison_data.append([
            model,
            result.get('r_squared', 0),
            result.get('rmse', 0),
            result.get('aic', 0),
            result.get('bic', 0),
            result.get('success', False)
        ])

    df_comparison = pd.DataFrame(comparison_data[1:], columns=comparison_data[0])

    # Create fitted data
    if selected_result.get('success', False):
        fitted_data = []
        fitted_data.append(['Time', 'Actual Rate', 'Fitted Rate', 'Residual'])

        time = np.arange(len(selected_result.get('residuals', [])))
        residuals = selected_result.get('residuals', [])
        fitted_rate = selected_result.get('fitted_rate', [])

        for i, t in enumerate(time):
            fitted_data.append([
                t,
                fitted_rate[i] + residuals[i] if i < len(residuals) else 0,  # Reconstruct actual
                fitted_rate[i] if i < len(fitted_rate) else 0,
                residuals[i] if i < len(residuals) else 0
            ])

        df_fitted = pd.DataFrame(fitted_data[1:], columns=fitted_data[0])
    else:
        df_fitted = pd.DataFrame(columns=['Time', 'Actual Rate', 'Fitted Rate', 'Residual'])

    # Write to BytesIO buffer
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_comparison.to_excel(writer, sheet_name='Model Comparison', index=False)
        df_fitted.to_excel(writer, sheet_name='Fitted Data', index=False)

    buffer.seek(0)
    return buffer.getvalue()


def export_reserves_table_to_csv(reserves_table: pd.DataFrame,
                                 metrics: Dict,
                                 filename: Optional[str] = None) -> bytes:
    """
    Export reserves table to CSV format.

    Args:
        reserves_table: DataFrame with reserves table data
        metrics: Dictionary with decline curve metrics
        filename: Optional filename for the export

    Returns:
        Bytes object containing CSV data
    """
    # Create summary data
    summary_data = []
    summary_data.append(['Metric', 'Value', 'Unit'])
    summary_data.append(['Qi (Initial Rate)', metrics.get('Qi', 0), 'STB/D'])
    summary_data.append(['Di (Decline Rate)', metrics.get('Di', 0), '1/month'])
    summary_data.append(['b (Decline Exponent)', metrics.get('b', 0), ''])
    summary_data.append(['EUR (Estimated Ultimate Recovery)', metrics.get('eur', 0), 'STB'])
    summary_data.append(['Time to Abandonment', metrics.get('time_to_abandonment', 0), 'months'])
    summary_data.append(['Effective Decline Rate', metrics.get('effective_decline_rate', 0), '%/year'])
    summary_data.append(['R-squared', metrics.get('r_squared', 0), ''])
    summary_data.append(['RMSE', metrics.get('rmse', 0), 'STB/D'])

    df_summary = pd.DataFrame(summary_data[1:], columns=summary_data[0])

    # Write to BytesIO buffer
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        reserves_table.to_excel(writer, sheet_name='Reserves Table', index=False)

    buffer.seek(0)
    return buffer.getvalue()


def export_complete_analysis_to_csv(results: Dict,
                                    reserves_table: pd.DataFrame,
                                    metrics: Dict,
                                    selected_model: str,
                                    filename: Optional[str] = None) -> bytes:
    """
    Export complete analysis to CSV format with all sheets.

    Args:
        results: Dictionary with fitting results for all models
        reserves_table: DataFrame with reserves table data
        metrics: Dictionary with decline curve metrics
        selected_model: Name of the selected model
        filename: Optional filename for the export

    Returns:
        Bytes object containing CSV data
    """
    # Get the selected model result
    selected_result = results.get(selected_model, {})

    # Create summary sheet
    summary_data = []
    summary_data.append(['Analysis Summary', '', ''])
    summary_data.append(['Selected Model', selected_model, ''])
    summary_data.append(['', '', ''])
    summary_data.append(['Parameters', 'Value', 'Error'])
    params = selected_result.get('parameters', {})
    summary_data.append(['Qi (Initial Rate)', params.get('Qi', 0), selected_result.get('perr', {}).get('Qi', 0)])
    summary_data.append(['Di (Decline Rate)', params.get('Di', 0), selected_result.get('perr', {}).get('Di', 0)])

    if 'b' in params:
        summary_data.append(['b (Decline Exponent)', params.get('b', 0), selected_result.get('perr', {}).get('b', 0)])

    summary_data.append(['', '', ''])
    summary_data.append(['Metrics', 'Value', 'Unit'])
    summary_data.append(['EUR', metrics.get('eur', 0), 'STB'])
    summary_data.append(['Time to Abandonment', metrics.get('time_to_abandonment', 0), 'months'])
    summary_data.append(['Effective Decline Rate', metrics.get('effective_decline_rate', 0), '%/year'])
    summary_data.append(['R-squared', selected_result.get('r_squared', 0), ''])
    summary_data.append(['RMSE', selected_result.get('rmse', 0), 'STB/D'])
    summary_data.append(['AIC', selected_result.get('aic', 0), ''])
    summary_data.append(['BIC', selected_result.get('bic', 0), ''])

    df_summary = pd.DataFrame(summary_data[1:], columns=summary_data[0])

    # Create model comparison sheet
    comparison_data = []
    comparison_data.append(['Model', 'R-squared', 'RMSE', 'AIC', 'BIC', 'Success'])

    for model, result in results.items():
        comparison_data.append([
            model,
            result.get('r_squared', 0),
            result.get('rmse', 0),
            result.get('aic', 0),
            result.get('bic', 0),
            result.get('success', False)
        ])

    df_comparison = pd.DataFrame(comparison_data[1:], columns=comparison_data[0])

    # Create fitted data sheet
    fitted_data = []
    fitted_data.append(['Time', 'Actual Rate', 'Fitted Rate', 'Residual'])

    if selected_result.get('success', False):
        time = np.arange(len(selected_result.get('residuals', [])))
        residuals = selected_result.get('residuals', [])
        fitted_rate = selected_result.get('fitted_rate', [])

        for i, t in enumerate(time):
            fitted_data.append([
                t,
                fitted_rate[i] + residuals[i] if i < len(residuals) else 0,  # Reconstruct actual
                fitted_rate[i] if i < len(fitted_rate) else 0,
                residuals[i] if i < len(residuals) else 0
            ])

    df_fitted = pd.DataFrame(fitted_data[1:], columns=fitted_data[0])

    # Create reserves table sheet
    df_reserves = reserves_table.copy()

    # Write to BytesIO buffer
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_comparison.to_excel(writer, sheet_name='Model Comparison', index=False)
        df_fitted.to_excel(writer, sheet_name='Fitted Data', index=False)
        df_reserves.to_excel(writer, sheet_name='Reserves Table', index=False)

    buffer.seek(0)
    return buffer.getvalue()


def generate_summary_report(results: Dict, metrics: Dict, selected_model: str) -> str:
    """
    Generate a text summary report of the decline curve analysis.

    Args:
        results: Dictionary with fitting results for all models
        metrics: Dictionary with decline curve metrics
        selected_model: Name of the selected model

    Returns:
        String containing the summary report
    """
    report = []
    report.append("=" * 60)
    report.append("DECLINE CURVE ANALYSIS SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")

    # Basic information
    report.append("ANALYSIS INFORMATION:")
    report.append(f"Selected Model: {selected_model}")
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Model parameters
    selected_result = results.get(selected_model, {})
    params = selected_result.get('parameters', {})

    report.append("MODEL PARAMETERS:")
    report.append(f"Qi (Initial Rate): {params.get('Qi', 0):.2f} STB/D")
    report.append(f"Di (Decline Rate): {params.get('Di', 0):.4f} 1/month")

    if 'b' in params:
        report.append(f"b (Decline Exponent): {params.get('b', 0):.4f}")
    report.append("")

    # Key metrics
    report.append("KEY METRICS:")
    report.append(f"EUR (Estimated Ultimate Recovery): {metrics.get('eur', 0):.0f} STB")
    report.append(f"Time to Abandonment: {metrics.get('time_to_abandonment', 0):.1f} months")
    report.append(f"Effective Decline Rate: {metrics.get('effective_decline_rate', 0)*100:.2f} %/year")
    report.append(f"R-squared: {selected_result.get('r_squared', 0):.4f}")
    report.append(f"RMSE: {selected_result.get('rmse', 0):.2f} STB/D")
    report.append("")

    # Model comparison
    report.append("MODEL COMPARISON:")
    report.append(f"{'Model':<15} {'R-squared':<12} {'RMSE':<10} {'AIC':<10} {'Success'}")
    report.append("-" * 60)

    for model, result in results.items():
        r2 = result.get('r_squared', 0)
        rmse = result.get('rmse', 0)
        aic = result.get('aic', 0)
        success = "Yes" if result.get('success', False) else "No"
        report.append(f"{model:<15} {r2:<12.4f} {rmse:<10.2f} {aic:<10.2f} {success}")

    # Add statistical analysis if available
    if 'statistical_summary' in results:
        stats = results['statistical_summary'].get(selected_model, {})
        report.append("")
        report.append("STATISTICAL ANALYSIS:")
        report.append("Monte Carlo Simulation:")
        report.append(f"  Mean EUR: {stats.get('mc_mean', 0):,.0f} STB")
        report.append(f"  Std Dev: {stats.get('mc_std', 0):,.0f} STB")
        report.append(f"  95% CI Lower: {stats.get('ci_lower', 0):,.0f} STB")
        report.append(f"  95% CI Upper: {stats.get('ci_upper', 0):,.0f} STB")
        report.append(f"  Standard Error: {stats.get('ci_std', 0):,.0f} STB")
        report.append(f"  Samples: {stats.get('mc_samples', 0)}")

        report.append("Sensitivity Analysis:")
        report.append("  First-Order Indices (S1):")
        report.append(f"    Qi (Initial Rate): {stats.get('sensitivity_S1', [])[0]:.3f}")
        report.append(f"    Di (Decline Rate): {stats.get('sensitivity_S1', [])[1]:.3f}")
        report.append(f"    b (Decline Exponent): {stats.get('sensitivity_S1', [])[2]:.3f}")
        report.append("  Total-Order Indices (ST):")
        report.append(f"    Qi: {stats.get('sensitivity_ST', [])[0]:.3f}")
        report.append(f"    Di: {stats.get('sensitivity_ST', [])[1]:.3f}")
        report.append(f"    b: {stats.get('sensitivity_ST', [])[2]:.3f}")

    report.append("")
    report.append("ANALYSIS COMPLETE")
    report.append("=" * 60)

    return "\n".join(report)


def export_summary_report_to_text(results: Dict, metrics: Dict, selected_model: str) -> bytes:
    """
    Export summary report to text format.

    Args:
        results: Dictionary with fitting results for all models
        metrics: Dictionary with decline curve metrics
        selected_model: Name of the selected model

    Returns:
        Bytes object containing text data
    """
    report = generate_summary_report(results, metrics, selected_model)
    return report.encode('utf-8')