"""
Decline Curve Analyzer - Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Import from local modules
from src.data_loader import (
    load_production_data, validate_production_data, preprocess_data,
    get_available_rates, detect_data_frequency
)
from src.fitting import fit_all_models, get_best_model, add_information_criteria, fit_arps_model_with_statistics
from src.calculations import (
    generate_reserves_table, calculate_decline_metrics,
    calculate_decline_curve_summary
)
from src.visualization import (
    create_rate_time_chart, create_log_chart, create_residuals_chart,
    create_model_comparison_chart, create_cumulative_production_chart
)
from src.models import ARPS_MODELS
from src.exports import (
    export_complete_analysis_to_csv, generate_summary_report
)
from src.multi_well import (
    detect_well_column, analyze_multi_well, create_multi_well_comparison_chart
)


# Page configuration
st.set_page_config(
    page_title="Decline Curve Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'hyperbolic'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'time_unit' not in st.session_state:
    st.session_state.time_unit = 'months'
if 'q_abandon' not in st.session_state:
    st.session_state.q_abandon = 10.0
if 'fluid_type' not in st.session_state:
    st.session_state.fluid_type = 'oil_rate'
if 'results_displayed' not in st.session_state:
    st.session_state.results_displayed = False


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Decline Curve Analyzer</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
            Arps DCA (Decline Curve Analysis) for Oil & Gas Production Forecasting
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Production Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with production data"
        )

        # Store uploaded file in session state
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.data_loaded = True
        else:
            st.session_state.data_loaded = False

        st.divider()

        # Analysis parameters
        st.subheader("Analysis Parameters")

        time_unit = st.selectbox(
            "Time Unit",
            options=['months', 'days', 'years'],
            index=0,
            help="Time unit for decline curve analysis"
        )
        st.session_state.time_unit = time_unit

        q_abandon = st.number_input(
            "Abandonment Rate",
            value=10.0,
            min_value=0.1,
            max_value=1000.0,
            step=0.1,
            help="Economic abandonment rate threshold"
        )
        st.session_state.q_abandon = q_abandon

        st.divider()

        # Fluid selection (will be populated after data load)
        st.subheader("Fluid Selection")
        fluid_type = st.selectbox(
            "Select Fluid",
            options=['oil_rate', 'gas_rate', 'water_rate', 'liquid_rate'],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select which fluid to analyze"
        )
        st.session_state.fluid_type = fluid_type

        # Check if data contains multiple wells
        if uploaded_file and st.session_state.get('data_loaded', False):
            df = st.session_state.get('loaded_data', None)
            if df is not None:
                well_column = detect_well_column(df)
                if well_column:
                    st.info(f"📊 **Multi-Well Data Detected**: Found {len(df[well_column].unique())} wells")
                    st.session_state['multi_well_analysis'] = True
                else:
                    st.session_state['multi_well_analysis'] = False

        # Unit display
        unit_labels = {
            'oil_rate': 'STB/D',
            'gas_rate': 'MSCF/D',
            'water_rate': 'STB/D',
            'liquid_rate': 'STB/D'
        }

        st.divider()

        # Download section
        st.subheader("📥 Export Results")

        if st.session_state.analysis_complete:
            # Export complete analysis
            export_format = st.selectbox(
                "Export Format",
                options=['Excel (XLSX)', 'Text Report'],
                index=0
            )

            if st.button("Download Results", type="primary"):
                download_results(export_format)
        else:
            st.info("Complete analysis to enable download")

        # Reset button
        if st.button("Reset Analysis"):
            reset_analysis()


def reset_analysis():
    """Reset the analysis state."""
    st.session_state.data_loaded = False
    st.session_state.analysis_complete = False
    st.session_state.results = None
    st.session_state.selected_model = 'hyperbolic'
    st.rerun()


def download_results(export_format):
    """Handle result downloads."""
    if not st.session_state.analysis_complete:
        st.error("No analysis results to download")
        return

    results = st.session_state.results

    if export_format == 'Excel (XLSX)':
        # Generate reserves table
        params = results['fitting_results'][st.session_state.selected_model]['parameters']
        max_time = results['processed_data']['time'].max()
        reserves_table = generate_reserves_table(
            params,
            (0, max_time * 2),  # Project to 2x current time
            freq='M',
            q_abandon=st.session_state.get('q_abandon', 10.0),
            time_unit=st.session_state.get('time_unit', 'months')
        )

        # Export
        excel_data = export_complete_analysis_to_csv(
            results['fitting_results'],
            reserves_table,
            results['metrics'],
            st.session_state.selected_model
        )

        st.download_button(
            label="📥 Download Excel",
            data=excel_data,
            file_name="decline_curve_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        # Text report
        report = generate_summary_report(
            results['fitting_results'],
            results['metrics'],
            st.session_state.selected_model
        )

        st.download_button(
            label="📥 Download Report",
            data=report.encode('utf-8'),
            file_name="decline_curve_analysis_report.txt",
            mime="text/plain"
        )


def load_and_validate_data(uploaded_file):
    """Load and validate uploaded production data."""
    try:
        # Load data
        df = load_production_data(uploaded_file)

        # Validate
        is_valid, issues = validate_production_data(df)

        if not is_valid:
            st.error("Data validation failed:")
            for issue in issues:
                st.error(f"• {issue}")
            return None, None

        # Get available rates
        available_rates = get_available_rates(df)

        if not available_rates:
            st.error("No rate columns found in data")
            return None, None

        # Detect frequency
        frequency = detect_data_frequency(df)

        return df, available_rates, frequency

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None


def run_analysis(df, rate_column, time_unit, q_abandon):
    """Run the complete decline curve analysis with statistical analysis."""
    try:
        # Preprocess data
        processed_df = preprocess_data(df, time_unit)

        # Get time and rate arrays
        time = processed_df['time'].values
        rate = processed_df[rate_column].values

        # Filter out zeros for fitting
        valid_mask = rate > 0
        if np.sum(valid_mask) < 3:
            st.error("Insufficient valid data points for analysis")
            return None

        time_valid = time[valid_mask]
        rate_valid = rate[valid_mask]

        # Fit all models with statistical analysis
        fitting_results = {}
        for model_name in ARPS_MODELS.keys():
            fitting_results[model_name] = fit_arps_model_with_statistics(
                model_name, time_valid, rate_valid,
                q_abandon=q_abandon, time_unit=time_unit
            )

        # Add information criteria
        fitting_results = add_information_criteria(fitting_results, len(time_valid))

        # Get best model
        best_model, best_result = get_best_model(fitting_results)

        # Calculate metrics
        metrics = calculate_decline_curve_summary(
            best_result,
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

        return {
            'processed_data': processed_df,
            'fitting_results': fitting_results,
            'best_model': best_model,
            'metrics': metrics,
            'reserves_table': reserves_table,
            'time': time_valid,
            'rate': rate_valid
        }

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None


def display_results(results, rate_column, time_unit):
    """Display analysis results for single well."""
    fitting_results = results['fitting_results']
    best_model = results['best_model']
    metrics = results['metrics']
    processed_df = results['processed_data']

    # Update session state
    st.session_state.results = results
    st.session_state.selected_model = best_model
    st.session_state.analysis_complete = True

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Rate-Time Chart",
        "📊 Model Comparison",
        "📋 Reserves Table",
        "🔍 Diagnostics",
        "📊 Statistical Analysis"
    ])

    # Unit for display
    unit = "STB/D" if 'oil' in rate_column or 'water' in rate_column else "MSCF/D"

    with tab1:
        st.subheader("Production Rate vs Time")

# Linear scale chart
        fig_linear = create_rate_time_chart(
            processed_df,
            fitting_results,
            selected_model=st.session_state.selected_model,
            rate_column=rate_column,
            unit=unit
        )
        st.plotly_chart(fig_linear, use_container_width=True, key=f"rate-time-linear-{st.session_state.selected_model}")

        # Log scale chart
        st.subheader("Log Scale")
        fig_log = create_log_chart(
            processed_df,
            fitting_results,
            selected_model=st.session_state.selected_model,
            rate_column=rate_column,
            unit=unit
        )
        st.plotly_chart(fig_log, use_container_width=True, key=f"rate-time-log-{st.session_state.selected_model}")

        # Cumulative production chart
        st.subheader("Cumulative Production")
        fig_cum = create_cumulative_production_chart(
            processed_df,
            fitting_results[st.session_state.selected_model],
            rate_column=rate_column,
            unit="STB" if 'oil' in rate_column else "MSCF"
        )
        st.plotly_chart(fig_cum, use_container_width=True, key=f"cumulative-production-{st.session_state.selected_model}")

    with tab2:
        st.subheader("Model Comparison")

# Model comparison chart
        fig_comparison = create_model_comparison_chart(fitting_results)
        st.plotly_chart(fig_comparison, use_container_width=True, key=f"model-comparison-{st.session_state.selected_model}")

        # Model selection
        st.subheader("Select Best Model")
        model_options = list(fitting_results.keys())
        selected_model = st.selectbox(
            "Choose Model",
            options=model_options,
            format_func=lambda x: {
                'exponential': 'Exponential (b=0)',
                'hyperbolic': 'Hyperbolic (0<b<1)',
                'harmonic': 'Harmonic (b=1)'
            }.get(x, x),
            index=model_options.index(best_model) if best_model in model_options else 0
        )
        st.session_state.selected_model = selected_model

        # Display parameters for selected model
        selected_result = fitting_results[selected_model]
        if selected_result['success']:
            st.subheader(f"Parameters for {selected_model.title()} Model")

            col1, col2, col3 = st.columns(3)
            params = selected_result['parameters']

            with col1:
                st.metric("Qi (Initial Rate)", f"{params['Qi']:.2f}",
                         delta=f"±{selected_result['perr']['Qi']:.2f}")

            with col2:
                st.metric("Di (Decline Rate)", f"{params['Di']:.4f}",
                         delta=f"±{selected_result['perr']['Di']:.4f}")

            with col3:
                if 'b' in params:
                    st.metric("b (Decline Exponent)", f"{params['b']:.4f}",
                             delta=f"±{selected_result['perr']['b']:.4f}")
                else:
                    b_val = 0 if selected_model == 'exponential' else 1
                    st.metric("b (Decline Exponent)", f"{b_val:.1f}")

        # Key metrics display
        st.divider()
        st.subheader("Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("EUR", f"{metrics['eur']:,.0f}",
                     help="Estimated Ultimate Recovery")

        with col2:
            st.metric("R²", f"{metrics['r_squared']:.4f}",
                     help="Coefficient of Determination")

        with col3:
            st.metric("RMSE", f"{metrics['rmse']:.2f}",
                     help="Root Mean Squared Error")

        with col4:
            st.metric("Effective Decline", f"{metrics['effective_decline_rate']*100:.1f}%",
                     help="Annual Effective Decline Rate")

    with tab3:
        st.subheader("Reserves Table")

        reserves_table = results['reserves_table']

        # Display summary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total EUR", f"{metrics['eur']:,.0f} STB")

        with col2:
            st.metric("Time to Abandonment",
                     f"{metrics['time_to_abandonment']:.0f} months")

        with col3:
            st.metric("Data Points", len(processed_df))

        # Display table
        st.dataframe(
            reserves_table[['time', 'rate', 'cumulative_production', 'remaining_reserves']].round(2),
            use_container_width=True,
            height=400
        )

    with tab4:
        st.subheader("Residuals Analysis")

        fig_residuals = create_residuals_chart(fitting_results)
        st.plotly_chart(fig_residuals, use_container_width=True)

# Display detailed statistics
        st.subheader("Detailed Statistics")

        stats_data = []
        for model_name, result in fitting_results.items():
            if result['success']:
                stats_data.append({
                    'Model': model_name,
                    'R²': result['r_squared'],
                    'RMSE': result['rmse'],
                    'AIC': result.get('aic', 0),
                    'BIC': result.get('bic', 0)
                })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df.round(4), use_container_width=True)

    with tab4:
        st.subheader("Residuals Analysis")

        fig_residuals = create_residuals_chart(fitting_results)
        st.plotly_chart(fig_residuals, use_container_width=True, key=f"residuals-{st.session_state.selected_model}")

    with tab5:
        st.subheader("Statistical Analysis")

        # Display statistical summary for best model
        if 'statistical_summary' in results:
            stats = results['statistical_summary'][best_model]

            # Monte Carlo simulation results
            st.subheader("🎲 Monte Carlo Simulation")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Mean EUR", f"{stats['mc_mean']:,.0f}")
                st.metric("Std Dev", f"{stats['mc_std']:,.0f}")

            with col2:
                st.metric("95% CI Lower", f"{stats['mc_ci_lower']:,.0f}")
                st.metric("95% CI Upper", f"{stats['mc_ci_upper']:,.0f}")

            with col3:
                st.metric("Samples", f"{stats['mc_samples']}")

            # Confidence intervals
            st.subheader("📏 Confidence Intervals")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Standard Error", f"{stats['ci_std']:,.0f}")
                st.metric("95% CI Lower", f"{stats['ci_lower']:,.0f}")

            with col2:
                st.metric("95% CI Upper", f"{stats['ci_upper']:,.0f}")
                st.metric("Confidence Level", f"{95}%")

            # Sensitivity analysis
            st.subheader("🔍 Sensitivity Analysis")
            st.write("**Parameter Influence on EUR:**")
            st.write("**First-Order Indices (S1):** Higher values indicate more influence")
            st.write(f"Qi (Initial Rate): {stats['sensitivity_S1'][0]:.3f}")
            st.write(f"Di (Decline Rate): {stats['sensitivity_S1'][1]:.3f}")
            st.write(f"b (Decline Exponent): {stats['sensitivity_S1'][2]:.3f}")

            st.write("**Total-Order Indices (ST):** Account for interactions")
            st.write(f"Qi: {stats['sensitivity_ST'][0]:.3f}")
            st.write(f"Di: {stats['sensitivity_ST'][1]:.3f}")
            st.write(f"b: {stats['sensitivity_ST'][2]:.3f}")

            # Summary statistics table
            st.subheader("📊 Summary Statistics")
            summary_data = [
                {'Metric': 'EUR', 'Value': f"{metrics['eur']:,.0f}"},
                {'Metric': '95% CI Lower', 'Value': f"{stats['ci_lower']:,.0f}"},
                {'Metric': '95% CI Upper', 'Value': f"{stats['ci_upper']:,.0f}"},
                {'Metric': 'Monte Carlo Mean', 'Value': f"{stats['mc_mean']:,.0f}"},
                {'Metric': 'Monte Carlo Std Dev', 'Value': f"{stats['mc_std']:,.0f}"},
                {'Metric': 'Time to Abandonment', 'Value': f"{metrics['time_to_abandonment']:.0f} months"},
                {'Metric': 'Effective Decline', 'Value': f"{metrics['effective_decline_rate']*100:.1f}%"}
            ]
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)


def display_data_preview(df, available_rates, frequency):
    """Display a preview of the loaded data."""
    st.subheader("📄 Data Preview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Records", len(df))

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        st.metric("Frequency", frequency.capitalize())

    # Show available rate columns
    st.write(f"**Available Rate Columns:** {', '.join(available_rates)}")

    # Show first few rows
    st.dataframe(df.head(10), use_container_width=True)


def main_content():
    """Main content area."""
    uploaded_file = st.session_state.get('uploaded_file', None)

    if uploaded_file is None:
        # Show welcome message
        st.markdown(
            """
            <div style='text-align: center; padding: 3rem;'>
                <h3>Welcome to the Decline Curve Analyzer</h3>
                <p>Upload your production data using the sidebar to get started.</p>
                <p>Your data should include:</p>
                <ul style='list-style: none; padding: 0;'>
                    <li>📅 Date column</li>
                    <li>🛢️ Oil rate (STB/D)</li>
                    <li>⛽ Gas rate (MSCF/D) - optional</li>
                    <li>💧 Water rate (STB/D) - optional</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Load data
    df, available_rates, frequency = load_and_validate_data(uploaded_file)

    if df is None:
        return

    # Display data preview
    display_data_preview(df, available_rates, frequency)

    # Get configuration from sidebar
    time_unit = st.session_state.get('time_unit', 'months')
    q_abandon = st.session_state.get('q_abandon', 10.0)
    fluid_type = st.session_state.get('fluid_type', 'oil_rate')

# Run analysis button
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Running decline curve analysis..."):
            results = run_analysis(df, fluid_type, time_unit, q_abandon)

            if results:
                st.success("Analysis completed successfully!")
                display_results(results, fluid_type, time_unit)
                st.session_state.analysis_complete = True
                st.session_state.results = results
            else:
                st.error("Analysis failed. Please check your data.")
                st.session_state.analysis_complete = False
                st.session_state.results = None

    # Show previous results if available
    if st.session_state.analysis_complete and st.session_state.results:
        # Only show results if they haven't been displayed yet
        if not hasattr(st.session_state, 'results_displayed'):
            display_results(st.session_state.results, fluid_type, time_unit)
            st.session_state.results_displayed = True


if __name__ == "__main__":
    main()
    main_content()
