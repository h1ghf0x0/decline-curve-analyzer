"""
Reserves Calculations - EUR and reserves table generation with statistical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import scipy.stats as stats
from .models import (
    exponential_decline, hyperbolic_decline, harmonic_decline,
    calculate_time_to_abandonment
)

# Try to import SALib, but handle case where it's not available
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False


def calculate_eur(qi: float, di: float, b: float,
                   q_abandon: float = 10.0, time_unit: str = 'months') -> float:
    """
    Calculate Estimated Ultimate Recovery (EUR).

    EUR is calculated by integrating the decline curve from t=0 to the time
    when the rate reaches the abandonment threshold.

    Args:
        qi: Initial production rate
        di: Initial decline rate (in 1/time_unit)
        b: Decline exponent (0 for exponential, 0<b<1 for hyperbolic, 1 for harmonic)
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit ('days', 'months', 'years')

    Returns:
        EUR in the same units as the rate multiplied by time
        (e.g., if rate is STB/day and time_unit is 'years', EUR is in STB)
    """
    # Calculate time to abandonment
    t_abandon = calculate_time_to_abandonment(qi, di, b, q_abandon)

    # Integration using analytical formulas
    if b == 0:
        # Exponential decline: Np = (qi - q_abandon) / Di
        eur = (qi - q_abandon) / di
    elif b == 1:
        # Harmonic decline: Np = (qi / Di) * ln(1 + Di * t_abandon)
        eur = (qi / di) * np.log(1.0 + di * t_abandon)
    else:
        # Hyperbolic decline: Np = (qi^b / (Di * (1-b))) * (qi^(1-b) - q_abandon^(1-b))
        # Alternative form: Np = (qi / (Di * (1-b))) * (1 - (q_abandon/qi)^(1-b))
        eur = (qi / (di * (1.0 - b))) * (1.0 - (q_abandon / qi) ** (1.0 - b))

    # Convert to appropriate units based on time_unit
    # Assuming qi is in rate per day, convert to annual or monthly basis
    if time_unit == 'days':
        # EUR is already in correct units (rate * days)
        pass
    elif time_unit == 'months':
        # Convert from daily rate to monthly
        eur *= 30.4375
    elif time_unit == 'years':
        # Convert from daily rate to yearly
        eur *= 365.25

    return max(eur, 0)


def monte_carlo_eur_simulation(params: Dict, q_abandon: float = 10.0,
                               time_unit: str = 'months', n_samples: int = 1000) -> Dict:
    """
    Perform Monte Carlo simulation to estimate EUR uncertainty.

    Args:
        params: Dictionary with fitted parameters and their uncertainties
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations
        n_samples: Number of Monte Carlo samples

    Returns:
        Dictionary with simulation results including mean, std, confidence intervals
    """
    # Extract parameters and uncertainties
    qi = params['parameters']['Qi']
    di = params['parameters']['Di']
    b = params.get('b', 0)
    qi_err = params['perr']['Qi']
    di_err = params['perr']['Di']
    b_err = params.get('perr', {}).get('b', 0)

    # Generate random samples from parameter distributions
    # Assuming normal distributions for parameters
    qi_samples = np.random.normal(qi, qi_err, n_samples)
    di_samples = np.random.normal(di, di_err, n_samples)
    if b_err > 0:
        b_samples = np.random.normal(b, b_err, n_samples)
    else:
        b_samples = np.full(n_samples, b)

    # Calculate EUR for each sample
    eur_samples = []
    for i in range(n_samples):
        try:
            eur = calculate_eur(qi_samples[i], di_samples[i], b_samples[i],
                               q_abandon, time_unit)
            eur_samples.append(eur)
        except:
            continue

    # Calculate statistics
    eur_samples = np.array(eur_samples)
    mean_eur = np.mean(eur_samples)
    std_eur = np.std(eur_samples)
    ci_95 = np.percentile(eur_samples, [2.5, 97.5])

    return {
        'mean': mean_eur,
        'std': std_eur,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'samples': eur_samples,
        'n_samples': len(eur_samples)
    }


def calculate_confidence_intervals(params: Dict, q_abandon: float = 10.0,
                                   time_unit: str = 'months', confidence_level: float = 0.95) -> Dict:
    """
    Calculate confidence intervals for EUR using parameter uncertainties.

    Args:
        params: Dictionary with fitted parameters and their uncertainties
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Dictionary with confidence interval bounds
    """
    # Extract parameters and uncertainties
    qi = params['parameters']['Qi']
    di = params['parameters']['Di']
    b = params.get('b', 0)
    qi_err = params['perr']['Qi']
    di_err = params['perr']['Di']
    b_err = params.get('perr', {}).get('b', 0)

    # Calculate partial derivatives for error propagation
    # EUR sensitivity to Qi
    if b == 0:
        d_eur_dqi = 1.0 / di
    elif b == 1:
        t_abandon = calculate_time_to_abandonment(qi, di, b, q_abandon)
        d_eur_dqi = (1.0 / di) * np.log(1.0 + di * t_abandon)
    else:
        d_eur_dqi = 1.0 / (di * (1.0 - b)) * (1.0 - (q_abandon / qi) ** (1.0 - b))

    # EUR sensitivity to Di
    if b == 0:
        d_eur_ddi = -(qi - q_abandon) / (di ** 2)
    elif b == 1:
        t_abandon = calculate_time_to_abandonment(qi, di, b, q_abandon)
        d_eur_ddi = -(qi / (di ** 2)) * np.log(1.0 + di * t_abandon)
    else:
        d_eur_ddi = -(qi / (di ** 2 * (1.0 - b))) * (1.0 - (q_abandon / qi) ** (1.0 - b))

    # EUR sensitivity to b (only for hyperbolic model)
    d_eur_db = 0.0
    if b > 0 and b < 1:
        # Complex derivative, using finite difference approximation
        b_plus = min(b + 0.01, 0.99)
        b_minus = max(b - 0.01, 0.01)
        eur_plus = calculate_eur(qi, di, b_plus, q_abandon, time_unit)
        eur_minus = calculate_eur(qi, di, b_minus, q_abandon, time_unit)
        d_eur_db = (eur_plus - eur_minus) / 0.02

    # Calculate variance using error propagation
    var_eur = (d_eur_dqi ** 2 * qi_err ** 2 +
               d_eur_ddi ** 2 * di_err ** 2 +
               d_eur_db ** 2 * b_err ** 2)

    std_eur = np.sqrt(var_eur)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = calculate_eur(qi, di, b, q_abandon, time_unit) - z_score * std_eur
    ci_upper = calculate_eur(qi, di, b, q_abandon, time_unit) + z_score * std_eur

    return {
        'std': std_eur,
        'ci_lower': max(ci_lower, 0),
        'ci_upper': ci_upper,
        'confidence_level': confidence_level
    }


def sensitivity_analysis(params: Dict, q_abandon: float = 10.0,
                        time_unit: str = 'months', n_samples: int = 1000) -> Dict:
    """
    Perform sensitivity analysis to determine parameter influence on EUR.

    Args:
        params: Dictionary with fitted parameters
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations
        n_samples: Number of samples for analysis

    Returns:
        Dictionary with sensitivity indices for each parameter
    """
    # Define problem for SALib
    problem = {
        'num_vars': 3,
        'names': ['Qi', 'Di', 'b'],
        'bounds': [[params['parameters']['Qi'] * 0.5, params['parameters']['Qi'] * 1.5],
                   [params['parameters']['Di'] * 0.5, params['parameters']['Di'] * 1.5],
                   [0.0, 1.0]]
    }

    # Generate samples using Saltelli's method
    param_values = saltelli.sample(problem, n_samples)

    # Calculate EUR for each sample
    eur_values = np.zeros(param_values.shape[0])
    for i, params_sample in enumerate(param_values):
        qi_sample = params_sample[0]
        di_sample = params_sample[1]
        b_sample = params_sample[2]
        try:
            eur_values[i] = calculate_eur(qi_sample, di_sample, b_sample,
                                         q_abandon, time_unit)
        except:
            eur_values[i] = np.nan

    # Remove invalid values
    valid_mask = ~np.isnan(eur_values)
    param_values = param_values[valid_mask]
    eur_values = eur_values[valid_mask]

    # Perform Sobol sensitivity analysis
    Si = sobol.analyze(problem, eur_values, print_to_console=False)

    return {
        'S1': Si['S1'],  # First-order indices
        'S2': Si['S2'],  # Second-order indices
        'ST': Si['ST'],  # Total-order indices
        'problem': problem,
        'n_valid_samples': len(eur_values)
    }


def calculate_statistical_summary(fitting_results: Dict, q_abandon: float = 10.0,
                                   time_unit: str = 'months') -> Dict:
    """
    Calculate comprehensive statistical summary for all fitted models.

    Args:
        fitting_results: Dictionary of fitting results from fit_all_models
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations

    Returns:
        Dictionary with statistical analysis for each model
    """
    summary = {}

    for model_name, result in fitting_results.items():
        if not result['success']:
            continue

        # Calculate basic EUR
        params = result['parameters']
        b = params.get('b', 0)
        eur = calculate_eur(params['Qi'], params['Di'], b,
                           q_abandon, time_unit)

        # Calculate confidence intervals
        ci = calculate_confidence_intervals(result, q_abandon, time_unit)

        # Perform Monte Carlo simulation
        mc_results = monte_carlo_eur_simulation(result, q_abandon, time_unit, 1000)

        # Perform sensitivity analysis
        sens_results = sensitivity_analysis(result, q_abandon, time_unit, 1000)

        summary[model_name] = {
            'eur': eur,
            'ci_lower': ci['ci_lower'],
            'ci_upper': ci['ci_upper'],
            'ci_std': ci['std'],
            'mc_mean': mc_results['mean'],
            'mc_std': mc_results['std'],
            'mc_ci_lower': mc_results['ci_95_lower'],
            'mc_ci_upper': mc_results['ci_95_upper'],
            'mc_samples': mc_results['n_samples'],
            'sensitivity_S1': sens_results['S1'].tolist(),
            'sensitivity_S2': sens_results['S2'].tolist(),
            'sensitivity_ST': sens_results['ST'].tolist(),
            'sensitivity_valid_samples': sens_results['n_valid_samples']
        }

    return summary


def calculate_statistical_decline_summary(fitting_result: Dict,
                                           q_abandon: float = 10.0,
                                           time_unit: str = 'months') -> Dict:
    """
    Generate a comprehensive statistical summary of decline curve analysis results.

    Args:
        fitting_result: Result dictionary from fit_arps_model
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations

    Returns:
        Dictionary with complete statistical analysis summary
    """
    # Get basic metrics
    metrics = calculate_decline_metrics(fitting_result['parameters'], q_abandon, time_unit)

    # Calculate confidence intervals
    ci = calculate_confidence_intervals(fitting_result, q_abandon, time_unit)

    # Perform Monte Carlo simulation
    mc_results = monte_carlo_eur_simulation(fitting_result, q_abandon, time_unit, 1000)

    # Perform sensitivity analysis
    sens_results = sensitivity_analysis(fitting_result, q_abandon, time_unit, 1000)

    # Add statistical metrics
    metrics['ci_lower'] = ci['ci_lower']
    metrics['ci_upper'] = ci['ci_upper']
    metrics['ci_std'] = ci['std']
    metrics['mc_mean'] = mc_results['mean']
    metrics['mc_std'] = mc_results['std']
    metrics['mc_ci_lower'] = mc_results['ci_95_lower']
    metrics['mc_ci_upper'] = mc_results['ci_95_upper']
    metrics['mc_samples'] = mc_results['n_samples']
    metrics['sensitivity_S1'] = sens_results['S1'].tolist()
    metrics['sensitivity_S2'] = sens_results['S2'].tolist()
    metrics['sensitivity_ST'] = sens_results['ST'].tolist()
    metrics['sensitivity_valid_samples'] = sens_results['n_valid_samples']

    return metrics


def calculate_decline_curve_summary(fitting_result: Dict,
                                     q_abandon: float = 10.0,
                                     time_unit: str = 'months') -> Dict:
    """
    Generate a comprehensive summary of decline curve analysis results.

    Args:
        fitting_result: Result dictionary from fit_arps_model
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations

    Returns:
        Dictionary with complete analysis summary
    """
    params = fitting_result.get('parameters', {})

    # Get metrics
    metrics = calculate_decline_metrics(params, q_abandon, time_unit)

    # Add fitting quality metrics
    metrics['r_squared'] = fitting_result.get('r_squared', np.nan)
    metrics['rmse'] = fitting_result.get('rmse', np.nan)
    metrics['model_name'] = fitting_result.get('model_name', 'unknown')

    return metrics


def calculate_cumulative_production(qi: float, di: float, b: float,
                                     t: np.ndarray) -> np.ndarray:
    """
    Calculate cumulative production at each time point.

    Args:
        qi: Initial production rate
        di: Initial decline rate
        b: Decline exponent
        t: Time array

    Returns:
        Cumulative production at each time point
    """
    t = np.asarray(t, dtype=float)
    cum_prod = np.zeros_like(t)

    if b == 0:
        # Exponential: Np = (qi / Di) * (1 - exp(-Di * t))
        cum_prod = (qi / di) * (1.0 - np.exp(-di * t))
    elif b == 1:
        # Harmonic: Np = (qi / Di) * ln(1 + Di * t)
        cum_prod = (qi / di) * np.log(1.0 + di * t)
    else:
        # Hyperbolic: Np = (qi^b / (Di * (1-b))) * (qi^(1-b) - q^(1-b))
        # where q = qi / (1 + b * Di * t)^(1/b)
        q_t = hyperbolic_decline(t, qi, di, b)
        cum_prod = (qi ** b / (di * (1.0 - b))) * (qi ** (1.0 - b) - q_t ** (1.0 - b))

    return cum_prod


def calculate_remaining_reserves(qi: float, di: float, b: float,
                                  q_abandon: float, t: np.ndarray) -> np.ndarray:
    """
    Calculate remaining reserves at each time point.

    Args:
        qi: Initial production rate
        di: Initial decline rate
        b: Decline exponent
        q_abandon: Economic abandonment rate threshold
        t: Time array

    Returns:
        Remaining reserves at each time point
    """
    # Total EUR
    eur = calculate_eur(qi, di, b, q_abandon)

    # Cumulative production at each time
    cum_prod = calculate_cumulative_production(qi, di, b, t)

    # Remaining reserves
    remaining = eur - cum_prod

    return np.maximum(remaining, 0)


def generate_reserves_table(params: Dict, time_range: tuple,
                            freq: str = 'M', q_abandon: float = 10.0,
                            time_unit: str = 'months') -> pd.DataFrame:
    """
    Generate a reserves table with monthly/annual projections.

    Args:
        params: Dictionary with fitted parameters (Qi, Di, b if applicable)
        time_range: Tuple of (start_time, end_time) in time units
        freq: Frequency of table ('M' for monthly, 'Y' for yearly, 'D' for daily)
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations

    Returns:
        DataFrame with columns: time, rate, cumulative_production, remaining_reserves
    """
    qi = params.get('Qi', 0)
    di = params.get('Di', 0)
    b = params.get('b', 0)

    start_time, end_time = time_range

    # Create time array based on frequency
    if freq == 'D':
        # Daily
        time_array = np.arange(start_time, end_time, 1)
    elif freq == 'M':
        # Monthly (approximately 30 days)
        time_array = np.arange(start_time, end_time, 1)
    elif freq == 'Y':
        # Yearly
        time_array = np.arange(start_time, end_time, 1)
    else:
        time_array = np.arange(start_time, end_time, 1)

    # Calculate rate at each time point
    if b == 0:
        rate = exponential_decline(time_array, qi, di)
    elif b == 1:
        rate = harmonic_decline(time_array, qi, di)
    else:
        rate = hyperbolic_decline(time_array, qi, di, b)

    # Calculate cumulative production
    cum_prod = calculate_cumulative_production(qi, di, b, time_array)

    # Calculate remaining reserves
    remaining = calculate_remaining_reserves(qi, di, b, q_abandon, time_array)

    # Create DataFrame
    df = pd.DataFrame({
        'time': time_array,
        'rate': rate,
        'cumulative_production': cum_prod,
        'remaining_reserves': remaining
    })

    # Add period label based on frequency
    if freq == 'M':
        df['period'] = df['time'].apply(lambda x: f"Month {int(x)}")
    elif freq == 'Y':
        df['period'] = df['time'].apply(lambda x: f"Year {int(x)}")
    else:
        df['period'] = df['time'].apply(lambda x: f"Day {int(x)}")

    return df


def calculate_decline_metrics(params: Dict, q_abandon: float = 10.0,
                               time_unit: str = 'months') -> Dict:
    """
    Calculate comprehensive decline metrics from fitted parameters.

    Args:
        params: Dictionary with fitted parameters (Qi, Di, b if applicable)
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations

    Returns:
        Dictionary with all calculated metrics
    """
    qi = params.get('Qi', 0)
    di = params.get('Di', 0)
    b = params.get('b', 0)

    # Calculate time to abandonment
    t_abandon = calculate_time_to_abandonment(qi, di, b, q_abandon)

    # Calculate EUR
    eur = calculate_eur(qi, di, b, q_abandon, time_unit)

    # Calculate effective decline rate (annual)
    if time_unit == 'months':
        di_annual = di * 12  # Convert monthly Di to annual
    elif time_unit == 'days':
        di_annual = di * 365.25
    else:
        di_annual = di

    effective_decline = 1.0 - (1.0 + b * di_annual) ** (-1.0 / b) if b > 0 else 1.0 - np.exp(-di_annual)

    # Calculate initial production potential
    # (first month/year production based on time unit)
    if time_unit == 'months':
        initial_period_prod = qi * 1.0  # First month at initial rate
    elif time_unit == 'years':
        initial_period_prod = qi * 365.25
    else:
        initial_period_prod = qi

    return {
        'Qi': qi,
        'Di': di,
        'Di_annual': di_annual,
        'b': b,
        'q_abandon': q_abandon,
        'eur': eur,
        'time_to_abandonment': t_abandon,
        'effective_decline_rate': effective_decline,
        'initial_period_production': initial_period_prod,
        'time_unit': time_unit
    }


def calculate_recovery_factor(eur: float, oip: float) -> float:
    """
    Calculate recovery factor given EUR and original oil/gas in place.

    Args:
        eur: Estimated Ultimate Recovery
        oip: Original Oil/Gas In Place

    Returns:
        Recovery factor as a decimal (0-1)
    """
    if oip <= 0:
        return 0.0
    return min(eur / oip, 1.0)