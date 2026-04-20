"""
Curve Fitting Engine - Arps DCA model fitting using scipy.optimize.curve_fit.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from typing import Dict, Optional, Tuple, Any

from .models import (
    ARPS_MODELS, exponential_decline, hyperbolic_decline, 
    harmonic_decline, MODEL_PARAM_COUNT
)


def calculate_initial_guesses(time: np.ndarray, rate: np.ndarray) -> Dict[str, Any]:
    """
    Generate initial parameter guesses for curve fitting.
    
    Args:
        time: Time array
        rate: Production rate array
        
    Returns:
        Dictionary with initial guesses for Qi, Di, and b
    """
    time = np.asarray(time, dtype=float)
    rate = np.asarray(rate, dtype=float)
    
    # Filter out zero/negative rates
    valid_mask = rate > 0
    if not np.any(valid_mask):
        return {'Qi': np.max(rate), 'Di': 0.1, 'b': 0.5}
    
    valid_time = time[valid_mask]
    valid_rate = rate[valid_mask]
    
    # Qi: Use the first rate value or maximum
    Qi_init = valid_rate[0]
    
    # Di: Estimate from the decline over the time period
    if len(valid_rate) > 1 and valid_rate[-1] > 0:
        # Average decline rate
        if valid_rate[0] > valid_rate[-1]:
            Di_init = np.log(valid_rate[0] / valid_rate[-1]) / valid_time[-1]
        else:
            Di_init = 0.01  # Small default value
    else:
        Di_init = 0.1
    
    # Clamp Di to reasonable range
    Di_init = np.clip(Di_init, 0.001, 1.0)
    
    # b: Estimate based on the shape of decline
    # Use log-rate vs log-time slope to estimate b
    b_init = 0.5  # Default to hyperbolic
    
    if len(valid_rate) > 3:
        try:
            # Calculate log-log slope for b estimation
            log_rate = np.log(valid_rate)
            log_time = np.log(valid_time + 0.01)  # Avoid log(0)
            
            # Simple linear regression on log-log
            if np.std(log_time) > 0:
                slope = np.polyfit(log_time, log_rate, 1)[0]
                # b is related to the slope (typically -1/b for hyperbolic)
                if slope < -0.1:
                    b_init = np.clip(-1.0 / slope, 0.1, 0.9)
        except:
            pass
    
    return {'Qi': Qi_init, 'Di': Di_init, 'b': b_init}


def fit_arps_model(model_name: str, time: np.ndarray, rate: np.ndarray,
                    initial_guess: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Fit an Arps DCA model to production data.

    Args:
        model_name: Name of the model ('exponential', 'hyperbolic', 'harmonic')
        time: Time array (must be in consistent units)
        rate: Production rate array
        initial_guess: Optional dictionary with initial parameter guesses

    Returns:
        Dictionary containing:
            - model_name: Name of the fitted model
            - parameters: Fitted parameters (Qi, Di, b if applicable)
            - perr: Parameter estimation errors
            - r_squared: Coefficient of determination
            - rmse: Root mean squared error
            - residuals: Residuals array
            - fitted_rate: Fitted rate values at data points
            - success: Whether the fit converged
            - message: Fit status message

    Raises:
        ValueError: If model name is not recognized
        RuntimeError: If fitting fails
    """
    time = np.asarray(time, dtype=float)
    rate = np.asarray(rate, dtype=float)

    if model_name not in ARPS_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_func = ARPS_MODELS[model_name]

    # Get initial guesses
    if initial_guess is None:
        initial_guess = calculate_initial_guesses(time, rate)

    # Set up bounds based on model type
    param_count = MODEL_PARAM_COUNT[model_name]

    # Define parameter bounds (lower, upper)
    if model_name == 'hyperbolic':
        p0 = [initial_guess['Qi'], initial_guess['Di'], initial_guess['b']]
        lower_bounds = [0.0, 0.0, 0.0]
        upper_bounds = [np.inf, 1.0, 1.0]
    else:  # exponential or harmonic
        p0 = [initial_guess['Qi'], initial_guess['Di']]
        lower_bounds = [0.0, 0.0]
        upper_bounds = [np.inf, 1.0]

    # Set max iterations
    maxfev = 10000

    try:
        # Perform curve fitting
        popt, pcov = curve_fit(
            model_func, time, rate,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=maxfev,
            method='trf'  # Trust Region Reflective
        )

        # Calculate fitted values
        fitted_rate = model_func(time, *popt)

        # Calculate residuals
        residuals = rate - fitted_rate

        # Calculate R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((rate - np.mean(rate)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Calculate RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))

        # Calculate parameter errors
        try:
            perr = np.sqrt(np.diag(pcov))
        except:
            perr = np.full(len(popt), np.nan)

        # Build results dictionary
        result = {
            'model_name': model_name,
            'parameters': {
                'Qi': popt[0],
                'Di': popt[1]
            },
            'perr': {
                'Qi': perr[0] if len(perr) > 0 else np.nan,
                'Di': perr[1] if len(perr) > 1 else np.nan
            },
            'r_squared': r_squared,
            'rmse': rmse,
            'residuals': residuals,
            'fitted_rate': fitted_rate,
            'success': True,
            'message': 'Fit converged successfully'
        }

        # Add b parameter for hyperbolic model
        if model_name == 'hyperbolic':
            result['parameters']['b'] = popt[2]
            result['perr']['b'] = perr[2] if len(perr) > 2 else np.nan

        return result

    except Exception as e:
        return {
            'model_name': model_name,
            'parameters': initial_guess,
            'perr': {'Qi': np.nan, 'Di': np.nan},
            'r_squared': 0.0,
            'rmse': np.nan,
            'residuals': np.zeros_like(rate),
            'fitted_rate': np.zeros_like(rate),
            'success': False,
            'message': f'Fit failed: {str(e)}'
        }

def fit_arps_model_with_statistics(model_name: str, time: np.ndarray, rate: np.ndarray,
                                    initial_guess: Optional[Dict] = None,
                                    q_abandon: float = 10.0,
                                    time_unit: str = 'months') -> Dict[str, Any]:
    """
    Fit an Arps DCA model to production data and calculate statistical metrics.

    Args:
        model_name: Name of the model ('exponential', 'hyperbolic', 'harmonic')
        time: Time array (must be in consistent units)
        rate: Production rate array
        initial_guess: Optional dictionary with initial parameter guesses
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations

    Returns:
        Dictionary containing:
            - model_name: Name of the fitted model
            - parameters: Fitted parameters (Qi, Di, b if applicable)
            - perr: Parameter estimation errors
            - r_squared: Coefficient of determination
            - rmse: Root mean squared error
            - residuals: Residuals array
            - fitted_rate: Fitted rate values at data points
            - success: Whether the fit converged
            - message: Fit status message
            - statistical_summary: Comprehensive statistical analysis
    """
    # First fit the model
    result = fit_arps_model(model_name, time, rate, initial_guess)

    if not result['success']:
        return result

    # Calculate statistical summary
    from .calculations import calculate_statistical_decline_summary
    statistical_summary = calculate_statistical_decline_summary(
        result, q_abandon, time_unit
    )

    # Add statistical summary to results
    result['statistical_summary'] = statistical_summary

    return result
    """
    Fit an Arps DCA model to production data and calculate statistical metrics.

    Args:
        model_name: Name of the model ('exponential', 'hyperbolic', 'harmonic')
        time: Time array (must be in consistent units)
        rate: Production rate array
        initial_guess: Optional dictionary with initial parameter guesses
        q_abandon: Economic abandonment rate threshold
        time_unit: Time unit for calculations

    Returns:
        Dictionary containing:
            - model_name: Name of the fitted model
            - parameters: Fitted parameters (Qi, Di, b if applicable)
            - perr: Parameter estimation errors
            - r_squared: Coefficient of determination
            - rmse: Root mean squared error
            - residuals: Residuals array
            - fitted_rate: Fitted rate values at data points
            - success: Whether the fit converged
            - message: Fit status message
            - statistical_summary: Comprehensive statistical analysis
    """
    # First fit the model
    result = fit_arps_model(model_name, time, rate, initial_guess)

    if not result['success']:
        return result

    # Calculate statistical summary
    from .calculations import calculate_statistical_decline_summary
    statistical_summary = calculate_statistical_decline_summary(
        result, q_abandon, time_unit
    )

    # Add statistical summary to results
    result['statistical_summary'] = statistical_summary

    return result
    """
    Fit an Arps DCA model to production data.
    
    Args:
        model_name: Name of the model ('exponential', 'hyperbolic', 'harmonic')
        time: Time array (must be in consistent units)
        rate: Production rate array
        initial_guess: Optional dictionary with initial parameter guesses
        
    Returns:
        Dictionary containing:
            - model_name: Name of the fitted model
            - parameters: Fitted parameters (Qi, Di, b if applicable)
            - perr: Parameter estimation errors
            - r_squared: Coefficient of determination
            - rmse: Root mean squared error
            - residuals: Residuals array
            - fitted_rate: Fitted rate values at data points
            - success: Whether the fit converged
            - message: Fit status message
            
    Raises:
        ValueError: If model name is not recognized
        RuntimeError: If fitting fails
    """
    time = np.asarray(time, dtype=float)
    rate = np.asarray(rate, dtype=float)
    
    if model_name not in ARPS_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_func = ARPS_MODELS[model_name]
    
    # Get initial guesses
    if initial_guess is None:
        initial_guess = calculate_initial_guesses(time, rate)
    
    # Set up bounds based on model type
    param_count = MODEL_PARAM_COUNT[model_name]
    
    # Define parameter bounds (lower, upper)
    if model_name == 'hyperbolic':
        p0 = [initial_guess['Qi'], initial_guess['Di'], initial_guess['b']]
        lower_bounds = [0.0, 0.0, 0.0]
        upper_bounds = [np.inf, 1.0, 1.0]
    else:  # exponential or harmonic
        p0 = [initial_guess['Qi'], initial_guess['Di']]
        lower_bounds = [0.0, 0.0]
        upper_bounds = [np.inf, 1.0]
    
    # Set max iterations
    maxfev = 10000
    
    try:
        # Perform curve fitting
        popt, pcov = curve_fit(
            model_func, time, rate,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=maxfev,
            method='trf'  # Trust Region Reflective
        )
        
        # Calculate fitted values
        fitted_rate = model_func(time, *popt)
        
        # Calculate residuals
        residuals = rate - fitted_rate
        
        # Calculate R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((rate - np.mean(rate)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # Calculate parameter errors
        try:
            perr = np.sqrt(np.diag(pcov))
        except:
            perr = np.full(len(popt), np.nan)
        
        # Build results dictionary
        result = {
            'model_name': model_name,
            'parameters': {
                'Qi': popt[0],
                'Di': popt[1]
            },
            'perr': {
                'Qi': perr[0] if len(perr) > 0 else np.nan,
                'Di': perr[1] if len(perr) > 1 else np.nan
            },
            'r_squared': r_squared,
            'rmse': rmse,
            'residuals': residuals,
            'fitted_rate': fitted_rate,
            'success': True,
            'message': 'Fit converged successfully'
        }
        
        # Add b parameter for hyperbolic model
        if model_name == 'hyperbolic':
            result['parameters']['b'] = popt[2]
            result['perr']['b'] = perr[2] if len(perr) > 2 else np.nan
        
        return result
        
    except Exception as e:
        return {
            'model_name': model_name,
            'parameters': initial_guess,
            'perr': {'Qi': np.nan, 'Di': np.nan},
            'r_squared': 0.0,
            'rmse': np.nan,
            'residuals': np.zeros_like(rate),
            'fitted_rate': np.zeros_like(rate),
            'success': False,
            'message': f'Fit failed: {str(e)}'
        }


def fit_all_models(time: np.ndarray, rate: np.ndarray,
                   initial_guess: Optional[Dict] = None) -> Dict[str, Dict]:
    """
    Fit all three Arps models and return results.
    
    Args:
        time: Time array
        rate: Production rate array
        initial_guess: Optional initial parameter guesses
        
    Returns:
        Dictionary with results for each model
    """
    results = {}
    
    for model_name in ARPS_MODELS.keys():
        results[model_name] = fit_arps_model(model_name, time, rate, initial_guess)
    
    return results


def get_best_model(results: Dict[str, Dict]) -> Tuple[str, Dict]:
    """
    Determine the best fitting model based on R-squared.
    
    Args:
        results: Dictionary of fitting results from fit_all_models
        
    Returns:
        Tuple of (best_model_name, best_result)
    """
    best_model = None
    best_r2 = -np.inf
    best_result = None
    
    for model_name, result in results.items():
        if result['success'] and result['r_squared'] > best_r2:
            best_r2 = result['r_squared']
            best_model = model_name
            best_result = result
    
    return best_model, best_result


def calculate_aic(n: int, k: int, rmse: float) -> float:
    """
    Calculate Akaike Information Criterion (AIC).
    
    AIC = n * ln(RSS/n) + 2k
    
    Args:
        n: Number of data points
        k: Number of parameters
        rmse: Root mean squared error
        
    Returns:
        AIC value
    """
    rss = (rmse ** 2) * n
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + 2 * k


def calculate_bic(n: int, k: int, rmse: float) -> float:
    """
    Calculate Bayesian Information Criterion (BIC).
    
    BIC = n * ln(RSS/n) + k * ln(n)
    
    Args:
        n: Number of data points
        k: Number of parameters
        rmse: Root mean squared error
        
    Returns:
        BIC value
    """
    rss = (rmse ** 2) * n
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + k * np.log(n)


def add_information_criteria(results: Dict[str, Dict], n: int) -> Dict[str, Dict]:
    """
    Add AIC and BIC to fitting results.
    
    Args:
        results: Dictionary of fitting results
        n: Number of data points
        
    Returns:
        Updated results dictionary with AIC and BIC
    """
    for model_name, result in results.items():
        k = MODEL_PARAM_COUNT.get(model_name, 2)
        rmse = result.get('rmse', np.nan)
        
        if not np.isnan(rmse):
            result['aic'] = calculate_aic(n, k, rmse)
            result['bic'] = calculate_bic(n, k, rmse)
        else:
            result['aic'] = np.nan
            result['bic'] = np.nan
    
    return results