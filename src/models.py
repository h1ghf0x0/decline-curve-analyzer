"""
Arps DCA Model Definitions - Exponential, Hyperbolic, and Harmonic decline models.
"""

import numpy as np
from typing import Callable, Dict


def exponential_decline(t: np.ndarray, Qi: float, Di: float) -> np.ndarray:
    """
    Exponential decline model (b = 0).
    
    The exponential decline model assumes a constant decline rate.
    q = Qi * exp(-Di * t)
    
    Args:
        t: Time array (must be in same units as Di)
        Qi: Initial production rate
        Di: Initial decline rate (1/time unit)
        
    Returns:
        Production rate at each time point
        
    Notes:
        - Most appropriate for wells with constant bottomhole pressure
        - Decline rate remains constant throughout the well's life
        - Most conservative EUR estimate of the three models
    """
    t = np.asarray(t, dtype=float)
    # Ensure non-negative rates
    q = Qi * np.exp(-Di * t)
    return np.maximum(q, 0)


def hyperbolic_decline(t: np.ndarray, Qi: float, Di: float, b: float) -> np.ndarray:
    """
    Hyperbolic decline model (0 < b < 1).

    The hyperbolic decline model is the most general Arps model.
    q = Qi / (1 + b * Di * t)^(1/b)

    Args:
t: Time array (must be in same units as Di)
Qi: Initial production rate
Di: Initial decline rate (1/time unit)
b: Decline exponent (0 < b < 1)

    Returns:
Production rate at each time point

    Notes:
- Most flexible model, can match various decline behaviors
- b-factor controls the shape of the decline curve
- When b approaches 0, approaches exponential decline
- When b approaches 1, approaches harmonic decline
    """
    t = np.asarray(t, dtype=float)
    # Handle special case when b=0 (exponential decline)
    if b == 0:
        return exponential_decline(t, Qi, Di)
    # Avoid division by zero or negative values
    denominator = 1.0 + b * Di * t
    # Ensure denominator is positive
    denominator = np.maximum(denominator, 1e-10)
    q = Qi / np.power(denominator, 1.0 / b)
    return np.maximum(q, 0)


def harmonic_decline(t: np.ndarray, Qi: float, Di: float) -> np.ndarray:
    """
    Harmonic decline model (b = 1).
    
    The harmonic decline model is a special case of hyperbolic decline.
    q = Qi / (1 + Di * t)
    
    Args:
        t: Time array (must be in same units as Di)
        Qi: Initial production rate
        Di: Initial decline rate (1/time unit)
        
    Returns:
        Production rate at each time point
        
    Notes:
        - Special case of hyperbolic decline with b = 1
        - Slowest decline rate of the three models
        - Most optimistic EUR estimate
        - Often used for gas wells or wells with increasing bottomhole pressure
    """
    t = np.asarray(t, dtype=float)
    q = Qi / (1.0 + Di * t)
    return np.maximum(q, 0)


# Dictionary mapping model names to functions
ARPS_MODELS: Dict[str, Callable] = {
    'exponential': exponential_decline,
    'hyperbolic': hyperbolic_decline,
    'harmonic': harmonic_decline
}

# Number of parameters for each model
MODEL_PARAM_COUNT = {
    'exponential': 2,  # Qi, Di
    'hyperbolic': 3,   # Qi, Di, b
    'harmonic': 2      # Qi, Di
}

# Display names for models
MODEL_DISPLAY_NAMES = {
    'exponential': 'Exponential (b=0)',
    'hyperbolic': 'Hyperbolic (0<b<1)',
    'harmonic': 'Harmonic (b=1)'
}


def get_model_function(model_name: str) -> Callable:
    """
    Get the Arps model function by name.
    
    Args:
        model_name: Name of the model ('exponential', 'hyperbolic', 'harmonic')
        
    Returns:
        Model function
        
    Raises:
        ValueError: If model name is not recognized
    """
    if model_name not in ARPS_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(ARPS_MODELS.keys())}")
    return ARPS_MODELS[model_name]


def calculate_decline_rate(qi: float, di: float, b: float, t: np.ndarray) -> np.ndarray:
    """
    Calculate the instantaneous decline rate at each time point.
    
    D(t) = Di / (1 + b * Di * t)
    
    For exponential decline (b=0), D(t) = Di (constant).
    
    Args:
        qi: Initial production rate
        di: Initial decline rate
        b: Decline exponent
        t: Time array
        
    Returns:
        Instantaneous decline rate at each time point
    """
    t = np.asarray(t, dtype=float)
    if b == 0:
        return np.full_like(t, di)
    return di / (1.0 + b * di * t)


def calculate_effective_decline_rate(di: float, b: float, period: float = 1.0) -> float:
    """
    Calculate the effective decline rate over a period.
    
    The effective decline rate is the percentage decline over a specific
    time period (usually 1 year).
    
    De = 1 - (1 + b * Di * period)^(-1/b)
    
    Args:
        di: Nominal (continuous) decline rate
        b: Decline exponent
        period: Time period for calculation (default: 1 year)
        
    Returns:
        Effective decline rate over the period
    """
    if b == 0:
        # Exponential: De = 1 - exp(-Di * period)
        return 1.0 - np.exp(-di * period)
    return 1.0 - np.power(1.0 + b * di * period, -1.0 / b)


def calculate_time_to_abandonment(qi: float, di: float, b: float, 
                                   q_abandon: float) -> float:
    """
    Calculate the time to reach abandonment rate.
    
    Args:
        qi: Initial production rate
        di: Initial decline rate
        b: Decline exponent
        q_abandon: Abandonment rate threshold
        
    Returns:
        Time to reach abandonment rate (in same units as di)
    """
    if q_abandon >= qi:
        return 0.0
    
    if b == 0:
        # Exponential: t = -ln(q_abandon/qi) / Di
        return -np.log(q_abandon / qi) / di
    
    # Hyperbolic/Harmonic: t = ((qi/q_abandon)^b - 1) / (b * Di)
    return ((qi / q_abandon) ** b - 1.0) / (b * di)