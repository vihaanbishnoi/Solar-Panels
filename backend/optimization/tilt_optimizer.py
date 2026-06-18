"""
Tilt Angle Optimizer
Uses ML model predictions to find the optimal solar panel tilt angle
"""

import os
import joblib
import numpy as np
import pandas as pd

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the efficiency loss model
MODEL_PATH = os.path.join(BASE_DIR, "efficiency_loss_model.pkl")
rf_model = None

try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"✅ Tilt optimizer model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Could not load tilt optimizer model: {e}")


def predict_loss(feature_vector: list) -> float:
    """
    Predict efficiency loss using the Random Forest model.
    
    Args:
        feature_vector: [temperature, humidity, wind_speed, irradiance, 
                        voltage, current, days_since_installation]
    
    Returns:
        Predicted efficiency loss (0-1)
    """
    if rf_model is None:
        # Fallback simulation if model not loaded
        return 0.15
    
    try:
        X = pd.DataFrame(
            [feature_vector],
            columns=rf_model.feature_names_in_
        )
        loss = rf_model.predict(X)[0]
        return max(0, min(loss, 1))
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.15


def solar_elevation_angle(latitude: float, hour: float) -> float:
    """
    Calculate the solar elevation angle based on latitude and hour of day.
    """
    max_elevation = 90 - abs(latitude)
    elevation = max_elevation * np.sin(np.pi * (hour - 6) / 12)
    return max(elevation, 0)


def effective_irradiance(ghi: float, tilt: float, solar_elevation: float) -> float:
    """
    Calculate effective irradiance on a tilted surface.
    """
    angle_diff = np.radians(abs(tilt - solar_elevation))
    return max(ghi * np.cos(angle_diff), 0)


def find_best_tilt(
    ghi: float,
    latitude: float,
    hour: float,
    feature_vector: list,
    tilt_min: int = 0,
    tilt_max: int = 60
) -> tuple:
    """
    Find the optimal tilt angle for maximum energy output.
    
    Args:
        ghi: Global Horizontal Irradiance in W/m²
        latitude: Location latitude in degrees
        hour: Hour of day (0-24)
        feature_vector: Environmental conditions for loss prediction
        tilt_min: Minimum tilt angle to consider
        tilt_max: Maximum tilt angle to consider
    
    Returns:
        Tuple of (best_tilt_angle, estimated_net_energy)
    """
    solar_elev = solar_elevation_angle(latitude, hour)
    loss = predict_loss(feature_vector)
    loss = min(loss, 0.8)  # Cap loss at 80%

    best_tilt = None
    best_energy = -1
    all_results = []

    for tilt in range(tilt_min, tilt_max + 1):
        eff_irr = effective_irradiance(ghi, tilt, solar_elev)
        net_energy = eff_irr * (1 - loss)
        
        all_results.append({
            'tilt': tilt,
            'effective_irradiance': round(eff_irr, 2),
            'net_energy': round(net_energy, 2)
        })

        if net_energy > best_energy:
            best_energy = net_energy
            best_tilt = tilt

    return best_tilt, best_energy, solar_elev, loss, all_results


def get_optimization_result(
    ghi: float,
    latitude: float,
    hour: float,
    temperature: float = 25.0,
    humidity: float = 50.0,
    wind_speed: float = 2.0,
    voltage: float = 38.0,
    current: float = 8.0,
    days_since_installation: int = 365,
    tilt_min: int = 0,
    tilt_max: int = 60
) -> dict:
    """
    Get a complete optimization result with all details.
    """
    feature_vector = [
        temperature,
        humidity,
        wind_speed,
        ghi,  # irradiance
        voltage,
        current,
        days_since_installation
    ]
    
    best_tilt, best_energy, solar_elev, loss, all_results = find_best_tilt(
        ghi=ghi,
        latitude=latitude,
        hour=hour,
        feature_vector=feature_vector,
        tilt_min=tilt_min,
        tilt_max=tilt_max
    )
    
    # Get energy at current/default tilt for comparison
    default_tilt = int(abs(latitude))  # Common rule of thumb
    default_energy = next(
        (r['net_energy'] for r in all_results if r['tilt'] == default_tilt),
        best_energy * 0.9
    )
    
    improvement = ((best_energy - default_energy) / default_energy * 100) if default_energy > 0 else 0
    
    return {
        'optimal_tilt': best_tilt,
        'estimated_energy': round(best_energy, 2),
        'solar_elevation': round(solar_elev, 2),
        'efficiency_loss': round(loss * 100, 2),
        'efficiency_retained': round((1 - loss) * 100, 2),
        'default_tilt': default_tilt,
        'default_energy': round(default_energy, 2),
        'improvement_percent': round(improvement, 2),
        'conditions': {
            'ghi': ghi,
            'latitude': latitude,
            'hour': hour,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed
        },
        'tilt_curve': all_results[::5]  # Every 5 degrees for chart
    }
