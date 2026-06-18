"""
Solar Geometry Calculations
Provides functions for calculating solar position and effective irradiance
"""

import numpy as np


def solar_elevation_angle(latitude: float, hour: float) -> float:
    """
    Calculate the solar elevation angle based on latitude and hour of day.
    
    Args:
        latitude: Location latitude in degrees
        hour: Hour of day (0-24, where 12 is solar noon)
    
    Returns:
        Solar elevation angle in degrees (0-90)
    """
    max_elevation = 90 - abs(latitude)
    elevation = max_elevation * np.sin(np.pi * (hour - 6) / 12)
    return max(elevation, 0)


def effective_irradiance(ghi: float, tilt: float, solar_elevation: float) -> float:
    """
    Calculate effective irradiance on a tilted surface.
    
    Args:
        ghi: Global Horizontal Irradiance in W/m²
        tilt: Panel tilt angle in degrees
        solar_elevation: Solar elevation angle in degrees
    
    Returns:
        Effective irradiance on the tilted surface in W/m²
    """
    angle_diff = np.radians(abs(tilt - solar_elevation))
    return max(ghi * np.cos(angle_diff), 0)
