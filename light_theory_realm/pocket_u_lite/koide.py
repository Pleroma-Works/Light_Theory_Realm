"""
Koide Equation Utilities
=========================

The Koide formula is an empirical relation discovered by Yoshio Koide (1981)
that relates the masses of the three charged leptons:

    Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²

For the experimentally measured lepton masses, Q ≈ 2/3 with remarkable precision.

This module provides utilities to:
- Calculate the Koide ratio Q for any mass triplet
- Validate if a triplet satisfies the Koide relation
- Predict the third mass given two masses and target Q
- Calculate geometric angles from Foot's vector interpretation

Geometric Interpretation
------------------------
The Koide ratio Q = 2/3 has deep geometric significance:

1. **Foot's Vector Interpretation**: The mass vector (√m_e, √m_μ, √m_τ)
   makes a 45° angle with the "democratic" direction (1, 1, 1) in flavor space.
   
2. **Descartes Circle Formula**: Q = 2/3 corresponds to an intersection
   angle φ = arccos(2/3) ≈ 48.2° between circles representing particles.

3. **Curvature Resonance**: Suggests masses arise from geometric structure
   in spacetime, not arbitrary parameters.

References
----------
- Koide, Y. (1981). "New view of quark and lepton mass hierarchy"
- Foot, R. (1994). "A note on Koide's lepton mass relation"
- KOIDE_GEOMETRIC_ORIGIN_ANALYSIS.md in THRML_Geometry
"""

import numpy as np
from typing import Dict, Tuple, Optional


def koide_ratio(m1: float, m2: float, m3: float) -> float:
    """
    Calculate the Koide ratio Q for three masses.
    
    Q = (m₁ + m₂ + m₃) / (√m₁ + √m₂ + √m₃)²
    
    For charged leptons (e, μ, τ), Q ≈ 2/3 ≈ 0.666667.
    
    Args:
        m1, m2, m3: Particle masses (must be positive, same units)
        
    Returns:
        Q value (dimensionless)
        
    Example:
        >>> Q = koide_ratio(0.511, 105.66, 1776.86)  # e, μ, τ in MeV
        >>> print(f"Q = {Q:.6f}")  # Q ≈ 0.661
    """
    if m1 <= 0 or m2 <= 0 or m3 <= 0:
        raise ValueError("All masses must be positive")
    
    numerator = m1 + m2 + m3
    denominator = (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)) ** 2
    return numerator / denominator


def check_koide_triplet(
    m1: float, 
    m2: float, 
    m3: float,
    target_Q: float = 2/3,
    tolerance: float = 0.01
) -> Dict[str, float]:
    """
    Check if three masses satisfy the Koide relation.
    
    Args:
        m1, m2, m3: Particle masses (same units)
        target_Q: Expected Q value (default: 2/3)
        tolerance: Acceptable absolute deviation from target
        
    Returns:
        Dictionary containing:
            - Q: Calculated Koide ratio
            - target: Target Q value
            - deviation: |Q - target|
            - relative_error: deviation / target
            - valid: True if within tolerance
            
    Example:
        >>> result = check_koide_triplet(0.511, 105.66, 1776.86)
        >>> print(f"Valid: {result['valid']}, Q = {result['Q']:.6f}")
    """
    Q = koide_ratio(m1, m2, m3)
    deviation = abs(Q - target_Q)
    
    return {
        'Q': Q,
        'target': target_Q,
        'deviation': deviation,
        'relative_error': deviation / target_Q,
        'valid': deviation < tolerance,
        'masses': (m1, m2, m3)
    }


def predict_third_mass(
    m1: float,
    m2: float,
    target_Q: float = 2/3
) -> Tuple[float, float]:
    """
    Predict the third mass given two masses and target Koide ratio.
    
    Solves the quadratic equation:
        Q(√m₁ + √m₂ + √m₃)² = m₁ + m₂ + m₃
    
    for m₃, which gives two solutions (we return the larger one as physical).
    
    Args:
        m1, m2: Two known masses
        target_Q: Target Koide ratio (default: 2/3)
        
    Returns:
        Tuple of (m3_solution, Q_check):
            - m3_solution: Predicted third mass
            - Q_check: Verification that Q(m1, m2, m3) ≈ target_Q
            
    Example:
        >>> # Predict tau mass from electron and muon
        >>> m_tau_pred, Q = predict_third_mass(0.511, 105.66)
        >>> print(f"Predicted tau: {m_tau_pred:.2f} MeV (Q = {Q:.6f})")
    """
    if m1 <= 0 or m2 <= 0:
        raise ValueError("Masses must be positive")
    if target_Q <= 1/3 or target_Q >= 1:
        raise ValueError("Target Q must be in range (1/3, 1)")
    
    # Let x = √m₃
    # Then: Q(√m₁ + √m₂ + x)² = m₁ + m₂ + x²
    # Expanding: Q(a + x)² = b + x² where a = √m₁ + √m₂, b = m₁ + m₂
    # Q(a² + 2ax + x²) = b + x²
    # Qa² + 2Qax + Qx² = b + x²
    # (Q - 1)x² + 2Qax + (Qa² - b) = 0
    
    a = np.sqrt(m1) + np.sqrt(m2)
    b = m1 + m2
    
    # Quadratic coefficients
    A = target_Q - 1
    B = 2 * target_Q * a
    C = target_Q * a**2 - b
    
    # Solve quadratic equation
    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        raise ValueError(f"No real solution exists for Q={target_Q}")
    
    x1 = (-B + np.sqrt(discriminant)) / (2*A)
    x2 = (-B - np.sqrt(discriminant)) / (2*A)
    
    # Take the positive solution (x = √m₃ must be positive)
    x = max(x1, x2)
    if x <= 0:
        raise ValueError("No positive solution found")
    
    m3 = x**2
    
    # Verify
    Q_check = koide_ratio(m1, m2, m3)
    
    return m3, Q_check


def koide_angle(m1: float, m2: float, m3: float) -> Dict[str, float]:
    """
    Calculate the geometric angle from Foot's vector interpretation.
    
    In flavor space, define:
        - Mass vector: v = (√m₁, √m₂, √m₃)
        - Democratic vector: u = (1, 1, 1)
    
    The angle θ between them is:
        cos(θ) = (v·u) / (|v||u|) = (√m₁ + √m₂ + √m₃) / √(3(m₁ + m₂ + m₃))
    
    For Q = 2/3, this gives θ = π/4 = 45° exactly!
    
    Args:
        m1, m2, m3: Particle masses
        
    Returns:
        Dictionary containing:
            - theta_rad: Angle in radians
            - theta_deg: Angle in degrees
            - cos_theta: Cosine of the angle
            - Q: Koide ratio
            - is_45_deg: True if angle ≈ 45° (within 1°)
            
    Example:
        >>> result = koide_angle(0.511, 105.66, 1776.86)
        >>> print(f"Angle: {result['theta_deg']:.2f}° (45° expected)")
    """
    if m1 <= 0 or m2 <= 0 or m3 <= 0:
        raise ValueError("All masses must be positive")
    
    # Mass vector v = (√m₁, √m₂, √m₃)
    v = np.array([np.sqrt(m1), np.sqrt(m2), np.sqrt(m3)])
    
    # Democratic vector u = (1, 1, 1)
    u = np.array([1.0, 1.0, 1.0])
    
    # Compute angle
    cos_theta = np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u))
    
    # Clamp to [-1, 1] to avoid numerical issues with arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    
    # Check if close to 45°
    is_45_deg = abs(theta_deg - 45.0) < 1.0
    
    # Also compute Q for reference
    Q = koide_ratio(m1, m2, m3)
    
    return {
        'theta_rad': float(theta_rad),
        'theta_deg': float(theta_deg),
        'cos_theta': float(cos_theta),
        'Q': float(Q),
        'is_45_deg': bool(is_45_deg),
        'target_45_deg': 45.0,
        'deviation_deg': float(abs(theta_deg - 45.0))
    }


def descartes_angle(Q: float) -> Dict[str, float]:
    """
    Calculate the Descartes circle intersection angle from Koide ratio.
    
    The Koide formula is mathematically equivalent to a generalized
    Descartes Circle Formula, where Q determines the intersection angle φ:
    
        φ = arccos(Q)
    
    For Q = 2/3, this gives φ ≈ 48.2°.
    
    Args:
        Q: Koide ratio
        
    Returns:
        Dictionary containing:
            - phi_rad: Descartes angle in radians
            - phi_deg: Descartes angle in degrees
            - Q: Input Koide ratio
            
    Example:
        >>> result = descartes_angle(2/3)
        >>> print(f"Descartes angle: {result['phi_deg']:.2f}°")
    """
    if Q < 1/3 or Q > 1:
        raise ValueError("Q must be in range [1/3, 1]")
    
    phi_rad = np.arccos(Q)
    phi_deg = np.degrees(phi_rad)
    
    return {
        'phi_rad': float(phi_rad),
        'phi_deg': float(phi_deg),
        'Q': float(Q)
    }
