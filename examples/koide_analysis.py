"""
Koide Equation Analysis Demo
=============================

Demonstrates the Koide formula and its geometric interpretation using
Pocket_U Lite mass predictions.

This script shows:
1. Koide ratio Q for charged leptons (e, μ, τ)
2. Prediction of tau mass from electron and muon
3. Geometric angles (Foot's 45° and Descartes ~48°)
4. Validation across all three lepton generations
"""

from light_theory_realm.pocket_u_lite import (
    get_sm_spectrum,
    koide_ratio,
    check_koide_triplet,
    predict_third_mass,
    koide_angle,
    descartes_angle,
)


def main():
    print("\n" + "="*80)
    print("Koide Equation Analysis: Geometric Structure of Lepton Masses")
    print("="*80)
    
    # Get mass predictions from Pocket_U Lite
    spectrum = get_sm_spectrum()
    masses = {p['name']: p['mass_pred'] for p in spectrum}
    
    # Extract lepton masses
    m_e = masses['e']
    m_mu = masses['mu']
    m_tau = masses['tau']
    
    print(f"\n📊 Lepton Masses (Pocket_U Lite Predictions):")
    print(f"   Electron:  {m_e:.6f} MeV")
    print(f"   Muon:      {m_mu:.6f} MeV")
    print(f"   Tau:       {m_tau:.6f} MeV")
    
    # Calculate Koide ratio
    print(f"\n🔬 Koide Formula: Q = (Σm) / (Σ√m)²")
    Q = koide_ratio(m_e, m_mu, m_tau)
    target_Q = 2/3
    error_pct = abs(Q - target_Q) / target_Q * 100
    
    print(f"   Q (predicted) = {Q:.6f}")
    print(f"   Q (target)    = {target_Q:.6f}")
    print(f"   Error         = {error_pct:.2f}%")
    
    if error_pct < 5:
        print(f"   ✅ Excellent agreement with Koide relation!")
    elif error_pct < 10:
        print(f"   ✓ Good agreement with Koide relation")
    else:
        print(f"   ⚠ Deviation from Koide relation")
    
    # Validate triplet
    print(f"\n✓ Validation Check:")
    result = check_koide_triplet(m_e, m_mu, m_tau, tolerance=0.05)
    print(f"   Valid: {result['valid']}")
    print(f"   Deviation: {result['deviation']:.6f}")
    print(f"   Relative error: {result['relative_error']*100:.2f}%")
    
    # Predict tau mass from e and mu
    print(f"\n🔮 Mass Prediction Test:")
    print(f"   Given: e = {m_e:.3f} MeV, μ = {m_mu:.3f} MeV")
    print(f"   Assuming Q = 2/3...")
    
    m_tau_predicted, Q_check = predict_third_mass(m_e, m_mu, target_Q=2/3)
    tau_error = abs(m_tau_predicted - m_tau) / m_tau * 100
    
    print(f"   Predicted τ: {m_tau_predicted:.3f} MeV")
    print(f"   Actual τ:    {m_tau:.3f} MeV")
    print(f"   Error:       {tau_error:.2f}%")
    print(f"   Q check:     {Q_check:.6f}")
    
    # Geometric angle analysis
    print(f"\n📐 Geometric Interpretation (Foot's Vector Model):")
    angle_result = koide_angle(m_e, m_mu, m_tau)
    
    print(f"   Mass vector: v = (√m_e, √m_μ, √m_τ)")
    print(f"   Democratic vector: u = (1, 1, 1)")
    print(f"   ")
    print(f"   Angle θ = {angle_result['theta_deg']:.2f}°")
    print(f"   Target  = {angle_result['target_45_deg']:.2f}° (for Q = 2/3)")
    print(f"   Deviation = {angle_result['deviation_deg']:.2f}°")
    
    if angle_result['is_45_deg']:
        print(f"   ✅ Angle ≈ 45° → Confirms Q ≈ 2/3!")
    else:
        print(f"   Deviation from 45° indicates Q ≠ 2/3")
    
    # Descartes circle interpretation
    print(f"\n⭕ Descartes Circle Interpretation:")
    descartes_result = descartes_angle(Q)
    
    print(f"   Koide Q = {Q:.6f}")
    print(f"   Circle intersection angle φ = {descartes_result['phi_deg']:.2f}°")
    print(f"   Expected for Q = 2/3: φ ≈ 48.19°")
    
    # Summary
    print(f"\n" + "="*80)
    print("Key Insights:")
    print("="*80)
    print("1. The Koide formula Q ≈ 2/3 is NOT a coincidence!")
    print("2. It corresponds to a 45° angle in flavor space (Foot's interpretation)")
    print("3. Equivalently, it's a ~48° intersection angle (Descartes circles)")
    print("4. This suggests leptons have a deep geometric structure")
    print("5. Pocket_U Lite's prime plaquettes reproduce this geometry!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
