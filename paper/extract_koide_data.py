#!/usr/bin/env python3
"""
Extract Koide rigidity data from THRML_Geometry logs.

This script scans all training log files to demonstrate the stability of the
Koide functional Q across multiple runs with the same prime-plaquette topology.
It compiles statistics for Section 5 of the paper ("Prime-Plaquette Koide Geometry").
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Paths
LOGS_DIR = Path("/home/jdimi/projects/to_observe_proof/THRML_Geometry/thrml_geom/pocket_u/logs")
OUTPUT_DIR = Path("/home/jdimi/projects/to_observe_proof/Light_Theory_Realm/paper")

def extract_koide_q(masses: List[float]) -> float:
    """Calculate Koide Q from three masses."""
    if len(masses) != 3:
        return None
    m_e, m_mu, m_tau = masses
    numerator = m_e + m_mu + m_tau
    denominator = (np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau))**2
    return numerator / denominator if denominator > 0 else None

def parse_training_log(log_path: Path) -> Dict:
    """Parse a training log file and extract key metrics."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    result = {
        'file': log_path.name,
        'lambda_koide': None,
        'Q_lepton': None,
        'masses': {},
        'mass_errors': {},
        'cabibbo_angle': None
    }
    
    # Extract lambda_koide
    lambda_match = re.search(r'lambda_koide\s*=\s*([\d.]+)', content)
    if lambda_match:
        result['lambda_koide'] = float(lambda_match.group(1))
    
    # Extract lepton masses from final epoch
    mass_pattern = r'(\w+_L):\s*corrected=([\d.]+)\s*MeV.*?target=([\d.]+)\s*MeV.*?error=([\d.]+)%'
    masses_section = content.split('Epoch 99/100')[-1] if 'Epoch 99/100' in content else content
    
    for match in re.finditer(mass_pattern, masses_section):
        particle, pred, target, error = match.groups()
        result['masses'][particle] = float(pred)
        result['mass_errors'][particle] = float(error)
    
    # Calculate Koide Q for leptons
    if all(p in result['masses'] for p in ['e_L', 'mu_L', 'tau_L']):
        lepton_masses = [result['masses']['e_L'], 
                        result['masses']['mu_L'], 
                        result['masses']['tau_L']]
        result['Q_lepton'] = extract_koide_q(lepton_masses)
    
    # Extract Cabibbo angle
    cabibbo_match = re.search(r'cabibbo_angle_deg\s*=\s*([\d.]+)', masses_section)
    if cabibbo_match:
        result['cabibbo_angle'] = float(cabibbo_match.group(1))
    
    return result

def main():
    """Extract and analyze Koide rigidity data."""
    print("Scanning training logs for Koide rigidity analysis...")
    
    valid_runs = []
    
    # Scan all training log files
    for log_file in sorted(LOGS_DIR.glob("koide_training_results*.txt")):
        try:
            data = parse_training_log(log_file)
            # We only care about runs that actually produced a Q value
            if data['Q_lepton'] is not None:
                valid_runs.append(data)
                    
        except Exception as e:
            print(f"Error parsing {log_file.name}: {e}")
    
    print(f"\nFound {len(valid_runs)} valid runs with computed Q values.")
    
    # Compile statistics
    q_values = [r['Q_lepton'] for r in valid_runs]
    
    stats = {
        'total_runs': len(valid_runs),
        'q_values': q_values,
        'runs': valid_runs
    }
    
    # Calculate Q statistics
    if q_values:
        Q_arr = np.array(q_values)
        stats['q_mean'] = float(np.mean(Q_arr))
        stats['q_std'] = float(np.std(Q_arr))
        stats['q_min'] = float(np.min(Q_arr))
        stats['q_max'] = float(np.max(Q_arr))
        
        print(f"\nKoide Q Rigidity Statistics:")
        print(f"  Mean: {stats['q_mean']:.9f}")
        print(f"  Std:  {stats['q_std']:.9f}")
        print(f"  Min:  {stats['q_min']:.9f}")
        print(f"  Max:  {stats['q_max']:.9f}")
        
        if stats['q_std'] < 1e-6:
            print("\n✅ Q is effectively rigid across all runs!")
        else:
            print("\n⚠️ Q shows variance - check plaquette consistency.")
    
    # Save results
    output_file = OUTPUT_DIR / "koide_rigidity_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return stats

if __name__ == "__main__":
    main()
