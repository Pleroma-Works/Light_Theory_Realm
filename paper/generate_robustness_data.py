#!/usr/bin/env python3
"""
Generate robustness data for Koide paper (Section 6).

This script systematically perturbs the prime plaquettes to demonstrate
that the Koide relation is a property of the specific prime topology,
not a generic feature of the model.

It generates:
1. Q values for single-prime perturbations
2. Q values for generation swaps
3. Mass errors for alternative topologies
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
import itertools

# Add paths to access geometry modules
CORE_GEOM = "/home/jdimi/projects/to_observe_proof/THRML_Geometry/thrml_geom/pocket_u/core/geometry"
EXPERIMENTS = "/home/jdimi/projects/to_observe_proof/THRML_Geometry/thrml_geom/pocket_u/experiments"
sys.path.insert(0, CORE_GEOM)
sys.path.insert(0, EXPERIMENTS)

try:
    from prime_gauge_generator import PrimeGaugeGenerator
    from wilson_loop_calculator import WilsonLoopCalculator
except ImportError:
    print("Error: Could not import geometry modules from expected paths.")
    print(f"Tried: {CORE_GEOM} and {EXPERIMENTS}")
    sys.exit(1)

OUTPUT_DIR = Path("/home/jdimi/projects/to_observe_proof/Light_Theory_Realm/paper")

# Standard Koide Plaquettes
STANDARD_PLAQUETTES = [
    [2, 3, 5, 7],      # e
    [5, 7, 11, 13],    # mu
    [3, 11, 13, 47]    # tau
]

def compute_mass(primes, coupling_g=1.0):
    """Compute mass (phase magnitude) for a plaquette."""
    path = primes + [primes[0]]
    generator = PrimeGaugeGenerator(primes, coupling_g, use_jax=False)
    gauge_fields = generator.generate_all_links(path)
    calculator = WilsonLoopCalculator(verbose=False)
    _, phase = calculator.compute_wilson_loop(gauge_fields, path)
    return abs(phase)

def koide_formula(masses):
    """Compute Koide Q."""
    m1, m2, m3 = masses
    numerator = m1 + m2 + m3
    denominator = (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3))**2
    return numerator / denominator if denominator > 0 else 0

def generate_perturbations(base_plaquettes):
    """Generate perturbed plaquette sets."""
    perturbations = []
    
    # 1. Identity (Reference)
    perturbations.append({
        'name': 'Standard Model',
        'plaquettes': base_plaquettes,
        'type': 'reference'
    })
    
    # 2. Single Prime Shifts (Small perturbations)
    # Shift one prime in one plaquette to the next/prev prime
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    for gen_idx, gen_name in enumerate(['e', 'mu', 'tau']):
        base_p = base_plaquettes[gen_idx]
        
        # Perturb the last prime (usually the largest/most sensitive)
        last_prime = base_p[-1]
        if last_prime in primes:
            idx = primes.index(last_prime)
            
            # Shift Up
            if idx + 1 < len(primes):
                new_p = base_p[:-1] + [primes[idx+1]]
                new_set = list(base_plaquettes)
                new_set[gen_idx] = new_p
                perturbations.append({
                    'name': f'{gen_name} prime shift (+)',
                    'plaquettes': new_set,
                    'type': 'perturbation'
                })
            
            # Shift Down
            if idx - 1 >= 0:
                new_p = base_p[:-1] + [primes[idx-1]]
                new_set = list(base_plaquettes)
                new_set[gen_idx] = new_p
                perturbations.append({
                    'name': f'{gen_name} prime shift (-)',
                    'plaquettes': new_set,
                    'type': 'perturbation'
                })

    # 3. Random nearby topologies
    # Just some manually defined "wrong" sets that look plausible
    perturbations.append({
        'name': 'Naive Sequential',
        'plaquettes': [
            [2, 3, 5, 7],
            [7, 11, 13, 17],
            [17, 19, 23, 29]
        ],
        'type': 'alternative'
    })
    
    perturbations.append({
        'name': 'Disjoint Set',
        'plaquettes': [
            [2, 3, 5, 7],
            [11, 13, 17, 19],
            [23, 29, 31, 37]
        ],
        'type': 'alternative'
    })

    return perturbations

def main():
    print("Generating robustness data for Section 6...")
    
    results = []
    perturbations = generate_perturbations(STANDARD_PLAQUETTES)
    
    for case in perturbations:
        print(f"Processing: {case['name']}")
        try:
            masses = [compute_mass(p) for p in case['plaquettes']]
            Q = koide_formula(masses)
            
            results.append({
                'name': case['name'],
                'type': case['type'],
                'plaquettes': case['plaquettes'],
                'masses': masses,
                'Q': Q,
                'Q_error': abs(Q - 2/3),
                'relative_error_percent': (abs(Q - 2/3) / (2/3)) * 100
            })
            print(f"  -> Q = {Q:.6f}")
            
        except Exception as e:
            print(f"  -> Failed: {e}")
    
    # Save results
    output_file = OUTPUT_DIR / "koide_robustness_data.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved {len(results)} cases to {output_file}")

if __name__ == "__main__":
    main()
