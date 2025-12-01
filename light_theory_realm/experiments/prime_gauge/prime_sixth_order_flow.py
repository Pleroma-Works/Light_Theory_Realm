"""
Prime Sixth-Order Flow Experiment

A minimal "prime light string" universe where:
- Geometry is fixed by a 6th-order elliptic operator on a prime graph
- It relaxes via gradient flow (pseudo-time, not Hamiltonian)
- We compute a toy ρ_IGBP on the resulting configuration

This is a structurally faithful Light Theory experiment that demonstrates:
1. 6th-order elliptic operator L₆ = Δ³ on discrete prime ring
2. Gradient flow dynamics: ∂_τ ψ = -H ψ where H = L₆²
3. Toy ρ_IGBP = ||∇S_rep||² / (|Ω| + ε) combining entropy and Berry-like twist

Physical interpretation:
- The prime ring [2,3,5,7] is a microscopic "light string"
- L₆ is the "information-elasticity" operator (punishes high-curvature deformations)
- Gradient flow is pseudo-time relaxation toward self-consistent geometry
- ρ_IGBP is the order parameter: "stiff/light-like" vs "soft/massive" regime
"""

import jax
import jax.numpy as jnp
from jax import jit
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from light_theory_realm.engine import CliffordEngine
from light_theory_realm.experiments.prime_gauge.prime_plaquette import PrimeLattice


# -----------------------------------------------------------------------------
# Prime ring + operators
# -----------------------------------------------------------------------------

def build_prime_ring():
    """
    Build a 4-node prime ring: 2 ↔ 3 ↔ 5 ↔ 7 ↔ 2
    
    Returns:
        primes: (4,) float array of prime numbers
        laplacian: (4,4) symmetric graph Laplacian Δ = D - A
        phases: (4,4) log-ratio phase matrix θ_ij = ln(p_j/p_i) for edges
    
    Notes:
        - Adjacency is bidirectional ring (i ↔ i±1 mod 4)
        - Laplacian is symmetric and positive semidefinite
        - Phases are only non-zero for actual edges
    """
    primes = jnp.array([2.0, 3.0, 5.0, 7.0])
    N = primes.shape[0]
    
    # Adjacency for bidirectional ring
    A = jnp.zeros((N, N))
    idx = jnp.arange(N)
    A = A.at[idx, (idx + 1) % N].set(1.0)  # Forward edges
    A = A.at[idx, (idx - 1) % N].set(1.0)  # Backward edges
    
    # Degree matrix
    D = jnp.diag(jnp.sum(A, axis=1))
    
    # Graph Laplacian (symmetric, positive semidefinite)
    laplacian = D - A
    
    # Log-ratio phases (only on existing edges)
    phases = jnp.zeros((N, N))
    for i in range(N):
        j_f = (i + 1) % N  # Forward neighbor
        j_b = (i - 1) % N  # Backward neighbor
        phases = phases.at[i, j_f].set(jnp.log(primes[j_f] / primes[i]))
        phases = phases.at[i, j_b].set(jnp.log(primes[j_b] / primes[i]))
    
    return primes, laplacian, phases


def build_sixth_order_operator(laplacian: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build the 6th-order elliptic operator L₆ = Δ³ and flow operator H = L₆².
    
    Args:
        laplacian: (N,N) symmetric graph Laplacian
    
    Returns:
        L6: 6th-order operator Δ³
        H: Flow operator (L₆)² for gradient descent
    
    Notes:
        - For gradient flow we use H = L₆² (positive semidefinite)
        - Since Δ is symmetric, L₆ is also symmetric
        - H is the dissipative flow operator for the action ∫|L₆ψ|²
    """
    # Δ³
    L2 = laplacian @ laplacian
    L6 = L2 @ laplacian
    
    # H = (Δ³)² for gradient flow
    H = L6 @ L6
    
    return L6, H


# -----------------------------------------------------------------------------
# ρ_IGBP components
# -----------------------------------------------------------------------------

def compute_S_rep(psi: jnp.ndarray, eps: float = 1e-12) -> float:
    """
    Compute global representational entropy S_rep from state ψ.
    
    Args:
        psi: (N,) state vector
        eps: Small constant to prevent log(0)
    
    Returns:
        S_rep: Scalar entropy in [0, log(N)]
    
    Formula:
        π_i = ψ_i² / Σψ_j²
        S_rep = -Σ π_i log(π_i + ε)
    """
    prob = psi ** 2
    prob = prob / (jnp.sum(prob) + eps)
    S_local = -prob * jnp.log(prob + eps)
    return float(jnp.sum(S_local))


def compute_local_S_rep(psi: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Compute node-wise entropy contributions.
    
    Args:
        psi: (N,) state vector
        eps: Small constant to prevent log(0)
    
    Returns:
        S_local: (N,) array of local entropy contributions -π_i log(π_i)
    """
    prob = psi ** 2
    prob = prob / (jnp.sum(prob) + eps)
    return -prob * jnp.log(prob + eps)


def compute_gradient_norm(S_local: jnp.ndarray, laplacian: jnp.ndarray) -> float:
    """
    Compute discrete entropy gradient norm ||∇S||² via Laplacian.
    
    Args:
        S_local: (N,) local entropy field
        laplacian: (N,N) graph Laplacian
    
    Returns:
        ||∇S||²: Scalar norm (non-negative)
    
    Formula:
        ∇S ≈ Δ·S_local (discrete derivative)
        ||∇S||² = (∇S)ᵀ(∇S)
    """
    grad_S = laplacian @ S_local
    return float(grad_S @ grad_S)


def compute_plaquette_holonomy(phases: jnp.ndarray) -> float:
    """
    Compute total phase accumulated around the ring plaquette.
    
    Args:
        phases: (N,N) phase matrix θ_ij = ln(p_j/p_i)
    
    Returns:
        Φ: Total holonomy phase (sum of forward edge phases)
    
    Notes:
        For [2,3,5,7] ring: Φ = ln(3/2) + ln(5/3) + ln(7/5) + ln(2/7) = ln(1) = 0
        This gives a "flat" but high-ρ_IGBP configuration.
    """
    N = phases.shape[0]
    total = 0.0
    for i in range(N):
        j_f = (i + 1) % N
        total += phases[i, j_f]
    return float(total)


def compute_berry_norm(phi: float) -> float:
    """
    Compute toy Berry curvature proxy |Ω| = |sin Φ|.
    
    Args:
        phi: Plaquette holonomy phase
    
    Returns:
        |Ω|: Berry curvature proxy in [0, 1]
    """
    return float(jnp.abs(jnp.sin(phi)))


def compute_igbp(psi: jnp.ndarray,
                 laplacian: jnp.ndarray,
                 phases: jnp.ndarray,
                 eps: float = 1e-9) -> Dict[str, float]:
    """
    Compute toy ρ_IGBP and all its components.
    
    Args:
        psi: (N,) current state
        laplacian: (N,N) graph Laplacian
        phases: (N,N) phase matrix
        eps: Regularization for division
    
    Returns:
        Dictionary with:
            - rho_igbp: ρ_IGBP = ||∇S||² / (|Ω| + ε)
            - grad_norm: ||∇S||² (entropy gradient norm)
            - omega: |Ω| (Berry curvature proxy)
            - phi: Φ (plaquette holonomy)
    
    Physical interpretation:
        - High ρ_IGBP: "stiff/light-like" regime (entropy gradients dominate)
        - Low ρ_IGBP: "soft/massive" regime (Berry twist dominates)
    """
    S_local = compute_local_S_rep(psi, eps=eps)
    grad_norm = compute_gradient_norm(S_local, laplacian)
    phi = compute_plaquette_holonomy(phases)
    omega = compute_berry_norm(phi)
    rho = grad_norm / (omega + eps)
    
    return {
        "rho_igbp": float(rho),
        "grad_norm": float(grad_norm),
        "omega": float(omega),
        "phi": float(phi),
    }


# -----------------------------------------------------------------------------
# Gradient flow
# -----------------------------------------------------------------------------

@jit
def gradient_flow_step(psi: jnp.ndarray,
                       H: jnp.ndarray,
                       dt: float) -> jnp.ndarray:
    """
    Single pseudo-time gradient flow step (JIT-compiled).
    
    Args:
        psi: (N,) current state
        H: (N,N) flow operator H = L₆²
        dt: Pseudo-time step size
    
    Returns:
        psi_next: (N,) updated and normalized state
    
    Evolution equation:
        ψ_{t+dt} = ψ_t - dt·H·ψ_t
    
    where H = L₆² is positive semidefinite, ensuring dissipative flow.
    We normalize after each step to keep ||ψ|| = 1.
    """
    update = H @ psi
    psi_next = psi - dt * update
    
    # Renormalization (keeps state bounded, aids interpretation)
    norm = jnp.linalg.norm(psi_next)
    psi_next = jnp.where(norm > 0, psi_next / norm, psi_next)
    
    return psi_next


# -----------------------------------------------------------------------------
# Main experiment runner
# -----------------------------------------------------------------------------

def run_prime_sixth_order_flow(
    n_steps: int = 200,
    dt: float = 0.01,
    psi_init: Optional[jnp.ndarray] = None,
    verbose_every: int = 20,
    log_to_file: bool = True,
) -> Dict[str, Any]:
    """
    Main driver for the Prime Sixth-Order Flow experiment.
    
    Args:
        n_steps: Number of pseudo-time iterations
        dt: Time step size for gradient flow
        psi_init: Initial state (4,) array, or None for random initialization
        verbose_every: Print diagnostics every N steps (0 to disable)
        log_to_file: If True, save output to timestamped log file
    
    Returns:
        Dictionary containing:
            - primes: (4,) prime numbers
            - laplacian: (4,4) graph Laplacian
            - phases: (4,4) phase matrix
            - L6: (4,4) 6th-order operator
            - H: (4,4) flow operator
            - psi_init: (4,) initial state
            - psi_final: (4,) final state
            - history: List of dicts with time series data
            - log_file: Path to log file (if log_to_file=True)
    
    Example:
        >>> results = run_prime_sixth_order_flow(n_steps=100, dt=0.01)
        >>> print(f"Final ρ_IGBP: {results['history'][-1]['rho_igbp']:.3e}")
    """
    # Setup logging
    log_file = None
    if log_to_file:
        # Create logs directory
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"prime_sixth_order_flow_{timestamp}.log"
        
        # Open log file with UTF-8 encoding
        log_handle = open(log_file, 'w', encoding='utf-8')
        
        def log_print(msg):
            """Print to both console and log file"""
            print(msg)
            log_handle.write(msg + '\n')
            log_handle.flush()
    else:
        def log_print(msg):
            """Print to console only"""
            print(msg)
    
    # Build infrastructure
    primes, laplacian, phases = build_prime_ring()
    L6, H = build_sixth_order_operator(laplacian)
    
    # Initialize state
    if psi_init is None:
        key = jax.random.PRNGKey(0)
        psi_init = jax.random.normal(key, (primes.shape[0],))
        psi_init = psi_init / jnp.linalg.norm(psi_init)
    
    psi = psi_init
    history = []
    
    # Evolution loop
    for step in range(n_steps + 1):
        # Compute diagnostics
        S = float(compute_S_rep(psi))
        igbp = compute_igbp(psi, laplacian, phases)
        
        # Record history
        entry = {
            "step": step,
            "S_rep": S,
            "rho_igbp": igbp["rho_igbp"],
            "grad_norm": igbp["grad_norm"],
            "omega": igbp["omega"],
            "phi": igbp["phi"],
            "psi": psi,
        }
        history.append(entry)
        
        # Print progress
        if verbose_every and (step % verbose_every == 0):
            log_print(
                f"step={step:4d} | S_rep={S:.6f} "
                f"| ρ_IGBP={igbp['rho_igbp']:.6e} "
                f"| |∇S|²={igbp['grad_norm']:.6e} "
                f"| |Ω|={igbp['omega']:.6e}"
            )
        
        # Apply gradient flow (skip on last step)
        if step < n_steps:
            psi = gradient_flow_step(psi, H, dt)
    
    # Close log file
    if log_to_file:
        log_handle.close()
    
    result = {
        "primes": primes,
        "laplacian": laplacian,
        "phases": phases,
        "L6": L6,
        "H": H,
        "psi_init": psi_init,
        "psi_final": psi,
        "history": history,
    }
    
    if log_to_file:
        result["log_file"] = str(log_file)
    
    return result


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

def demo_prime_sixth_order_flow():
    """
    Demonstration of the prime sixth-order flow experiment.
    
    Shows:
    - Evolution of ψ under 6th-order gradient flow
    - Time series of ρ_IGBP and its components
    - Physical interpretation of the results
    """
    print("=" * 80)
    print("Prime Sixth-Order Flow Experiment")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  Prime ring: [2, 3, 5, 7]")
    print("  Operator: L₆ = Δ³ (6th-order elliptic)")
    print("  Dynamics: Gradient flow ∂_τ ψ = -H ψ where H = L₆²")
    print("  Observable: ρ_IGBP = ||∇S_rep||² / (|Ω| + ε)")
    print()
    print("-" * 80)
    print()
    
    # Run experiment with logging
    results = run_prime_sixth_order_flow(
        n_steps=200,
        dt=0.01,
        verbose_every=20,
        log_to_file=True,
    )
    
    print()
    print("-" * 80)
    print()
    print("Final Results:")
    print(f"  ψ_final = {results['psi_final']}")
    print(f"  Φ (holonomy) = {results['history'][-1]['phi']:.6e}")
    print(f"  |Ω| (Berry) = {results['history'][-1]['omega']:.6e}")
    print(f"  S_rep = {results['history'][-1]['S_rep']:.6f}")
    print(f"  ρ_IGBP = {results['history'][-1]['rho_igbp']:.6e}")
    print()
    
    if 'log_file' in results:
        print(f"Log saved to: {results['log_file']}")
        print()
    
    print("Physical Interpretation:")
    print("  - Φ ≈ 0: Ring phases cancel (ln(3/2)+ln(5/3)+ln(7/5)+ln(2/7) = 0)")
    print("  - |Ω| ≈ 0: Flat Berry curvature → high ρ_IGBP")
    print("  - High ρ_IGBP: 'Stiff/light-like' regime (entropy gradients dominate)")
    print()
    print("Next Steps:")
    print("  1. Swap in PrimeWilsonLoop for genuine geometric holonomy")
    print("  2. Compare ρ_IGBP on different prime rings (lepton vs quark plaquettes)")
    print("  3. Couple to ReebFlowDynamics for time-aligned evolution")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demo_prime_sixth_order_flow()
