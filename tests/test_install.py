"""Test package installation"""
try:
    from light_theory_realm import CliffordEngine, CliffordQGT, __version__
    print("✓ Core modules imported successfully")
    print(f"✓ CliffordEngine: {CliffordEngine}")
    print(f"✓ CliffordQGT: {CliffordQGT}")
    
    # Test basic functionality
    engine = CliffordEngine(seed=42)
    print(f"✓ Engine initialized: {engine}")
    
    print(f"✓ Package version: {__version__}")
    
    print("\n✓ Package installation successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
