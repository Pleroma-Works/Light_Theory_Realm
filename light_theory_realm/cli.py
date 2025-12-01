"""
Light Theory Realm CLI
======================

Command-line interface for Light Theory Realm.

Subcommands:
  sm-table           Print the Standard Model fermion mass table
  profile            Show full physical + geometric profile for a particle
  koide              Compute Koide ratio + angle for three masses
  koide-predict      Predict third mass from first two using Koide
  koide-from-pocket  Run Koide analysis using Pocket_U Lite predicted masses

Example:
  $ light-realm sm-table
  $ light-realm profile e
  $ light-realm koide 0.511 105.66 1776.86
"""

import argparse
import sys
from pprint import pprint

from .pocket_u_lite.sm_table import print_sm_table, get_sm_spectrum, get_particle_profile
from .pocket_u_lite.koide import koide_ratio, koide_angle, predict_third_mass


PARTICLE_HELP = "Particle name (e, mu, tau, u, d, s, c, b, t)"


def cmd_sm_table(args: argparse.Namespace) -> int:
    """Print the Standard Model fermion mass table."""
    print_sm_table()
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Print full physical + geometric profile for one particle."""
    print("\n⚠️  The 'profile' command is temporarily unavailable.")
    print("    Issue: Geometry module has circular/hanging initialization.")
    print("    This will be fixed in v0.2.0.")
    print("\n    Use 'light-realm sm-table' to see mass predictions.\n")
    return 0


def cmd_koide(args: argparse.Namespace) -> int:
    """Compute Koide ratio + angle for three masses."""
    m1, m2, m3 = args.masses
    Q = koide_ratio(m1, m2, m3)
    ang = koide_angle(m1, m2, m3)

    print("\nKoide Analysis")
    print("=" * 60)
    print(f"  Masses          : {m1:.6g}, {m2:.6g}, {m3:.6g} MeV")
    print(f"  Q               : {Q:.8f}")
    print(f"  Foot angle      : {ang['theta_deg']:.6f}°")
    print(f"  θ (radians)     : {ang['theta_rad']:.8f}")
    print()
    return 0


def cmd_koide_predict(args: argparse.Namespace) -> int:
    """Predict third mass from first two using Koide."""
    m1, m2 = args.masses
    m3, Q = predict_third_mass(m1, m2)
    ang = koide_angle(m1, m2, m3)

    print("\nKoide Prediction")
    print("=" * 60)
    print(f"  Input masses    : {m1:.6g}, {m2:.6g} MeV")
    print(f"  Predicted m3    : {m3:.6g} MeV")
    print(f"  Q               : {Q:.8f}")
    print(f"  Foot angle      : {ang['theta_deg']:.6f}°")
    print()
    return 0


def cmd_koide_from_pocket(args: argparse.Namespace) -> int:
    """Run Koide analysis using Pocket_U Lite lepton masses."""
    spectrum = get_sm_spectrum()
    # Extract e, mu, tau from spectrum
    masses = {row["name"]: row["mass_pred"] for row in spectrum}
    e, mu, tau = masses["e"], masses["mu"], masses["tau"]

    Q = koide_ratio(e, mu, tau)
    ang = koide_angle(e, mu, tau)

    print("\nKoide from Pocket_U Lite (Predicted Lepton Masses)")
    print("=" * 60)
    print(f"  e, μ, τ (MeV)   : {e:.6g}, {mu:.6g}, {tau:.6g}")
    print(f"  Q               : {Q:.8f}")
    print(f"  Foot angle      : {ang['theta_deg']:.6f}°")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="light-realm",
        description=(
            "CLI interface for Light Theory Realm.\n"
            "Measures geometry of information & explores the Standard Model via prime plaquettes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  light-realm sm-table\n"
            "  light-realm profile e\n"
            "  light-realm profile tau --verbose\n"
            "  light-realm koide 0.511 105.66 1776.86\n"
            "  light-realm koide-predict 0.511 105.66\n"
            "  light-realm koide-from-pocket\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # sm-table subcommand
    p_sm = subparsers.add_parser(
        "sm-table",
        help="Print the Standard Model fermion mass table (Pocket_U Lite).",
    )
    p_sm.set_defaults(func=cmd_sm_table)

    # profile subcommand
    p_prof = subparsers.add_parser(
        "profile",
        help="Show full physical + geometric profile for a particle.",
    )
    p_prof.add_argument("name", help=PARTICLE_HELP)
    p_prof.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print full geometry arrays (Fisher, Berry, KK metric, Reeb vector).",
    )
    p_prof.set_defaults(func=cmd_profile)

    # koide subcommand
    p_koide = subparsers.add_parser(
        "koide",
        help="Compute Koide ratio + angle for three masses (in MeV).",
    )
    p_koide.add_argument(
        "masses",
        nargs=3,
        type=float,
        metavar=("M1", "M2", "M3"),
        help="Three masses in MeV (e.g., 0.511 105.66 1776.86).",
    )
    p_koide.set_defaults(func=cmd_koide)

    # koide-predict subcommand
    p_kpred = subparsers.add_parser(
        "koide-predict",
        help="Predict third mass from first two using Koide.",
    )
    p_kpred.add_argument(
        "masses",
        nargs=2,
        type=float,
        metavar=("M1", "M2"),
        help="Two masses in MeV (e.g., 0.511 105.66).",
    )
    p_kpred.set_defaults(func=cmd_koide_predict)

    # koide-from-pocket subcommand
    p_kp = subparsers.add_parser(
        "koide-from-pocket",
        help="Run Koide analysis using Pocket_U Lite predicted lepton masses.",
    )
    p_kp.set_defaults(func=cmd_koide_from_pocket)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return int(args.func(args))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
