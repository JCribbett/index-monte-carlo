"""Command-line frontend.

Example:
    python -m indices_mc.cli --index sp500 --lookback 10 \
        --horizon 10 --target 8000 --iterations 20000 --show

    python -m indices_mc.cli --list-indices
"""

from __future__ import annotations

import argparse
import sys

from . import presets
from .analysis import convergence_curve, percentile_bands, target_probability
from .calibration import calibrate
from .engine import simulate
from .parameters import DriftSpec, Parameters

LIMITATIONS = (
    "This is conditional illustration, not prediction. The result holds only "
    "while the calibrated mu and sigma continue to hold; constant-parameter GBM "
    "does not capture regime shifts, fat tails, or volatility clustering. "
    "Educational use only - not financial advice. See readme.txt."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="indices_mc",
        description="Monte Carlo GBM simulation of a stock index (educational).",
    )
    p.add_argument("--index", default=None, metavar="KEY",
                   help="National index preset key (e.g. sp500, dax, nikkei225). "
                        "See --list-indices. Overrides --ticker.")
    p.add_argument("--ticker", default="^GSPC",
                   help="Index ticker for any yfinance symbol (default: ^GSPC). "
                        "Ignored if --index is given.")
    p.add_argument("--list-indices", action="store_true",
                   help="List the available national index presets and exit.")
    p.add_argument("--lookback", type=float, default=10.0,
                   help="Calibration lookback window in years (default: 10).")
    p.add_argument("--horizon", type=float, default=10.0,
                   help="Simulation horizon in years (default: 10).")
    p.add_argument("--target", type=float, default=None,
                   help="Target index level for the probability estimate.")
    p.add_argument("--iterations", type=int, default=10_000,
                   help="Number of simulated paths (default: 10,000).")
    p.add_argument("--steps-per-year", type=int, default=252,
                   help="Time steps per year (default: 252 = daily).")
    p.add_argument("--seed", type=int, default=12345,
                   help="RNG seed for reproducibility (default: 12345).")
    p.add_argument("--drift-low", type=float, default=None,
                   help="Triangular drift low bound (enables parameter uncertainty).")
    p.add_argument("--drift-high", type=float, default=None,
                   help="Triangular drift high bound (enables parameter uncertainty).")
    p.add_argument("--show", action="store_true",
                   help="Display the cone and convergence plots.")
    p.add_argument("--save", metavar="PREFIX", default=None,
                   help="Save plots to PREFIX_cone.png and PREFIX_convergence.png.")
    return p


def _print_index_list() -> None:
    print("Available national index presets:\n")
    print(f"  {'key':<14} {'ticker':<14} {'country':<18} name")
    print(f"  {'-'*14} {'-'*14} {'-'*18} {'-'*24}")
    for p in presets.all_presets():
        print(f"  {p.key:<14} {p.ticker:<14} {p.country:<18} {p.name}")
    print("\nUse: --index <key>   (or --ticker <symbol> for any other yfinance symbol)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_indices:
        _print_index_list()
        return 0

    if args.index is not None:
        try:
            preset = presets.get(args.index)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            print("Run --list-indices to see available keys.", file=sys.stderr)
            return 2
        ticker = preset.ticker
        label = f"{preset.name} ({preset.country}) [{ticker}]"
    else:
        ticker = args.ticker
        known = presets.by_ticker(ticker)
        label = f"{known.name} ({known.country}) [{ticker}]" if known else ticker

    print(f"Calibrating {label} over ~{args.lookback:g}y lookback ...")
    cal = calibrate(ticker, lookback_years=args.lookback)
    print("  " + cal.summary())

    if args.drift_low is not None and args.drift_high is not None:
        drift = DriftSpec.triangular(args.drift_low, cal.mu, args.drift_high)
        print(f"  Drift treated as Triangular({args.drift_low:.4f}, "
              f"{cal.mu:.4f}, {args.drift_high:.4f}) — propagating parameter uncertainty.")
    else:
        drift = DriftSpec.fixed(cal.mu)

    params = Parameters.from_calibration(
        cal,
        horizon_years=args.horizon,
        drift=drift,
        steps_per_year=args.steps_per_year,
        n_iterations=args.iterations,
        seed=args.seed,
    )

    print(f"Simulating {params.n_iterations:,} paths over {params.horizon_years:g}y "
          f"({params.n_steps:,} steps) ...")
    result = simulate(params)

    bands = percentile_bands(result)
    print("\nPercentile bands (probability cone):")
    print(f"  {'year':>5} " + " ".join(f"{f'P{p}':>12}" for p in bands.percentiles))
    for i, yr in enumerate(bands.years):
        row = " ".join(f"{bands.values[j, i]:>12,.0f}" for j in range(len(bands.percentiles)))
        print(f"  {int(yr):>5} {row}")

    if args.target is not None:
        prob = target_probability(result, args.target, args.horizon)
        curve = convergence_curve(result, args.target, args.horizon)
        print(f"\nP(level >= {args.target:,.0f} by year {args.horizon:g}) = "
              f"{prob:.4f}  ({prob * 100:.2f}%)")
        print(f"  Convergence: final estimate {curve.final:.4f} over "
              f"{params.n_iterations:,} iterations.")

    if args.show or args.save:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .plotting import plot_cone, plot_convergence

        fig1, ax1 = plt.subplots(figsize=(9, 5.5))
        plot_cone(bands, params.spot, ax=ax1, target=args.target)
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        if args.target is not None:
            plot_convergence(convergence_curve(result, args.target, args.horizon), ax=ax2)

        if args.save:
            fig1.savefig(f"{args.save}_cone.png", dpi=120, bbox_inches="tight")
            if args.target is not None:
                fig2.savefig(f"{args.save}_convergence.png", dpi=120, bbox_inches="tight")
            print(f"\nSaved plots to {args.save}_cone.png"
                  + (f" and {args.save}_convergence.png" if args.target is not None else ""))
        if args.show:
            plt.show()

    print("\n" + "-" * 72)
    print(LIMITATIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
