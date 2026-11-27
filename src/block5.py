"""Block 5 runner (P5): cost-sensitive study, Analysis E/B, fairness, cross-lender."""
import argparse

from src.analysis import block5_cost, block5_cross_lender, block5_errors, block5_fairness
from src.utils.io import config


def main():
    parser = argparse.ArgumentParser(description="Run Block 5 (Person 5) analyses")
    parser.add_argument("--stage", choices=["cost", "errors", "fairness", "cross_lender", "all"], default="all")
    args = parser.parse_args()
    cfg = config()

    if args.stage in ("cost", "all"):
        block5_cost.run(cfg)
    if args.stage in ("errors", "all"):
        block5_errors.run(cfg)
    if args.stage in ("fairness", "all"):
        block5_fairness.run(cfg)
    if args.stage in ("cross_lender", "all"):
        block5_cross_lender.run(cfg)


if __name__ == "__main__":
    main()
