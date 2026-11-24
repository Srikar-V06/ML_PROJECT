"""Run the real Block 1 workflow from the repository root."""
import pandas as pd

from src.audit.leakage import run as audit
from src.data.download import ensure
from src.data.prep_lending_club import run as prepare_lending
from src.data.prep_prosper import run as prepare_prosper
from src.data.splits import freeze
from src.evaluation.comparison import run as compare
from src.report import run as report
from src.train import run as train
from src.utils.io import cli_config


def run(cfg):
    ensure(cfg)
    prepare_lending(cfg)
    freeze(pd.read_parquet(cfg["paths"]["sample"], columns=["id", "issue_d", "target"]), cfg)
    audit(cfg)
    for model in ("constant", "grade_only", "logreg_baseline", "woe_scorecard"):
        train(model, cfg)
    compare(cfg)
    prepare_prosper(cfg)
    report(cfg)


if __name__ == "__main__":
    run(cli_config())
