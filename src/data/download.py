"""Resolve the supplied loan tape locally and fetch missing companion inputs."""
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

from src.utils.io import cli_config


def ensure(cfg):
    loan = Path(cfg["paths"]["lending_club"])
    if not loan.is_file():
        raise FileNotFoundError(loan.resolve())
    for key in ("dictionary", "prosper"):
        dest = Path(cfg["paths"][key])
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(".tmp")
            urlretrieve(cfg["sources"][key], temp)
            temp.replace(dest)
    if not zipfile.is_zipfile(cfg["paths"]["dictionary"]):
        raise ValueError("Invalid dictionary workbook")


if __name__ == "__main__":
    ensure(cli_config())
