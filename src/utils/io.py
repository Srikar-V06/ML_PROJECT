import argparse
import hashlib
import json
from pathlib import Path

import yaml


def config(path="config/base.yaml"):
    return yaml.safe_load(Path(path).read_text())


def cli_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    return config(parser.parse_args().config)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def artifact(cfg, name):
    path = Path(cfg["paths"]["artifacts"]) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
