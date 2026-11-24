from datetime import datetime, timezone
import hashlib
import json
import subprocess

from src.utils.io import artifact


def log(cfg, step, rows, columns, **details):
    sha = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
    event = dict(step=step, rows=rows, columns=columns, git_sha=sha.stdout.strip() or None,
                 timestamp=datetime.now(timezone.utc).isoformat(),
                 config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(), **details)
    with artifact(cfg, "run.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
