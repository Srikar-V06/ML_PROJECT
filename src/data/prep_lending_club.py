"""Two streaming reads. The first establishes the population and the second retrieves sampled IDs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.audit.schema import classify
from src.utils.io import artifact, cli_config, digest, write_json
from src.utils.logging import log

CONT = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low"]
CAT = ["grade", "term", "purpose", "home_ownership"]
KEYS = ["id", "member_id", "issue_d", "loan_status"] + CONT + CAT


def parse(frame):
    for col in ("int_rate", "revol_util"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col].astype("string").str.rstrip("%"), errors="raise")
    frame["term"] = frame.term.astype("string").str.extract(r"(\d+)", expand=False).astype("Int16")
    for col in ("issue_d", "earliest_cr_line", "sec_app_earliest_cr_line"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col], format="%b-%Y", errors="raise")
    if "emp_length" in frame:
        years = frame.emp_length.replace("< 1 year", "0")
        frame["emp_length"] = years.str.extract(r"(\d+)", expand=False).astype("Float64")
    frame["id"] = pd.to_numeric(frame.id, errors="raise").astype("int64")
    return frame


def allocate(counts, n):
    ideal = counts / counts.sum() * n
    quota = np.floor(ideal).astype(int)
    order = (ideal - quota).sort_values(ascending=False, kind="stable").index
    quota.loc[order[:n - quota.sum()]] += 1
    return quota


def stratify(pool, n, seed):
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} mature loans for sample of {n}")
    strata = pool[["quarter", "grade", "target"]].copy()
    sizes = strata.groupby(list(strata), dropna=False).target.transform("size")
    rare = sizes < 2
    # Pool rare grades with the largest grade of the same quarter and outcome.
    common = strata.groupby(["quarter", "target"]).grade.transform(lambda g: g.mode().iloc[0])
    strata.loc[rare, "grade"] = common[rare]
    strata["year"] = pool.issue_d.dt.year
    yr = strata.groupby("year").size()
    yearly = allocate(yr, n)
    # Early vintages can be tiny. Give each the nearest feasible integer
    # denominator that represents its population rate within the agreed tolerance.
    rates = pool.groupby(pool.issue_d.dt.year).target.mean()
    for year, rate in rates.items():
        count = yearly[year]
        if abs(round(count * rate) / count - rate) >= 0.0015:
            candidates = sorted(range(max(1, count - 1000), min(yr[year], count + 1000) + 1),
                                key=lambda k: (abs(k - count), k))
            yearly[year] = next(k for k in candidates if abs(round(k * rate) / k - rate) < 0.0015)
    yearly.loc[yr.idxmax()] += n - yearly.sum()
    rng = np.random.default_rng(seed)
    selected = []
    for year, group in strata.groupby("year"):
        outcomes = allocate(group.groupby("target").size(), yearly[year])
        for target, part in group.groupby("target"):
            groups = part.groupby(["quarter", "grade"], dropna=False)
            quota = allocate(groups.size(), outcomes[target])
            for key, rows in groups:
                selected.extend(rng.choice(rows.index, quota[key], replace=False))
    return pool.loc[selected], int(rare.sum())


def representation(cfg, pool, sample):
    rows = []
    def rate(label, a, b):
        rows.append(dict(feature=label, measure="default_rate", population=a, sample=b, difference=abs(a-b)))
    rate("overall", pool.target.mean(), sample.target.mean())
    for year, group in pool.groupby(pool.issue_d.dt.year):
        rate(str(year), group.target.mean(), sample.loc[sample.issue_d.dt.year == year, "target"].mean())
    for col in CONT + CAT:
        a, b = pool[col], sample[col]
        ks = np.nan
        if col in CONT:
            ks = ks_2samp(a.dropna(), b.dropna()).statistic
            edges = np.unique(a.dropna().quantile(np.linspace(0, 1, 11)).to_numpy(dtype=float))
            bins = np.r_[-np.inf, edges[1:-1], np.inf]
            a, b = (pd.cut(s.astype(float), bins).astype("string") for s in (a, b))
        p, q = (s.astype("string").fillna("<missing>").value_counts(normalize=True) for s in (a, b))
        p, q = p.align(q, fill_value=0)
        psi = ((p-q) * np.log(p.clip(lower=1e-8)/q.clip(lower=1e-8))).sum()
        rows.append(dict(feature=col, measure="distribution", ks=ks, psi=psi, difference=(p-q).abs().max()))
    table = pd.DataFrame(rows)
    table.to_csv(artifact(cfg, "sampling_representativeness.csv"), index=False)
    table.dropna(subset=["psi"]).plot.barh(x="feature", y="psi", legend=False, xlabel="Population stability index")
    plt.tight_layout()
    plt.savefig(artifact(cfg, "plots/sampling_representativeness.png"))
    plt.close()
    assert table.psi.max() < .01
    assert table.query("measure == 'default_rate'").difference.max() < .0015


def run(cfg):
    source = cfg["paths"]["lending_club"]
    cols = pd.read_csv(source, nrows=0).columns
    clean = [c for c in cols if classify(c)[0] == "origination_time"]
    assert len(clean) >= 50
    frames = []
    footers = 0
    for chunk in pd.read_csv(source, usecols=KEYS, dtype={"id": "string", "member_id": "string"},
                             chunksize=60000, low_memory=False):
        footer = chunk.id.str.startswith(("Total amount funded", "Loans that do not meet the credit policy"),
                                         na=False) & chunk.loan_status.isna() & chunk.issue_d.isna()
        footers += int(footer.sum())
        frames.append(parse(chunk.loc[~footer].copy()))
    full = pd.concat(frames, ignore_index=True)
    del frames
    assert full.id.is_unique and full.issue_d.notna().all() and full.term.isin([36, 60]).all()
    raw = len(full)
    statuses = full.loan_status.value_counts(dropna=False).to_dict()
    full["quarter"] = full.issue_d.dt.to_period("Q").astype(str)
    full["target"] = full.loan_status.map(cfg["target"])
    # Legacy policy loans with a terminal outcome are resolved for censoring only.
    resolved = full.loan_status.isin(cfg["target"]) | full.loan_status.str.match(
        r"Does not meet the credit policy\. Status:(Fully Paid|Charged Off)$", na=False)
    vintage = full.assign(resolved=resolved).groupby(["quarter", "term"]).resolved.agg(["size", "mean"])
    vintage.to_csv(artifact(cfg, "tables/vintage_resolution.csv"))
    curves = vintage["mean"].unstack()
    curves.plot(figsize=(10, 4), ylabel="Fraction resolved", xlabel="Issue quarter")
    plt.axhline(cfg["maturity_threshold"], linestyle="--", color="grey")
    plt.tight_layout()
    plt.savefig(artifact(cfg, "plots/censoring_by_vintage.png"))
    plt.close()
    mature = curves[36].dropna().ge(cfg["maturity_threshold"]).cummin()
    cutoff = mature[mature].index[-1]
    known = full.member_id.notna()
    repeated = full.loc[known].sort_values(["issue_d", "id"]).member_id.duplicated()
    pool = full.drop(index=repeated[repeated].index)
    terminal = pool.target.notna()
    immature = int((terminal & pool.quarter.gt(cutoff)).sum())
    pool = pool.loc[terminal & pool.quarter.le(cutoff)].copy()
    pool["target"] = pool.target.astype("int8")
    sample, rare = stratify(pool, cfg["sample_size"], cfg["seed"])
    representation(cfg, pool, sample)
    stats = dict(raw_rows=raw, csv_footer_records=footers, member_ids_present=int(known.sum()),
                 duplicate_members_removed=int(repeated.sum()), status_counts=statuses, cutoff=cutoff,
                 immature_terminal_removed=immature, mature_rows=len(pool), sampled_rows=len(sample),
                 rare_stratum_rows=rare, source_sha256=digest(source), raw_columns=len(cols))
    write_json(artifact(cfg, "tables/population.json"), stats)
    ids = set(sample.id.astype(str))
    del full, pool, sample
    selected = []
    for block in pd.read_csv(source, dtype="string", chunksize=20000):
        subset = block.loc[block.id.isin(ids)]
        if len(subset):
            selected.append(subset.copy())
    data = parse(pd.concat(selected, ignore_index=True))
    del selected
    for c in data.select_dtypes("string"):
        if c in ("id", "member_id", "zip_code", "emp_title", "title", "desc"):
            continue
        try:
            data[c] = pd.to_numeric(data[c])
        except ValueError:
            pass
    data = data.sort_values(["issue_d", "id"]).reset_index(drop=True)
    assert len(data) == len(ids) and data.id.is_unique
    data["target"] = data.loan_status.map(cfg["target"]).astype("int8")
    for path in ("sample", "diagnostics"):
        Path(cfg["paths"][path]).parent.mkdir(parents=True, exist_ok=True)
    data[["id", "issue_d", "target", *clean]].to_parquet(cfg["paths"]["sample"], index=False)
    data[["id", *[c for c in cols if c not in clean and c not in ("id", "issue_d")]]].to_parquet(
        cfg["paths"]["diagnostics"], index=False)
    log(cfg, "lending_club", len(data), len(clean), **stats)


if __name__ == "__main__":
    run(cli_config())
