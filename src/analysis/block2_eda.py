"""Training-cohort figures and captions computed from the values displayed."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from src.analysis.block2_trees import save_plot
from src.features.block2 import build_preprocessor
from src.utils.io import artifact


def eda(X, y, dates, flags, cfg):
    notes = []
    frame = X.assign(target=y)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, ax in zip(["grade", "term"], axes):
        rates = frame.groupby(key).target.mean()
        rates.plot.bar(ax=ax, rot=0, ylabel="Default fraction")
        notes.append(f"Highest observed Train default fraction by {key} was "
                     f"{rates.max():.1%} for {rates.idxmax()}.")
    save_plot(fig, cfg, "segments")

    cols = ["loan_amnt", "annual_inc", "fico_range_low", "dti"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, ax in zip(cols, axes.flat):
        values = X[col].dropna()
        lo, hi = values.quantile([.01, .99])
        ax.hist(values.clip(lo, hi), bins=35)
        ax.set(xlabel=col, ylabel="Loans")
    notes.append(f"Median Train loan amount was {X.loan_amnt.median():,.0f} and median annual "
                 f"income was {X.annual_inc.median():,.0f}. Histogram tails are clipped at "
                 "the 1st and 99th percentiles for display.")
    save_plot(fig, cfg, "distributions")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for col, ax in zip(["fico_range_low", "dti"], axes):
        deciles = pd.qcut(X[col], 10, labels=False, duplicates="drop")
        rates = y.groupby(deciles).mean()
        ax.plot(rates.index + 1, rates, marker="o")
        ax.set(xlabel=f"{col} quantile bin (low to high)", ylabel="Default fraction")
        if len(rates):
            notes.append(f"From the lowest to highest observed {col} bin, default frequency "
                         f"changed from {rates.iloc[0]:.1%} to {rates.iloc[-1]:.1%}.")
    save_plot(fig, cfg, "deciles")

    corr = X[cols + ["int_rate", "installment", "revol_util", "open_acc"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6))
    heat = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(heat, ax=ax, label="Spearman correlation")
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    if len(pairs):
        a, b = pairs.abs().idxmax()
        notes.append(f"The strongest absolute rank correlation displayed was between {a} and {b} "
                     f"(rho {pairs.loc[(a, b)]:+.2f}).")
    save_plot(fig, cfg, "correlation")

    quarterly = frame.groupby(dates.dt.to_period("Q")).agg(default_rate=("target", "mean"),
                                                           median_dti=("dti", "median"))
    axes = quarterly.plot(subplots=True, figsize=(9, 6), legend=False)
    axes[0].set_ylabel("Default fraction")
    axes[1].set_ylabel("Median DTI")
    notes.append(f"From {quarterly.index[0]} to {quarterly.index[-1]}, the Train default fraction "
                 f"moved from {quarterly.default_rate.iloc[0]:.1%} to {quarterly.default_rate.iloc[-1]:.1%} "
                 f"and median DTI from {quarterly.median_dti.iloc[0]:.2f} to {quarterly.median_dti.iloc[-1]:.2f}.")
    save_plot(axes[0].figure, cfg, "drift")

    sample = X.sample(min(1500, len(X)), random_state=cfg["seed"])
    matrix = build_preprocessor("linear", flags).fit_transform(sample)
    labels = y.loc[sample.index].to_numpy()
    views = {"PCA": PCA(n_components=2, random_state=cfg["seed"]),
             "t-SNE": TSNE(perplexity=min(30, len(sample) - 1), init="pca", random_state=cfg["seed"])}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for (name, reducer), ax in zip(views.items(), axes):
        coords = reducer.fit_transform(matrix)
        for code, label in [(0, "Paid"), (1, "Default")]:
            ax.scatter(*coords[labels == code].T, label=label, alpha=.45, s=9)
        ax.set(title=name, xlabel="Dimension 1", ylabel="Dimension 2")
        ax.legend()
        neighbors = NearestNeighbors(n_neighbors=1).fit(coords).kneighbors(return_distance=False)[:, 0]
        agreement = (labels[neighbors] == labels).mean()
        chance = labels.mean()**2 + (1 - labels.mean())**2
        notes.append(f"In the {name} view of {len(sample):,} sampled Train loans, {agreement:.1%} had a "
                     f"nearest neighbor of the same class (class-frequency reference {chance:.1%}).")
    save_plot(fig, cfg, "class_separation")
    artifact(cfg, "block2/eda_findings.md").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
