import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.cost_sensitive import curve
from src.utils.io import artifact


def plots(cfg, predictions, prefix):
    for kind in ["roc", "pr", "calibration", "ks", "distribution", "cost"]:
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for model, frame in predictions.items():
            y, p = frame.y_true.to_numpy(), frame.y_prob.to_numpy()
            if kind in ("roc", "ks"):
                fp, tp, t = roc_curve(y, p)
                if kind == "roc":
                    ax.plot(fp, tp, label=model)
                    ax.set(xlabel="False-positive rate", ylabel="True-positive rate")
                else:
                    ax.plot(t[1:], (tp-fp)[1:], label=model)
                    ax.set(xlabel="Default probability threshold", ylabel="TPR minus FPR")
            elif kind == "pr":
                precision, recall, _ = precision_recall_curve(y, p)
                ax.plot(recall, precision, label=model)
                ax.set(xlabel="Recall", ylabel="Precision")
            elif kind == "calibration":
                obs, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
                ax.plot(pred, obs, marker="o", label=model)
                ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate")
            elif kind == "distribution":
                for label in (0, 1):
                    ax.hist(p[y == label], bins=np.linspace(0, 1, 41), histtype="step", density=True,
                            label=f"{model} / {label}")
                ax.set(xlabel="Predicted default probability", ylabel="Density")
            else:
                for ratio in cfg["cost_ratios"]:
                    t, cost = curve(y, p, ratio)
                    ax.plot(t, cost, label=f"{model} / {ratio:g}:1")
                ax.set(xlabel="Rejection threshold", ylabel="Cost units per 1,000")
        if kind in ("roc", "calibration"):
            ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(artifact(cfg, f"plots/{prefix}_{kind}.png"))
        plt.close(fig)
