"""Classical tree diagnostics. The outer validation set is descriptive only."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.tree import plot_tree

from src.utils.io import artifact


def save_plot(fig, cfg, name):
    fig.tight_layout()
    fig.savefig(artifact(cfg, f"plots/block2_{name}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def leaf_rules(model, names):
    tree = model.tree_
    rules = []

    def walk(node, conditions):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            values = tree.value[node, 0]
            rules.append(dict(rule=" AND ".join(conditions) or "All loans",
                              train_loans=int(tree.n_node_samples[node]),
                              train_default_rate=float(values[1] / values.sum())))
            return
        feature = names[tree.feature[node]]
        group, label = feature.split("__", 1)
        for child, sign in [(left, "<="), (right, ">")]:
            condition = f"{label} {sign} {tree.threshold[node]:.8g}"
            if group == "missing":
                field = label.removeprefix("missingindicator_")
                condition = f"{field} is {'present' if child == left else 'missing'}"
            elif group in {"category", "title"}:
                condition = f"{label} indicator is {'off' if child == left else 'on'}"
            elif (child == left) == bool(tree.missing_go_to_left[node]):
                condition = f"({condition} or missing)"
            walk(child, [*conditions, condition])

    walk(0, [])
    return pd.DataFrame(rules).sort_values("train_default_rate", ascending=False)


def decision_diagnostics(pipeline, train, val, cfg):
    X, y = train
    V, vy = val
    prep, tree = pipeline.steps[0][1], pipeline.steps[1][1]
    x, v = prep.transform(X), prep.transform(V)
    rows = []
    for alpha in sorted({0., .00001, .0001, .001, .01, tree.ccp_alpha}):
        pruned = clone(tree).set_params(ccp_alpha=alpha).fit(x, y)
        rows.append(dict(ccp_alpha=alpha, leaves=pruned.get_n_leaves(),
                         val_auc=roc_auc_score(vy, pruned.predict_proba(v)[:, 1])))
    curve = pd.DataFrame(rows)
    curve.to_csv(artifact(cfg, "tables/block2_pruning.csv"), index=False)
    ax = curve.plot(x="ccp_alpha", y="val_auc", marker="o", ylabel="Validation ROC AUC", legend=False)
    ax.set_xscale("symlog", linthresh=.00001)
    ax.set_xlim(left=0)
    save_plot(ax.figure, cfg, "pruning")
    shallow = clone(tree).set_params(max_depth=3, ccp_alpha=0).fit(x, y)
    names = prep.get_feature_names_out().tolist()
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(shallow, feature_names=names, class_names=["Paid", "Default"], filled=True, ax=ax)
    save_plot(fig, cfg, "shallow_tree")
    rules = leaf_rules(shallow, names)
    rules.to_csv(artifact(cfg, "tables/block2_tree_rules.csv"), index=False)
    leaf = rules.iloc[0]
    artifact(cfg, "block2/tree_findings.md").write_text(
        f"The highest-default leaf in the depth-three tree contains {leaf.train_loans} Train loans. "
        f"Its observed default fraction is {leaf.train_default_rate:.1%}, compared with {y.mean():.1%} "
        f"across Train. An application reaches this leaf when every condition below holds.\n\n"
        f"{leaf.rule}\n\nThese are descriptive underwriting segments. Leaf frequencies are "
        "in-sample estimates.\n", encoding="utf-8")


def forest_diagnostics(pipeline, train, val, cfg):
    X, y = train
    V, vy = val
    prep, forest = pipeline.steps[0][1], pipeline.steps[1][1]
    x = prep.transform(X)
    growing = clone(forest).set_params(oob_score=True, warm_start=True)
    rows = []
    for count in sorted({16, 32, 64, forest.n_estimators}):
        growing.set_params(n_estimators=count).fit(x, y)
        votes = growing.oob_decision_function_
        covered = votes.sum(axis=1) > 0
        rows.append(dict(trees=count, covered=int(covered.sum()),
                         oob_auc=roc_auc_score(y[covered], votes[covered, 1])))
    curve = pd.DataFrame(rows)
    curve.to_csv(artifact(cfg, "tables/block2_oob.csv"), index=False)
    ax = curve.plot(x="trees", y="oob_auc", marker="o", ylabel="Out-of-bag ROC AUC", legend=False)
    save_plot(ax.figure, cfg, "oob")
    sample = V.sample(min(len(V), 1000), random_state=cfg["seed"])
    result = permutation_importance(forest, prep.transform(sample), vy.loc[sample.index],
                                    scoring="roc_auc", n_repeats=3, random_state=cfg["seed"])
    importance = pd.DataFrame(dict(feature=prep.get_feature_names_out(), impurity=forest.feature_importances_,
                                   permutation_auc_drop=result.importances_mean,
                                   permutation_sd=result.importances_std))
    importance.sort_values("permutation_auc_drop", ascending=False, inplace=True)
    importance.to_csv(artifact(cfg, "tables/block2_rf_importance.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, metric in zip(axes, ["impurity", "permutation_auc_drop"]):
        importance.nlargest(10, metric).sort_values(metric).plot.barh(x="feature", y=metric, ax=ax, legend=False)
        ax.set_xlabel(metric.replace("_", " "))
    save_plot(fig, cfg, "rf_importance")
