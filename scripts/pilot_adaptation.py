"""Dry run of the adaptation question on the pilot's accidental machine change.

Partway through pilot pass 2 every query slowed down (most likely Windows
energy saver; see results/pilot/pass2/README.md). The queries measured after
that change point are treated as a small "new machine": the source model is
trained on pass 1 labels and adapted with k pass-2 labels from the slowed
part.

Design, following the plan in FINAL_PROPOSAL_PLAN.md:
  - The slowed queries are split once per repeat, inside every template, into
    an adaptation pool (70%) and a test set (30%). Test query IDs are also
    removed from the source training data, so no test query is ever seen.
  - The pool is visited in one random arrival order per repeat, shared by all
    methods; the first k queries are the target labels.
  - Source model: the pilot's cost line + random forest on its residuals.
  - Methods: no adaptation; speed factor (source prediction plus the mean log
    error on the k queries); ridge correction (a ridge regression of the log
    error on the plan features, penalty chosen by leave-one-out, intercept not
    penalized, so a large penalty gives back the speed factor); target only
    (the source pipeline trained on the k queries alone). The oracle is target
    only trained on the whole pool.

It also describes the shift itself: how much of it the template explains
(one-way ANOVA on log(pass 2 / pass 1), before and after the change point)
and how much PostgreSQL's cost explains.

Writes results/pilot/adaptation/summary.json, results/pilot/adapt_numbers.tex
(LaTeX macros) and results/pilot/table_adapt.tex.

Usage: python scripts/pilot_adaptation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from compare_passes import change_point  # noqa: E402
from train_eval import ALPHAS, HYBRID, design, forest, qerror  # noqa: E402

P1 = HERE / "results/pilot"
P2 = HERE / "results/pilot/pass2"
OUT = HERE / "results/pilot/adaptation"
REPEATS = 20
KS = [5, 10, 25, 50]
TEST_FRAC = 0.3
METHODS = ["No adaptation", "Speed factor", "Ridge correction", "Target only"]
K_NAMES = {5: "Kfive", 10: "Kten", 25: "Ktwentyfive", 50: "Kfifty", "pool": "Kpool"}
M_NAMES = {"No adaptation": "None", "Speed factor": "Speed", "Ridge correction": "Ridge", "Target only": "Target"}


def fit_hybrid(train: pd.DataFrame, y: np.ndarray, seed: int):
    """The pilot's cost line plus a random forest fitted to its residuals."""
    line = LinearRegression().fit(design(train, "Cost-only"), y)
    rf = forest(seed).fit(design(train, HYBRID), y - line.predict(design(train, "Cost-only")))
    return lambda d: line.predict(design(d, "Cost-only")) + rf.predict(design(d, HYBRID))


def split_by_template(t: pd.DataFrame, rng: np.random.Generator) -> tuple[list, list]:
    pool, test = [], []
    for _, g in t.groupby("template", sort=True):
        ids = rng.permutation(g.query_id.to_numpy())
        n_test = 0 if len(ids) < 2 else max(1, round(TEST_FRAC * len(ids)))
        test.extend(ids[:n_test])
        pool.extend(ids[n_test:])
    return pool, test


def template_share(log_ratio: np.ndarray, template: np.ndarray) -> tuple[float, float]:
    """Share of the variance of log_ratio between templates, and the ANOVA p-value."""
    df = pd.DataFrame({"r": log_ratio, "t": template})
    total = ((df.r - df.r.mean()) ** 2).sum()
    between = (df.groupby("t").r.transform("mean") - df.r.mean()).pow(2).sum()
    groups = [g.r.to_numpy() for _, g in df.groupby("t") if len(g) > 1]
    return float(between / total), float(f_oneway(*groups).pvalue)


def main() -> None:
    a = pd.read_csv(P1 / "runtime_dataset.csv")
    b = pd.read_csv(P2 / "runtime_dataset.csv")
    a = a[a.status == "ok"]
    m = a.merge(b[["query_id", "runtime_ms", "order", "status"]], on="query_id", suffixes=("", "_2"))
    a = a.set_index("query_id", drop=False)
    m = m[m.status_2 == "ok"].sort_values("order_2").reset_index(drop=True)
    log_ratio = np.log(m.runtime_ms_2 / m.runtime_ms).to_numpy()
    cut = change_point(log_ratio)
    before, after = m.iloc[:cut], m.iloc[cut:]

    # The shift itself.
    share_after, p_after = template_share(log_ratio[cut:], after.template.to_numpy())
    share_before, p_before = template_share(log_ratio[:cut], before.template.to_numpy())
    x_cost = np.log1p(after.total_cost.to_numpy()).reshape(-1, 1)
    cost_r2 = float(LinearRegression().fit(x_cost, log_ratio[cut:]).score(x_cost, log_ratio[cut:]))
    per_t = pd.Series(log_ratio[cut:]).groupby(after.template.to_numpy()).agg(["mean", "size"])
    per_t = per_t[per_t["size"] >= 3]
    shift = {"n_after": int(len(after)), "n_templates_after": int(after.template.nunique()),
             "template_share_after": share_after, "anova_p_after": p_after,
             "template_share_before": share_before, "anova_p_before": p_before,
             "cost_r2_after": cost_r2,
             "template_slowdown_min": float(np.exp(per_t["mean"].min())),
             "template_slowdown_max": float(np.exp(per_t["mean"].max())),
             "templates_with_3_or_more": int(len(per_t))}

    # Adaptation dry run.
    target = after.set_index("query_id", drop=False)
    y_target = np.log(target.runtime_ms_2)
    rows = []
    pool_sizes = []
    for r in range(REPEATS):
        rng = np.random.default_rng(r)
        pool, test = split_by_template(target, rng)
        pool_sizes.append(len(pool))
        src = a.drop(index=test)
        f_src = fit_hybrid(src, np.log(src.runtime_ms.to_numpy()), seed=r)
        te = target.loc[test]
        y_te = te.runtime_ms_2.to_numpy()
        base_te = f_src(te)
        order = rng.permutation(pool)

        def score(pred_log):
            return float(np.median(qerror(y_te, np.exp(pred_log))))

        for k in KS + ["pool"]:
            ids = order if k == "pool" else order[:k]
            tr = target.loc[ids]
            e = y_target.loc[ids].to_numpy() - f_src(tr)
            ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(design(tr, "Ridge"), e)
            preds = {
                "No adaptation": base_te,
                "Speed factor": base_te + e.mean(),
                "Ridge correction": base_te + ridge.predict(design(te, "Ridge")),
                "Target only": fit_hybrid(tr, y_target.loc[ids].to_numpy(), seed=r)(te),
            }
            for name, p in preds.items():
                rows.append({"repeat": r, "k": str(k), "method": name, "median_q": score(p),
                             "alpha": float(ridge[-1].alpha_)})

    res = pd.DataFrame(rows)
    tab = res.groupby(["method", "k"]).median_q.agg(["mean", "std"]).reset_index()
    alphas = res[res.method == "Ridge correction"].groupby("k").alpha.median().to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "runs.csv", index=False)
    summary = {"shift": shift, "repeats": REPEATS, "test_frac": TEST_FRAC,
               "pool_size_median": float(np.median(pool_sizes)),
               "median_q": {f"{row.method} | k={row.k}": {"mean": row["mean"], "std": row["std"]}
                            for _, row in tab.iterrows()},
               "ridge_alpha_median_by_k": alphas}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    def cell(method, k):
        v = tab[(tab.method == method) & (tab.k == str(k))]
        return f"{v['mean'].iloc[0]:.2f}"

    macros = {
        "ShiftN": shift["n_after"],
        "ShiftTemplates": shift["n_templates_after"],
        "ShiftTemplateShare": f"{100 * share_after:.0f}",
        "ShiftTemplateShareBefore": f"{100 * share_before:.0f}",
        "ShiftAnovaP": f"{p_after:.0e}".replace("e-", "\\times 10^{-") + "}" if p_after < 1e-3 else f"{p_after:.3f}",
        "ShiftCostRsq": f"{cost_r2:.3f}",
        "ShiftSlowMin": f"{shift['template_slowdown_min']:.2f}",
        "ShiftSlowMax": f"{shift['template_slowdown_max']:.2f}",
        "AdRepeats": REPEATS,
        "AdPoolSize": f"{np.median(pool_sizes):.0f}",
    }
    for method in METHODS:
        for k in KS + ["pool"]:
            macros[f"Ad{M_NAMES[method]}{K_NAMES[k]}"] = cell(method, k)
    (HERE / "results/pilot/adapt_numbers.tex").write_text(
        "% Generated by scripts/pilot_adaptation.py. Do not edit by hand.\n" +
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8", newline="\n")

    lines = ["% Generated by scripts/pilot_adaptation.py. Do not edit by hand.",
             "\\begin{tabular}{@{}l" + "c" * (len(KS) + 1) + "@{}}", "\\toprule",
             "Method & " + " & ".join(f"$k={k}$" for k in KS) + " & whole pool \\\\", "\\midrule"]
    for method in METHODS:
        lines.append(method + " & " + " & ".join(cell(method, k) for k in KS + ["pool"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (HERE / "results/pilot/table_adapt.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
