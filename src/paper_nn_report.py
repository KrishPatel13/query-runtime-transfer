"""Macros and a table from the summary of src/paper_nn.py (the neural model of H6).

Reads <run>/summary.json and writes <run>/nn_numbers.tex, with one macro per
workload, selection, method and k (geometric means over the run's pairs),
the H6 counts per fine-tuning mode and k, and the model's settings, and
<run>/table_nn.tex, the errors with round-robin selection at k = 10, 25 and
100 (kH, 33 on JOB) for both source models.

Macro names: <prefix><Workload><Sel><Method>K<k>, with Sel Rand or Rr,
Method one of HybSpeed, HybRidge, HybTpl, NnNone, NnSpeed, NnRidge, NnTpl,
NnOracle, NnFull, NnHead, and k spelled digit by digit (K25 is KTwoFive), as
src/paper_report.py spells them; H6 counts are
<prefix><Workload>HSix<Mode>K<k><Below|Point|Pairs|Gm|MaxHi>.

Usage: python src/paper_nn_report.py --run results/paper/nn_dev --prefix NNDev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
WORKLOAD = {"tpch": "Tpch", "tpcds": "Tpcds", "tpch10": "TpchTen", "job": "Job"}
SELECTION = {"random": "Rand", "round-robin": "Rr"}
METHOD = {"hybrid:speed": "HybSpeed", "hybrid:ridge": "HybRidge", "hybrid:lowdim+template": "HybTpl",
          "hybrid:none": "HybNone", "nn:none": "NnNone", "nn:speed": "NnSpeed", "nn:ridge": "NnRidge",
          "nn:lowdim+template": "NnTpl", "nn:oracle": "NnOracle", "nn:full-ft": "NnFull",
          "nn:head-ft": "NnHead"}
DISPLAY = {"tpch": "TPC-H", "tpcds": "TPC-DS", "tpch10": "TPC-H SF10", "job": "JOB"}
TABLE_ROWS = [("Cost line plus forest", None), ("speed factor", "hybrid:speed"),
              ("template correction", "hybrid:lowdim+template"),
              ("Neural (DACE-style)", None), ("no adaptation", "nn:none"), ("speed factor", "nn:speed"),
              ("template correction", "nn:lowdim+template"), ("fine-tuning, all weights", "nn:full-ft"),
              ("fine-tuning, head only", "nn:head-ft"), ("oracle (whole target pool)", "nn:oracle")]
TABLE_KS = ["10", "25", "100"]


def table_ks(cfg: dict, w: str) -> list[str]:
    """TABLE_KS with the workload's kH in place of 100 (paper_nn's by_workload: 33 on JOB)."""
    kh = str(cfg.get("by_workload", {}).get(w, {}).get("kh", 100))
    return TABLE_KS[:-1] + [kh]


def kname(k: str) -> str:
    return "K" + "".join(DIGITS[int(d)] for d in k)


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def fig_nn(s: dict, out: Path) -> None:
    """Error against k with round-robin selection, one panel per workload: the speed factor and the
    template correction on the cost line plus forest, the template correction on the neural model,
    and the neural model fine-tuned on the same queries (fig_nn.pdf and .png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper_report import COLORS, INK, log_ticks, save, style

    fine = "#eb6834"
    series = [("hybrid:speed", "Speed factor", COLORS["speed"], "-"),
              ("hybrid:lowdim+template", "Template correction", COLORS["lowdim+template"], "-"),
              ("nn:lowdim+template", "Template correction, neural", COLORS["lowdim+template"], "--"),
              ("nn:full-ft", "Fine-tuning, all weights", fine, "-"),
              ("nn:head-ft", "Fine-tuning, head only", fine, ":")]
    ws = s["config"]["workloads"]
    fig, axes = plt.subplots(1, len(ws), figsize=(1.75 * len(ws), 1.85), squeeze=False, sharey=True)
    grids = []                    # each panel's own k grid (the second round's workloads have different grids)
    for ax, w in zip(axes.flat, ws):
        g = s["gm_over_pairs"][w]["round-robin"]
        ks = sorted(int(k) for k in g["nn:full-ft"])
        grids.append(ks)
        for m, label, color, ls in series:
            ax.plot(ks, [g[m][str(k)] for k in ks], ls, color=color, lw=1.1, marker="o", ms=2, label=label)
        ax.axhline(g["nn:oracle"][str(ks[0])], color=INK, lw=0.7, ls=(0, (1, 2)), label="Oracle, neural")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(DISPLAY[w], fontsize=7.5)
        style(ax)
    lo, hi = axes.flat[0].get_ylim()
    for ax, ks in zip(axes.flat, grids):
        log_ticks(ax, lo, hi, ticks=(1, 1.1, 1.2, 1.3, 1.4, 1.5))
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("target queries $k$", fontsize=7)
    axes.flat[0].set_ylabel("error", fontsize=7.5)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.24, 1, 1), w_pad=0.4)
    save(fig, out, "fig_nn")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    s = json.loads((a.run / "summary.json").read_text(encoding="utf-8"))
    p, cfg = a.prefix, s["config"]
    out = ["% Generated by src/paper_nn_report.py. Do not edit by hand.\n"]
    out.append(macro(f"{p}Epochs", cfg["epochs"]))
    out.append(macro(f"{p}Lr", f"{cfg['lr']:g}"))
    out.append(macro(f"{p}Hidden", cfg["hidden"]))
    out.append(macro(f"{p}Batch", cfg["batch"]))
    out.append(macro(f"{p}Dropout", f"{cfg['dropout']:g}"))
    fits = list(s["source_seconds"].values())
    out.append(macro(f"{p}SourceSeconds", f"{sum(fits) / len(fits):.0f}"))
    for w in cfg["workloads"]:
        W = WORKLOAD[w]
        params = sorted({v for k, v in s["parameters"].items() if k.startswith(w + "|")})
        out.append(macro(f"{p}{W}Params", f"{params[0]:,}"))
        for sel, methods in s["gm_over_pairs"][w].items():
            for m, by_k in methods.items():
                if m not in METHOD:
                    continue
                for k, v in by_k.items():
                    out.append(macro(f"{p}{W}{SELECTION[sel]}{METHOD[m]}{kname(k)}", fmt(v)))
        for k, modes in s["h6"][w].items():
            for mode, h in modes.items():
                stem = f"{p}{W}HSix{mode.capitalize()}{kname(k)}"
                out.append(macro(stem + "Below", h["below_1"]))
                out.append(macro(stem + "Point", h["point_below_1"]))
                out.append(macro(stem + "Pairs", h["pairs"]))
                out.append(macro(stem + "Gm", fmt(h["gm"], 3)))
                out.append(macro(stem + "MaxHi", fmt(h["max_hi"], 3)))
    (a.run / "nn_numbers.tex").write_text("".join(out), encoding="utf-8")

    ws = cfg["workloads"]
    cols = "l" + "r" * (len(TABLE_KS) * len(ws))
    lines = ["% Generated by src/paper_nn_report.py. Do not edit by hand.\n",
             f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\n\\toprule\n",
             "& " + " & ".join(f"\\multicolumn{{{len(TABLE_KS)}}}{{c}}{{{DISPLAY[w]}}}" for w in ws)
             + " \\\\\n",
             "".join(f"\\cmidrule(lr){{{2 + i * len(TABLE_KS)}-{1 + (i + 1) * len(TABLE_KS)}}}"
                     for i in range(len(ws))) + "\n",
             "$k$ & " + " & ".join(k for w in ws for k in table_ks(cfg, w)) + " \\\\\n\\midrule\n"]
    for label, m in TABLE_ROWS:
        if m is None:
            lines.append(f"\\multicolumn{{{1 + len(TABLE_KS) * len(ws)}}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\\n")
            continue
        cells = [fmt(s["gm_over_pairs"][w]["round-robin"][m][k]) for w in ws for k in table_ks(cfg, w)]
        lines.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (a.run / "table_nn.tex").write_text("".join(lines), encoding="utf-8")
    fig_nn(s, a.run)
    print(f"wrote {a.run / 'nn_numbers.tex'}, {a.run / 'table_nn.tex'} and {a.run / 'fig_nn.pdf'}")


if __name__ == "__main__":
    main()
