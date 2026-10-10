"""Tests for the neural plan model (src/nn_model.py): plan parsing, the tree attention mask and
determinism. The networks are tiny and train for a few epochs, so the file runs in seconds."""
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# PyTorch is only in the environment of requirements-nn.txt; without it these tests are skipped.
torch = pytest.importorskip("torch")

import nn_model as nn  # noqa: E402
from features import extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLANS = {"tpch": ROOT / "results/runs/M1/pass1/plans", "tpcds": ROOT / "results/runs/M1_tpcds/pass1/plans"}
LABELS = {"tpch": ROOT / "results/runs/M1/labels.csv", "tpcds": ROOT / "results/runs/M1_tpcds/labels.csv"}
SMALL = replace(nn.CONFIG, layers=2, hidden=16, mlp=(16, 8), dropout=0.1, epochs=3, batch=4)


def node(kind, cost, rows, children=(), **extra):
    return {"Node Type": kind, "Startup Cost": cost / 2, "Total Cost": cost, "Plan Rows": rows,
            "Plan Width": 8, **extra, "Plans": list(children)}


def example():
    """Limit -> Gather(2 workers) -> Hash Join(Seq Scan, Hash -> Seq Scan), and an InitPlan Seq Scan."""
    join = node("Hash Join", 900, 50, [node("Seq Scan", 400, 1000), node("Hash", 300, 20, [node("Seq Scan", 250, 20)])])
    gather = node("Gather", 1000, 50, [join], **{"Workers Planned": 2})
    return [{"Plan": node("Limit", 1100, 10, [gather, node("Seq Scan", 5, 1)])}]


def synthetic_plans(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        leaves = [node("Seq Scan", float(rng.uniform(10, 1e5)), float(rng.integers(1, 10**6)))
                  for _ in range(int(rng.integers(1, 4)))]
        inner = node("Hash Join" if i % 2 else "Nested Loop", float(rng.uniform(1e3, 1e6)), 100, leaves)
        out[f"q{i:03d}"] = nn.parse_plan([{"Plan": node("Aggregate", float(rng.uniform(1e3, 1e6)), 1, [inner])}])
    return out


def fitted(seed=0, n=24):
    plans = synthetic_plans(n, 1)
    ids = sorted(plans)
    y = np.array([np.log1p(plans[q].estimates[0, 0]) / 3 for q in ids])
    return plans, ids, y, nn.PlanModel(plans, SMALL).fit(ids, y, seed=seed)


# ----------------------------------------------------------------- parsing

def test_parse_order_heights_sizes_workers():
    p = nn.parse_plan(example())
    assert p.types == ["Limit", "Gather", "Hash Join", "Seq Scan", "Hash", "Seq Scan", "Seq Scan"]
    assert p.height.tolist() == [0, 1, 2, 3, 3, 4, 1]
    assert p.size.tolist() == [7, 5, 4, 1, 2, 1, 1]
    assert p.workers.tolist() == [0, 2, 0, 0, 0, 0, 0]
    assert p.estimates[0].tolist() == [1100, 550, 10, 8]       # total cost, startup cost, rows, width


@pytest.mark.parametrize("workload", ["tpch", "tpcds"])
def test_real_plans_agree_with_plan_features(workload):
    if not PLANS[workload].exists():
        pytest.skip("plans not present")
    labels = pd.read_csv(LABELS[workload]).set_index("query_id")
    for q in labels.index[::25]:
        with open(PLANS[workload] / f"{q}.json", encoding="utf-8") as f:
            raw = json.load(f)
        p, feats = nn.parse_plan(raw), extract_features(raw)
        assert len(p) == feats["node_count"] == labels.loc[q, "node_count"]
        assert p.height.max() == feats["max_depth"]
        assert p.estimates[0, 0] == pytest.approx(labels.loc[q, "total_cost"])
        assert p.estimates[0, 2] == pytest.approx(labels.loc[q, "plan_rows"])
        assert p.workers.sum() == feats["workers_planned"]
        assert p.size[0] == len(p) and (p.size >= 1).all()


def test_encoder_unseen_type_and_scaling():
    plans = synthetic_plans(10, 2)
    enc = nn.NodeEncoder().fit(list(plans.values()))
    x = enc.transform(nn.parse_plan([{"Plan": node("WindowAgg", 10, 1, [node("Seq Scan", 5, 1)])}]))
    assert x.shape == (2, enc.dim)
    assert x[0, len(enc.types)] == 1.0 and x[0, :len(enc.types)].sum() == 0       # the unseen slot
    assert x[1, enc.index["Seq Scan"]] == 1.0
    train = np.vstack([enc.transform(p) for p in plans.values()])
    num = train[:, len(enc.types) + 1:]
    q25, med, q75 = np.quantile(num, [0.25, 0.5, 0.75], axis=0)
    assert np.allclose(med, 0.0, atol=1e-6)
    spread = q75 > q25
    assert np.allclose((q75 - q25)[spread], 1.0, atol=1e-5)
    workers = num[:, -1]
    assert (workers == 0).all()                       # no spread: centred, not scaled


def test_qerror_loss():
    z_pred, z = torch.tensor([0.0, 1.0, -0.5]), torch.tensor([0.5, 1.0, 0.5])
    expect = np.mean(np.exp(2.0 * np.array([0.5, 0.0, 1.0])))
    assert nn.qerror_loss(z_pred, z, sd=2.0).item() == pytest.approx(expect, rel=1e-6)
    assert np.isfinite(nn.qerror_loss(torch.tensor([100.0]), torch.tensor([0.0]), sd=2.0).item())


# -------------------------------------------------------------------- mask

def test_subtree_mask():
    p = nn.parse_plan(example())
    m = nn.subtree_mask(p.size, 9)
    expect = np.zeros((9, 9), dtype=bool)
    for i, members in enumerate([range(0, 7), range(1, 6), range(2, 6), [3], [4, 5], [5], [6], [7], [8]]):
        expect[i, list(members)] = True
    assert (m == expect).all()


def test_attention_stays_in_subtree():
    torch.manual_seed(0)
    p = nn.parse_plan(example())
    body = nn.Body(5, SMALL).eval()
    x = np.random.default_rng(0).normal(size=(len(p), 5)).astype(np.float32)
    y = x.copy()
    y[6] += 1.0                          # the InitPlan scan: outside the Gather's subtree
    with torch.no_grad():
        a = body.nodes(*nn.Stack([x], [p.size]).batch()[:2])
        b = body.nodes(*nn.Stack([y], [p.size]).batch()[:2])
    assert torch.equal(a[1:6], b[1:6])   # the Gather and everything under it see nothing of node 6
    assert not torch.allclose(a[0], b[0])  # the root sees its whole plan
    assert not torch.allclose(a[6], b[6])


def test_batch_padding_does_not_change_roots():
    plans, ids, y, m = fitted()
    plans["long"] = nn.parse_plan(example())       # a longer plan pads the others in a batch
    xs = [m.encoder.transform(plans[q]) for q in ids[:5] + ["long"]]
    sizes = [plans[q].size for q in ids[:5] + ["long"]]
    m.body.eval()
    with torch.no_grad():
        together = m.body(*nn.Stack(xs, sizes).batch())
        alone = torch.cat([m.body(*nn.Stack([x], [s]).batch()) for x, s in zip(xs, sizes)])
    assert torch.allclose(together, alone, atol=1e-6)


# ------------------------------------------------------------- determinism

def test_fit_is_deterministic():
    plans, ids, y, m1 = fitted(seed=0)
    _, _, _, m2 = fitted(seed=0)
    _, _, _, m3 = fitted(seed=1)
    assert np.array_equal(m1.predict(ids), m2.predict(ids))
    assert not np.array_equal(m1.predict(ids), m3.predict(ids))
    for a, b in zip(m1.body.state_dict().values(), m2.body.state_dict().values()):
        assert torch.equal(a, b)


def test_fit_is_the_same_in_another_process(tmp_path):
    rng = np.random.default_rng(4)
    ids = [f"q{i:02d}" for i in range(16)]
    for q in ids:
        leaf = node("Seq Scan", float(rng.uniform(10, 1e5)), float(rng.integers(1, 10**5)))
        raw = [{"Plan": node("Aggregate", float(rng.uniform(1e3, 1e6)), 1, [node("Hash Join", 900, 5, [leaf])])}]
        (tmp_path / f"{q}.json").write_text(json.dumps(raw), encoding="utf-8")
    y = rng.normal(5.0, 1.0, len(ids))
    np.save(tmp_path / "y.npy", y)
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import json, numpy as np; "
            "import nn_model as nn; "
            f"ids = {ids!r}; plans = nn.load_plans({str(tmp_path)!r}, ids); cfg = nn.Config(**{asdict(SMALL)!r}); "
            f"m = nn.PlanModel(plans, cfg).fit(ids, np.load({str(tmp_path / 'y.npy')!r}), seed=0); "
            "print(json.dumps(m.predict(ids).tolist()))")
    other = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                      check=True).stdout)
    here = nn.PlanModel(nn.load_plans(tmp_path, ids), SMALL).fit(ids, y, seed=0).predict(ids)
    assert here.tolist() == other


@pytest.mark.parametrize("mode", ["full", "head"])
def test_fine_tuning_is_deterministic_and_leaves_the_source(mode):
    plans, ids, y, m = fitted()
    before = m.predict(ids)
    body_before = {k: v.clone() for k, v in m.body.state_dict().items()}
    k_ids, k_y = ids[:5], y[:5] + 0.7
    t1 = m.fine_tuned(k_ids, k_y, mode, seed=3, lr=1e-2, steps=20)
    t2 = m.fine_tuned(k_ids, k_y, mode, seed=3, lr=1e-2, steps=20)
    assert np.array_equal(t1.predict(ids), t2.predict(ids))
    assert not np.array_equal(t1.predict(ids), before)
    assert np.array_equal(m.predict(ids), before)
    for key, v in m.body.state_dict().items():
        assert torch.equal(v, body_before[key])
    if mode == "head":
        assert t1.body is m.body
    # The fit moves the k queries toward their new labels.
    assert np.abs(t1.predict(k_ids) - k_y).mean() < np.abs(before[:5] - k_y).mean()


def test_checkpoints_match_shorter_runs():
    plans, ids, y, m = fitted()
    seen = {}
    m.fine_tuned(ids[:6], y[:6] + 0.3, "full", seed=0, lr=3e-3, steps=10, checkpoints=[4, 10],
                 callback=lambda s, mod: seen.__setitem__(s, mod.predict(ids)))
    short = m.fine_tuned(ids[:6], y[:6] + 0.3, "full", seed=0, lr=3e-3, steps=4)
    assert np.array_equal(seen[4], short.predict(ids))
    full = m.fine_tuned(ids[:6], y[:6] + 0.3, "full", seed=0, lr=3e-3, steps=10)
    assert np.array_equal(seen[10], full.predict(ids))


def test_fine_tune_rule_is_defined_for_every_k():
    for mode in ("full", "head"):
        for k in (1, 5, 7, 33, 100, 500):
            lr, steps = nn.fine_tune_settings(mode, k)
            assert lr > 0 and steps > 0
        assert nn.fine_tune_settings(mode, 100) == tuple(nn.FINE_TUNE[mode][100])
