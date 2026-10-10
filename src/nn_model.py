"""A neural plan model after DACE, the neural baseline of the paper (H6).

The model follows DACE (Liang et al., "DACE: A Database-Agnostic Cost
Estimator", ICDE 2024, Sec. IV and V-A). A plan is read as the sequence of
its nodes in depth-first order (each node before its children, children in
plan order), and every node is a vector of

  node type   one-hot over the node types of the training plans, plus one
              slot for a type never seen in training
  estimates   log(1 + x) of the optimizer's Total Cost, Startup Cost, Plan
              Rows and Plan Width
  height      the length of the path from the node to the root (DACE's
              height; the root has 0)
  workers     Workers Planned where the node has it (Gather and Gather
              Merge), else 0
with every number centred on its median and divided by its interquartile
range, both over the nodes of the training plans, as DACE's scaler (the
RobustScaler of Zero-Shot) does.
No table, column or predicate enters: the model sees only what the
optimizer has already summarized. DACE feeds these vectors without an
embedding to one post-norm transformer encoder layer with a single attention
head and a hidden width of 128, under a tree-structured attention mask: a
node attends only to the nodes of its own subtree (itself included; in
depth-first order these are the node and the size - 1 nodes after it). An
MLP with layers of 128, 64 and 1 outputs maps a node's vector to the log
runtime of its sub-plan; here it is applied to the root.

Differences from DACE
  loss        DACE trains on the runtime of every sub-plan, the Q-error of
              each node weighted by 0.5 ** height. This study has the
              runtime of the root only, in every environment, so the model
              is trained on the root label alone: the mean Q-error of the
              plan, exp(|log prediction - log runtime|).
  inputs      DACE's node has the type, the cost and the cardinality; the
              startup cost, the width, the height and the planned workers
              are added (the height enters DACE only through its loss
              weights).
  training    DACE trains on about 190,000 plans in batches of 512; the
              source pools here have 294 to 476 plans, so batches have
              Config.batch plans. The dropout, the learning rate and the
              number of epochs in CONFIG were chosen by grouped
              cross-validation inside the source pools of the development
              pairs (src/paper_nn.py cv; the groups are templates, and
              TPC-DS families of near-copy templates); nothing of a target
              machine entered.
  adaptation  DACE adapts to new hardware with LoRA on the MLP's layers.
              Here (PlanModel.fine_tuned) the source model is fine-tuned on
              the k target queries, all k in one batch, with Adam, no weight
              decay and the same loss:
                full  every weight is trained
                head  the encoder is frozen and the whole MLP is trained, on
                      the root vectors of the source encoder (the part that
                      DACE's LoRA adapts)
              The learning rate and the number of steps depend on k alone
              (fine_tune_settings): FINE_TUNE was chosen on the development
              pairs by the error on the target pool queries outside the k
              (src/paper_nn.py tune), never on test queries.

Optimization: AdamW (weight decay 0.01), Config.batch plans per step in a
seeded order, gradients clipped to norm 1, labels standardized by the
training mean and standard deviation (the loss is computed in log
milliseconds).

Determinism: every fit seeds torch's generator and its own shuffling
generator, uses torch's deterministic algorithms and THREADS threads, so a
fit repeated in any process with the same data and seed gives the same
weights.

Usage (src/paper_nn.py drives the evaluation)
  plans = load_plans(Path("results/runs/M1/pass1/plans"), query_ids)
  model = PlanModel(plans).fit(train_ids, np.log(train_ms), seed=0)
  log_ms = model.predict(test_ids)
  tuned = model.fine_tuned(k_ids, np.log(k_ms), mode="full", seed=0)
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

THREADS = 2
ESTIMATES = ["Total Cost", "Startup Cost", "Plan Rows", "Plan Width"]
CLIP = 1.0
PREDICT_BATCH = 256
GROUP = 8          # plans per attention group (Stack)


@dataclass(frozen=True)
class Config:
    """Network and training settings. The network's are DACE's (Sec. V-A)."""
    layers: int = 1          # one encoder layer
    heads: int = 1           # no multi-head attention
    hidden: int = 128        # the transformer's hidden width
    mlp: tuple = (128, 64)   # the MLP's layers before its single output
    dropout: float = 0.0
    lr: float = 3e-3
    weight_decay: float = 1e-2
    batch: int = 32
    epochs: int = 800


# dropout, lr and epochs were chosen by src/paper_nn.py cv (grouped cross-validation in the source pools
# of the development pairs, 2026-10-08): the best of 3 x 3 settings to 400 epochs and their extensions
# to lr 1e-2 and 1600 epochs. Geometric mean Q-error over held-out templates: 2.58 (TPC-H 2.36, TPC-DS
# 2.83), against 2.61 at 1600 epochs and 2.62 at 400.
CONFIG = Config()
MAX_LOG_ERROR = 30.0     # the Q-error loss is exp of the log error, capped to stay finite

# (lr, steps) of fine-tuning on k target queries, chosen by src/paper_nn.py tune on the development pairs
# (2026-10-08; lr 3e-4 to 1e-2, 10 to 400 steps, scored on the target pool queries outside the k); a k
# between two entries takes the nearer one on a log scale. The entries at 400 steps lie on a plateau:
# within 0.002 of 200 steps, and lr 3e-3, which moves further in as many steps, is no better.
FINE_TUNE = {"full": {5: (3e-4, 200), 10: (1e-3, 100), 25: (1e-3, 200), 50: (1e-3, 400), 100: (1e-3, 400)},
             "head": {5: (1e-3, 200), 10: (1e-3, 400), 25: (1e-3, 400), 50: (1e-3, 400), 100: (1e-3, 400)}}


def set_determinism(threads: int = THREADS) -> None:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def fine_tune_settings(mode: str, k: int) -> tuple[float, int]:
    """(learning rate, steps) for fine-tuning on k queries: the FINE_TUNE entry nearest to k in log."""
    table = FINE_TUNE[mode]
    near = min(table, key=lambda x: (abs(math.log(x) - math.log(max(k, 1))), x))
    return table[near]


# ------------------------------------------------------------------- plans

@dataclass
class ParsedPlan:
    """The nodes of one plan in depth-first order: type, raw estimates (n x 4), height, planned
    workers and the size of every node's subtree (itself included)."""
    types: list
    estimates: np.ndarray
    height: np.ndarray
    workers: np.ndarray
    size: np.ndarray

    def __len__(self) -> int:
        return len(self.types)


def parse_plan(explain) -> ParsedPlan:
    """Parse the output of EXPLAIN (FORMAT JSON) (the list, or its first element)."""
    top = explain[0] if isinstance(explain, list) else explain
    types, est, height, workers, size = [], [], [], [], []

    def visit(node: dict, h: int) -> None:
        i = len(types)
        types.append(str(node.get("Node Type", "")))
        est.append([float(node.get(key, 0.0) or 0.0) for key in ESTIMATES])
        height.append(h)
        workers.append(float(node.get("Workers Planned", 0) or 0))
        size.append(1)
        for child in node.get("Plans", []) or []:
            visit(child, h + 1)
        size[i] = len(types) - i

    visit(top["Plan"], 0)
    return ParsedPlan(types, np.array(est, float).reshape(-1, len(ESTIMATES)), np.array(height, float),
                      np.array(workers, float), np.array(size, int))


def load_plans(directory: Path, query_ids) -> dict:
    """ParsedPlan of every query ID from <directory>/<query_id>.json."""
    out = {}
    for q in query_ids:
        with open(Path(directory) / f"{q}.json", encoding="utf-8") as f:
            out[str(q)] = parse_plan(json.load(f))
    return out


def subtree_mask(size: np.ndarray, length: int) -> np.ndarray:
    """allowed[i, j]: node i attends to node j, which holds when j lies in i's subtree.

    In depth-first order the subtree of node i is the nodes i to
    i + size[i] - 1. Rows past the plan (padding) attend to themselves only,
    so that their softmax is defined; no node of the plan attends to them.
    """
    n = len(size)
    j = np.arange(length)
    i = np.arange(n)
    allowed = np.zeros((length, length), dtype=bool)
    allowed[:n] = (j[None, :] >= i[:, None]) & (j[None, :] < (i + np.asarray(size))[:, None])
    allowed[n:, n:] = np.eye(length - n, dtype=bool)
    return allowed


class NodeEncoder:
    """Node vectors of plans, with the type vocabulary and the scaling fitted on training plans."""

    def fit(self, plans: list) -> "NodeEncoder":
        self.types = sorted({t for p in plans for t in p.types})
        self.index = {t: i for i, t in enumerate(self.types)}
        num = np.vstack([self._numbers(p) for p in plans])
        q25, self.center, q75 = np.quantile(num, [0.25, 0.5, 0.75], axis=0)
        self.scale = np.where(q75 > q25, q75 - q25, 1.0)
        return self

    @staticmethod
    def _numbers(p: ParsedPlan) -> np.ndarray:
        return np.c_[np.log1p(np.maximum(p.estimates, 0.0)), p.height, p.workers]

    @property
    def dim(self) -> int:
        return len(self.types) + 1 + len(ESTIMATES) + 2

    def transform(self, p: ParsedPlan) -> np.ndarray:
        n, v = len(p), len(self.types) + 1
        x = np.zeros((n, self.dim), dtype=np.float32)
        x[np.arange(n), [self.index.get(t, v - 1) for t in p.types]] = 1.0
        x[:, v:] = (self._numbers(p) - self.center) / self.scale
        return x


# ----------------------------------------------------------------- network

class Stack:
    """Encoded plans of a list of query IDs, from which batches are cut.

    A batch keeps its nodes packed, one row per node of its plans, so that
    the linear layers do no work on padding. Only the attention lays plans
    out in a square, and it does so in groups of GROUP plans of similar
    length, each padded to its own longest plan under the plans'
    subtree_mask: a long TPC-DS plan does not pad a whole batch.
    """

    def __init__(self, xs: list, sizes: list):
        self.lengths = np.array([len(x) for x in xs])
        self.start = np.r_[0, np.cumsum(self.lengths)[:-1]]
        self.x = torch.from_numpy(np.vstack(xs))
        n = int(self.lengths.max())
        self.allowed = torch.from_numpy(np.stack([subtree_mask(s, n) for s in sizes]))

    def __len__(self) -> int:
        return len(self.lengths)

    def batch(self, idx=None) -> tuple:
        """(packed node inputs, attention groups, row of the root of every plan of idx in its order).

        Nodes are packed plan by plan in order of length, and every group is
        (first row, end row, plan of every row, its slot in the plan, masks).
        """
        idx = np.arange(len(self)) if idx is None else np.asarray(idx)
        order = np.argsort(self.lengths[idx], kind="stable")
        plans = idx[order]
        lens = self.lengths[plans]
        first = np.r_[0, np.cumsum(lens)[:-1]]
        nodes = np.concatenate([np.arange(self.start[i], self.start[i] + n) for i, n in zip(plans, lens)])
        roots = np.empty(len(idx), dtype=np.int64)
        roots[order] = first
        t = torch.from_numpy
        groups = []
        for g in range(0, len(plans), GROUP):
            gl = lens[g:g + GROUP]
            n = int(gl.max())
            plan = np.repeat(np.arange(len(gl)), gl)
            slot = np.concatenate([np.arange(m) for m in gl])
            r0 = int(first[g])
            groups.append((r0, r0 + int(gl.sum()), t(plan), t(slot), self.allowed[t(plans[g:g + GROUP]), :n, :n]))
        return self.x[t(nodes)], groups, t(roots)


class TreeAttention(nn.Module):
    """Multi-head self-attention of packed nodes, restricted to each group's allowed[plan, i, j]."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, groups: list) -> torch.Tensor:
        d = h.shape[1]
        dh = d // self.heads
        qkv = self.qkv(h)
        parts = []
        for r0, r1, plan, slot, allowed in groups:
            b, n = allowed.shape[0], allowed.shape[1]
            laid = qkv.new_zeros(b, n, 3 * d).index_put((plan, slot), qkv[r0:r1])
            q, k, v = laid.view(b, n, 3, self.heads, dh).permute(2, 0, 3, 1, 4)
            scores = (q @ k.transpose(-1, -2) / math.sqrt(dh)).masked_fill(~allowed[:, None], float("-inf"))
            a = self.drop(torch.softmax(scores, dim=-1))
            parts.append((a @ v).transpose(1, 2).reshape(b, n, d)[plan, slot])
        return self.out(torch.cat(parts))


class EncoderLayer(nn.Module):
    """A post-norm transformer layer (as torch.nn.TransformerEncoderLayer, GELU) with tree attention."""

    def __init__(self, d: int, heads: int, hidden: int, dropout: float):
        super().__init__()
        if d % heads:
            raise ValueError(f"{heads} heads do not divide the node vector's length {d}")
        self.attn = TreeAttention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, groups: list) -> torch.Tensor:
        h = self.norm1(h + self.drop1(self.attn(h, groups)))
        return self.norm2(h + self.drop2(self.ff(h)))


class Body(nn.Module):
    """The encoder over the node vectors (no embedding, as in DACE); returns the root's vector of
    every plan of a batch."""

    def __init__(self, n_in: int, cfg: Config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_in, cfg.heads, cfg.hidden, cfg.dropout)
                                     for _ in range(cfg.layers)])

    def nodes(self, x: torch.Tensor, groups: list) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, groups)
        return h

    def forward(self, x: torch.Tensor, groups: list, roots: torch.Tensor) -> torch.Tensor:
        return self.nodes(x, groups)[roots]


def make_head(n_in: int, cfg: Config) -> nn.Module:
    """DACE's MLP: ReLU layers of cfg.mlp outputs, then one output."""
    layers, width = [], n_in
    for out in cfg.mlp:
        layers += [nn.Linear(width, out), nn.ReLU(), nn.Dropout(cfg.dropout)]
        width = out
    return nn.Sequential(*layers, nn.Linear(width, 1))


def qerror_loss(z_pred: torch.Tensor, z: torch.Tensor, sd: float) -> torch.Tensor:
    """Mean Q-error of standardized log predictions: exp(|log prediction - log runtime|)."""
    return torch.exp(torch.clamp(sd * torch.abs(z_pred - z), max=MAX_LOG_ERROR)).mean()


# ------------------------------------------------------------------- model

class PlanModel:
    """The node encoder, the network and the label scaling of one fitted model.

    plans maps query IDs to ParsedPlan. A model is used through query IDs:
    fit(ids, log labels), predict(ids), and as a source model of
    adapt_eval.Pair through __call__(frame), which predicts the rows of a
    frame indexed by query ID.
    """

    def __init__(self, plans: dict, config: Config = CONFIG):
        self.plans = plans
        self.config = config
        self.encoder = None
        self.body = self.head = None
        self.mu = self.sd = None
        self.train_ids: list = []
        self._encoded: dict = {}
        self._roots: dict | None = None      # root vectors of a frozen encoder, by query ID

    # -- data
    def _plan(self, q):
        x = self._encoded.get(q)
        if x is None:
            x = self._encoded[q] = self.encoder.transform(self.plans[q])
        return x

    def _stack(self, ids: list) -> Stack:
        return Stack([self._plan(q) for q in ids], [self.plans[q].size for q in ids])

    # -- fitting
    def fit(self, ids, y_log: np.ndarray, seed: int = 0, epochs: int | None = None,
            checkpoints=(), callback=None) -> "PlanModel":
        """Train a new network on the plans of ids and their log labels.

        callback(epoch, model) is called after every epoch in checkpoints
        (cross-validation reads several training lengths off one run).
        """
        set_determinism()
        cfg = self.config
        epochs = cfg.epochs if epochs is None else epochs
        ids = [str(q) for q in ids]
        y = np.asarray(y_log, float)
        self.train_ids = ids
        self.encoder = NodeEncoder().fit([self.plans[q] for q in ids])
        self._encoded, self._roots = {}, None
        self.mu, self.sd = float(y.mean()), float(y.std()) or 1.0
        torch.manual_seed(seed)
        self.body, self.head = Body(self.encoder.dim, cfg), make_head(self.encoder.dim, cfg)
        params = list(self.body.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        stack = self._stack(ids)
        z = torch.tensor((y - self.mu) / self.sd, dtype=torch.float32)
        gen = torch.Generator().manual_seed(seed)
        marks = set(checkpoints)
        for epoch in range(1, epochs + 1):
            self.body.train()
            self.head.train()
            perm = torch.randperm(len(ids), generator=gen).numpy()
            for start in range(0, len(ids), cfg.batch):
                idx = perm[start:start + cfg.batch]
                loss = qerror_loss(self.head(self.body(*stack.batch(idx))).squeeze(-1), z[idx], self.sd)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, CLIP)
                opt.step()
            if epoch in marks and callback is not None:
                callback(epoch, self)
        self._roots = {}
        return self

    def fine_tuned(self, ids, y_log: np.ndarray, mode: str, seed: int = 0, lr: float | None = None,
                   steps: int | None = None, checkpoints=(), callback=None) -> "PlanModel":
        """A copy of this model fine-tuned on the k plans of ids (mode 'full' or 'head').

        The source model is left unchanged. lr and steps default to the rule
        fine_tune_settings(mode, k). callback(step, model) as in fit.
        """
        set_determinism()
        ids = [str(q) for q in ids]
        if lr is None or steps is None:
            lr0, steps0 = fine_tune_settings(mode, len(ids))
            lr, steps = lr if lr is not None else lr0, steps if steps is not None else steps0
        new = copy.copy(self)
        new.head = copy.deepcopy(self.head)
        z = torch.tensor((np.asarray(y_log, float) - self.mu) / self.sd, dtype=torch.float32)
        torch.manual_seed(seed)
        if mode == "head":
            new._roots = self._roots if self._roots is not None else {}
            roots = new._root_vectors(ids)
            params = list(new.head.parameters())

            def loss_fn():
                return qerror_loss(new.head(roots).squeeze(-1), z, self.sd)
        elif mode == "full":
            new.body = copy.deepcopy(self.body)
            new._encoded = dict(self._encoded)
            new._roots = None
            batch = new._stack(ids).batch()
            params = list(new.body.parameters()) + list(new.head.parameters())

            def loss_fn():
                new.body.train()
                return qerror_loss(new.head(new.body(*batch)).squeeze(-1), z, self.sd)
        else:
            raise ValueError(f"unknown fine-tuning mode {mode}")
        opt = torch.optim.Adam(params, lr=lr)
        marks = set(checkpoints)
        for step in range(1, steps + 1):
            new.head.train()
            loss = loss_fn()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            if step in marks and callback is not None:
                callback(step, new)
        if mode == "full":
            new._roots = {}
        return new

    # -- prediction
    def _root_vectors(self, ids: list) -> torch.Tensor:
        """Root vectors of the encoder in eval mode, cached while the encoder is frozen."""
        self.body.eval()
        cache = self._roots if self._roots is not None else {}
        missing = [q for q in dict.fromkeys(ids) if q not in cache]
        if missing:
            stack = self._stack(missing)
            with torch.no_grad():
                for start in range(0, len(missing), PREDICT_BATCH):
                    idx = np.arange(start, min(start + PREDICT_BATCH, len(missing)))
                    for i, row in zip(idx, self.body(*stack.batch(idx))):
                        cache[missing[i]] = row
        return torch.stack([cache[q] for q in ids])

    def predict(self, ids) -> np.ndarray:
        """Log runtime (log ms when trained on log ms) of the plans of ids."""
        ids = [str(q) for q in ids]
        roots = self._root_vectors(ids)
        self.head.eval()
        with torch.no_grad():
            z = self.head(roots).squeeze(-1).numpy().astype(float)
        return self.mu + self.sd * z

    def __call__(self, frame) -> np.ndarray:
        return self.predict(frame.index)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in list(self.body.parameters()) + list(self.head.parameters()))

    def describe(self) -> dict:
        return {**asdict(self.config), "inputs": self.encoder.dim, "node_types": self.encoder.types,
                "parameters": self.n_parameters()}
