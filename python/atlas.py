"""Stage 1 of a GPT-2 small activation atlas: streaming, mergeable statistics in the neuron basis.

Pipeline
--------
1. `tokenize`: OpenWebText -> `tokens.npy`, uint16 [n_rows, seq_len]. Each row is [BOS] followed by
   a (seq_len - 1)-token window of a single document. Windows never cross documents, and partial
   tail windows are dropped. By default each document contributes one randomly chosen window and
   rows come from two OpenWebText files, so 1M tokens span ~8k documents and the two files give a
   corpus systematic. `docs.npy` holds (file, doc_id, offset) for each row.
2. `run`: stream rows through the model. Forward hooks reduce each batch on the device into
   mergeable accumulators (moments, histograms, top-k, residual covariance), so raw activations
   never leave the device. Writes one `state-<start>-<stop>.pt` per row range: `--shards N` runs N
   ranges (none straddling two files) in one process, `--rows` one range, on any machine.
3. `export`: merge all states in the directory and write `neurons.parquet`, `topk.parquet`,
   `resid.npz` and `histograms.npz`, ready for querying with DuckDB. With several shards it adds
   bootstrap errors (resampling shards within each file) and per-shard moments (`shards.parquet`).

    python -m atlas tokenize results/atlas --n-tokens 1000000
    python -m atlas run results/atlas --shards 8       # or --rows 0:4000 on one machine, 4000:7874 on another
    python -m atlas export results/atlas
    python -c "import duckdb; duckdb.sql(\"select * from 'results/atlas/topk.parquet' where layer = 9 and unit = 123 order by rank\").show()"

Conventions
-----------
- Activations are a pure function of (weights, tokens), so exemplars are stored as pointers
  (row, pos) into `tokens.npy`, not as tensors. To get anything else, rerun the model on those rows.
- Position 0 (BOS) is excluded from all statistics. Its residual norm is an outlier by orders of
  magnitude (an attention sink) and would dominate every mean, variance and top-k.
- Activations follow TransformerLens's processed convention (`enable_compatibility_mode()`: folded
  LayerNorm, centered writing weights), as in ioi_hydra and the SAELens SAEs. MLP neuron values are
  unchanged by this processing; the residual stream has its mean over d_model removed. The default
  backend, `HFGPT2`, reproduces this from plain HF transformers; `TLGPT2` is the bridge itself, kept
  as the reference (tests/test_backends.py checks they agree).
- MPS has no float64, so each batch is reduced on the device in float32. Moments and covariances
  are then merged on the CPU in float64 using Chan et al.'s parallel formulas.
- All statistics are conditional on the corpus (OpenWebText) and on positions 1..seq_len-1.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

torch.set_grad_enabled(False)

BOS = 50256  # <|endoftext|>: GPT-2's BOS and document separator
OWT_REPO = "Skylion007/openwebtext"

# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------


def owt_texts(shard: int = 0):
    """Yields documents from one of the 80 OpenWebText Parquet files (about 100M tokens each)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(OWT_REPO, f"plain_text/train-{shard:05d}-of-00080.parquet", repo_type="dataset")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=["text"]):
        yield from batch.column("text").to_pylist()


def tokenize(texts, tokenizer, n_tokens: int, seq_len: int = 128, max_windows_per_doc: int | None = None,
             seed: int = 0, batch_size: int = 256):
    """Splits documents into rows of [BOS] + (seq_len - 1) tokens from one document.

    With max_windows_per_doc, a document with more windows contributes that many, chosen at random
    (seeded, so reproducible): rows then span many more documents, at the cost of tokenizing more text.

    Returns tokens, uint16 [n_rows, seq_len], and docs, int64 [n_rows, 2] holding
    (doc_id, offset of the row's first non-BOS token within the document).
    """
    w = seq_len - 1
    n_rows = -(-n_tokens // seq_len)
    rng = np.random.default_rng(seed)
    rows, docs = [], []
    doc_id = 0
    for batch in itertools.batched(texts, batch_size):
        for ids in tokenizer(list(batch))["input_ids"]:
            offs = range(0, len(ids) - w + 1, w)
            if max_windows_per_doc and len(offs) > max_windows_per_doc:
                offs = sorted(int(o) for o in rng.choice(offs, max_windows_per_doc, replace=False))
            for off in offs:
                rows.append([BOS] + ids[off : off + w])
                docs.append((doc_id, off))
                if len(rows) == n_rows:
                    return np.array(rows, dtype=np.uint16), np.array(docs, dtype=np.int64)
            doc_id += 1
    raise ValueError(f"ran out of text after {len(rows)} rows")


# ---------------------------------------------------------------------------------------------
# Mergeable accumulators
#
# Each one has update(batch) and merge(other), and merging is exact: accumulating shard A and
# shard B separately and then merging gives the same result as accumulating A + B in one pass.
# state() / from_state() round-trip through torch.save as plain dicts of tensors.
# ---------------------------------------------------------------------------------------------


class Accumulator:
    def state(self) -> dict:
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in vars(self).items()}

    @classmethod
    def from_state(cls, state: dict, device="cpu"):
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj.to(device)
        return obj

    def to(self, device):
        return self


class Moments(Accumulator):
    """Per-unit count, mean, M2 (sum of squared deviations), max and number of values > 0."""

    def __init__(self, d: int):
        self.n = 0
        self.mean = torch.zeros(d, dtype=torch.float64)
        self.m2 = torch.zeros(d, dtype=torch.float64)
        self.max = torch.full((d,), -torch.inf, dtype=torch.float64)
        self.n_pos = torch.zeros(d, dtype=torch.int64)

    def update(self, x: torch.Tensor):
        """x: [N, d] on any device."""
        # One fused var_mean and a single device -> CPU transfer (each .cpu() is a sync).
        var, mean = torch.var_mean(x, 0, correction=0)
        n_pos = (x > 0).sum(0, dtype=torch.float32)  # exact: counts per batch are far below 2**24
        mean, m2, mx, n_pos = torch.stack([mean, var * x.shape[0], x.max(0).values, n_pos]).cpu().double()
        b = Moments.__new__(Moments)
        b.n, b.mean, b.m2, b.max, b.n_pos = x.shape[0], mean, m2, mx, n_pos.long()
        self.merge(b)

    def merge(self, b: Moments):
        if b.n == 0:
            return
        n = self.n + b.n
        delta = b.mean - self.mean
        self.m2 += b.m2 + delta**2 * (self.n * b.n / n)
        self.mean += delta * (b.n / n)
        self.max = torch.maximum(self.max, b.max)
        self.n_pos += b.n_pos
        self.n = n

    @property
    def var(self) -> torch.Tensor:
        return self.m2 / self.n


class Covariance(Accumulator):
    """Full covariance (population, i.e. divided by n) of a [N, d] stream. Use it for the residual
    stream (d = 768); it's too big for MLP neurons."""

    def __init__(self, d: int):
        self.n = 0
        self.mean = torch.zeros(d, dtype=torch.float64)
        self.c = torch.zeros(d, d, dtype=torch.float64)

    def update(self, x: torch.Tensor):
        b = Covariance.__new__(Covariance)
        mean = x.mean(0)
        xc = x - mean
        b.n = x.shape[0]
        b.mean = mean.cpu().double()
        b.c = (xc.T @ xc).cpu().double()
        self.merge(b)

    def merge(self, b: Covariance):
        if b.n == 0:
            return
        n = self.n + b.n
        delta = b.mean - self.mean
        self.c += b.c + torch.outer(delta, delta) * (self.n * b.n / n)
        self.mean += delta * (b.n / n)
        self.n = n

    @property
    def cov(self) -> torch.Tensor:
        return self.c / self.n


_ONES: dict = {}


def _ones_int32(n: int, device) -> torch.Tensor:
    """A cached vector of n int32 ones on `device`, so histogram updates don't allocate one per call."""
    buf = _ONES.get(str(device))
    if buf is None or len(buf) < n:
        buf = _ONES[str(device)] = torch.ones(n, dtype=torch.int32, device=device)
    return buf[:n]


def asinh_edges(n_bins: int = 256, lim: float = 1e4, scale: float = 0.01) -> torch.Tensor:
    """Bin edges spaced evenly in asinh(x / scale): linear (width about scale * 0.1) near zero,
    logarithmic (about 12% relative width) for |x| >> scale, covering [-lim, lim]."""
    u = float(np.arcsinh(lim / scale))
    return torch.sinh(torch.linspace(-u, u, n_bins + 1, dtype=torch.float64)) * scale


class Histogram(Accumulator):
    """Per-unit counts over fixed edges. Bin 0 is underflow and bin n_bins + 1 is overflow.
    Bin i (1 <= i <= n_bins) holds edges[i - 1] < x <= edges[i]."""

    def __init__(self, d: int, edges: torch.Tensor, device="cpu"):
        self.edges = edges
        self.counts = torch.zeros(d, len(edges) + 1, dtype=torch.int64, device=device)
        self.to(device)

    def to(self, device):
        self.counts = self.counts.to(device)
        self._edges = self.edges.to(device, torch.float32)
        return self

    def state(self) -> dict:
        s = super().state()
        del s["_edges"]
        return s

    def update(self, x: torch.Tensor):
        d, nb = self.counts.shape
        flat = (torch.bucketize(x, self._edges) + torch.arange(d, device=x.device) * nb).flatten()
        # int32 scatter_add is ~25% faster than bincount on MPS; per-batch counts fit in int32.
        batch = torch.zeros(d * nb, dtype=torch.int32, device=x.device).scatter_add_(0, flat, _ones_int32(len(flat), x.device))
        self.counts += batch.view(d, nb)

    def merge(self, b: Histogram):
        assert torch.equal(self.edges, b.edges), "histograms must share edges to merge"
        self.counts += b.counts.to(self.counts.device)

    def quantiles(self, qs) -> np.ndarray:
        """[d, len(qs)], linearly interpolated within bins. Under- and overflow clamp to the outer edges."""
        counts = self.counts.cpu().numpy().astype(np.float64)
        edges = self.edges.numpy()
        lo = np.concatenate([[edges[0]], edges])  # lower edge of each bin, incl. under/overflow
        hi = np.concatenate([edges, [edges[-1]]])
        cdf = np.cumsum(counts, axis=1)
        out = np.empty((counts.shape[0], len(qs)))
        for j, q in enumerate(qs):
            target = q * cdf[:, -1]
            i = np.minimum((cdf < target[:, None]).sum(1), counts.shape[1] - 1)
            rows = np.arange(len(i))
            below = np.where(i > 0, cdf[rows, i - 1], 0.0)
            frac = np.divide(target - below, counts[rows, i], out=np.zeros(len(i)), where=counts[rows, i] > 0)
            out[:, j] = lo[i] + frac * (hi[i] - lo[i])
        return out


class TopK(Accumulator):
    """Per-unit k largest values and the global token index of each (row * seq_len + pos)."""

    def __init__(self, d: int, k: int, device="cpu"):
        self.values = torch.full((k, d), -torch.inf, device=device)
        self.index = torch.full((k, d), -1, dtype=torch.int64, device=device)

    def to(self, device):
        self.values, self.index = self.values.to(device), self.index.to(device)
        return self

    def update(self, x: torch.Tensor, index: torch.Tensor):
        """x: [N, d]; index: [N] global token index of each row of x."""
        v, i = x.topk(min(self.values.shape[0], x.shape[0]), dim=0)
        self._merge(v, index[i])

    def merge(self, b: TopK):
        self._merge(b.values.to(self.values.device), b.index.to(self.index.device))

    def _merge(self, values, index):
        k = self.values.shape[0]
        v, i = torch.cat([self.values, values]).topk(k, dim=0)
        self.values, self.index = v, torch.cat([self.index, index]).gather(0, i)


# ---------------------------------------------------------------------------------------------
# Atlas: one set of accumulators per hook point
# ---------------------------------------------------------------------------------------------


def default_hooks(cfg) -> dict[str, int]:
    """Hook name -> width. MLP neurons in every layer, plus the residual stream before every layer
    and after the last one (13 points, i.e. what feeds ln_final)."""
    L = cfg.n_layers
    hooks = {f"blocks.{l}.mlp.hook_post": cfg.d_mlp for l in range(L)}
    hooks |= {f"blocks.{l}.hook_resid_pre": cfg.d_model for l in range(L)}
    hooks[f"blocks.{L - 1}.hook_resid_post"] = cfg.d_model
    return hooks


def hook_layer(name: str) -> int:
    """Layer index of a hook. resid_post of layer l is resid_pre of layer l + 1."""
    l = int(name.split(".")[1])
    return l + 1 if name.endswith("hook_resid_post") else l


def hook_kind(name: str) -> str:
    return "resid" if "hook_resid" in name else name.split(".")[-1].removeprefix("hook_")


def hist_mask(index: torch.Tensor, T: int, stride: int) -> torch.Tensor:
    """Which tokens feed the histograms: every `stride`-th non-BOS token by global counter
    row * (T - 1) + (pos - 1). Depends only on the token's global position, so any sharding or batching
    selects the same tokens and merging stays exact; with T - 1 odd, the selection cycles through all positions."""
    return ((index // T) * (T - 1) + index % T - 1) % stride == 0


class Atlas:
    """Accumulators for every hook. Moments, top-k and covariance see every token; histograms (and so the
    exported quantiles) see every `hist_stride`-th token, which cuts their cost, the largest, by that factor."""

    def __init__(self, hooks: dict[str, int], k: int = 20, n_bins: int = 256, hist_stride: int = 1, device="cpu"):
        edges = asinh_edges(n_bins)
        self.hooks = dict(hooks)
        self.hist_stride = hist_stride
        self.n_rows = 0
        self.acc: dict[str, dict[str, Accumulator]] = {}
        for name, d in hooks.items():
            a = {"moments": Moments(d), "hist": Histogram(d, edges, device), "topk": TopK(d, k, device)}
            if hook_kind(name) == "resid":
                a["cov"] = Covariance(d)
            self.acc[name] = a

    def update(self, name: str, act: torch.Tensor, index: torch.Tensor):
        """act: [B, T, d] from a hook; index: [B * (T - 1)] global token index of positions 1..T-1."""
        x = act[:, 1:].reshape(-1, act.shape[-1]).float()
        a = self.acc[name]
        a["moments"].update(x)
        a["hist"].update(x if self.hist_stride == 1 else x[hist_mask(index, act.shape[1], self.hist_stride)])
        a["topk"].update(x, index)
        if "cov" in a:
            a["cov"].update(x)

    def merge(self, other: Atlas):
        assert self.hooks == other.hooks
        assert self.hist_stride == other.hist_stride, "histograms from different token subsamples cannot merge"
        self.n_rows += other.n_rows
        for name, a in self.acc.items():
            for kind, acc in a.items():
                acc.merge(other.acc[name][kind])

    def state(self) -> dict:
        return {
            "hooks": self.hooks,
            "hist_stride": self.hist_stride,
            "n_rows": self.n_rows,
            "acc": {name: {kind: acc.state() for kind, acc in a.items()} for name, a in self.acc.items()},
        }

    @classmethod
    def from_state(cls, state: dict, device="cpu") -> Atlas:
        classes = {"moments": Moments, "hist": Histogram, "topk": TopK, "cov": Covariance}
        obj = cls.__new__(cls)
        obj.hooks, obj.n_rows = state["hooks"], state["n_rows"]
        obj.hist_stride = state.get("hist_stride", 1)  # states written before subsampling existed
        obj.acc = {
            name: {kind: classes[kind].from_state(s, device) for kind, s in a.items()}
            for name, a in state["acc"].items()
        }
        return obj


# ---------------------------------------------------------------------------------------------
# Model backends
#
# A backend has `cfg` (n_layers, d_model, d_mlp) and `run(tokens, hooks)`, which calls
# hooks[name](act) with act [B, T, d] at each named hook point. Names and values follow
# TransformerLens. A hook may raise _Stop to end the forward pass early.
# ---------------------------------------------------------------------------------------------


class _Stop(Exception):
    """Raised from the last hook so the forward pass skips the remaining layers and the unembed."""


class HFGPT2:
    """GPT-2 in plain Hugging Face transformers, with TransformerLens hook names and conventions.

    Centering the writing weights is linear, so it equals subtracting each token's mean over d_model
    from everything written to the residual stream (resid_*, attn_out, mlp_out). Folding LayerNorm
    leaves the MLP pre- and post-activations unchanged. This avoids the bridge's per-module hook
    overhead (about 1.8x faster on an M4) and keeps the model plain for bf16 and torch.compile later.
    """

    def __init__(self, device="cpu", name="gpt2"):
        from transformers import GPT2Model

        self.model = GPT2Model.from_pretrained(name).to(device).eval()  # no LM head: nothing to unembed
        c = self.model.config
        self.cfg = SimpleNamespace(n_layers=c.n_layer, d_model=c.n_embd, d_mlp=c.n_inner or 4 * c.n_embd)

    def _register(self, name: str, fn):
        _, layer, *rest = name.split(".")
        block = self.model.h[int(layer)]
        center = lambda x: x - x.mean(-1, keepdim=True)
        first = lambda out: out[0] if isinstance(out, tuple) else out
        match rest:
            case ["hook_resid_pre"]:
                return block.register_forward_pre_hook(
                    lambda m, args, kw: fn(center(args[0] if args else kw["hidden_states"])), with_kwargs=True)
            case ["hook_resid_mid"]:
                return block.ln_2.register_forward_pre_hook(lambda m, args: fn(center(args[0])))
            case ["hook_resid_post"]:
                return block.register_forward_hook(lambda m, args, out: fn(center(first(out))))
            case ["hook_attn_out"]:
                return block.attn.register_forward_hook(lambda m, args, out: fn(center(first(out))))
            case ["hook_mlp_out"]:
                return block.mlp.register_forward_hook(lambda m, args, out: fn(center(out)))
            case ["mlp", "hook_pre"]:
                return block.mlp.act.register_forward_pre_hook(lambda m, args: fn(args[0]))
            case ["mlp", "hook_post"]:
                return block.mlp.act.register_forward_hook(lambda m, args, out: fn(out))
        raise KeyError(f"HFGPT2 has no hook {name!r}")

    def run(self, tokens: torch.Tensor, hooks: dict):
        handles = [self._register(name, fn) for name, fn in hooks.items()]
        try:
            self.model(tokens)
        except _Stop:
            pass
        finally:
            for h in handles:
                h.remove()


class TLGPT2:
    """The TransformerLens bridge in compatibility mode: the reference implementation."""

    def __init__(self, device="cpu", name="gpt2"):
        from transformer_lens.model_bridge import TransformerBridge

        self.model = TransformerBridge.boot_transformers(name, device=device)
        self.model.enable_compatibility_mode()
        self.cfg = self.model.cfg

    def run(self, tokens: torch.Tensor, hooks: dict):
        # Bind each fn to the name we registered: the bridge's hook.name is the canonical name, not the alias.
        with self.model.hooks(fwd_hooks=[(name, lambda act, hook, fn=fn: fn(act)) for name, fn in hooks.items()]):
            try:
                self.model(tokens, return_type=None)
            except _Stop:
                pass


BACKENDS = {"hf": HFGPT2, "tl": TLGPT2}


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------


def run(model, tokens: np.ndarray, start: int, stop: int, batch_size: int = 64, k: int = 20, hist_stride: int = 1,
        device="cpu", progress=True) -> Atlas:
    from tqdm.auto import tqdm

    hooks = default_hooks(model.cfg)
    atlas = Atlas(hooks, k=k, hist_stride=hist_stride, device=device)
    last = list(hooks)[-1]
    T = tokens.shape[1]
    pos = torch.arange(1, T, device=device)
    index = None

    def make_hook(name):
        def hook(act):
            atlas.update(name, act, index)
            if name == last:
                raise _Stop

        return hook

    callbacks = {name: make_hook(name) for name in hooks}

    for r0 in tqdm(range(start, stop, batch_size), disable=not progress, unit="batch"):
        r1 = min(r0 + batch_size, stop)
        toks = torch.from_numpy(tokens[r0:r1].astype(np.int64)).to(device)
        index = (torch.arange(r0, r1, device=device)[:, None] * T + pos).flatten()
        model.run(toks, callbacks)
        atlas.n_rows += r1 - r0
    return atlas


# ---------------------------------------------------------------------------------------------
# Statistical errors: bootstrap over shards
#
# Each shard is an independent sample of rows, so resampling shards with replacement and pooling
# them gives bootstrap errors for any mergeable statistic. Shards are resampled within their
# OpenWebText file, so the file-to-file difference stays a separate, systematic, comparison.
# Rows within a shard are correlated (neighbouring documents, shared topics); resampling whole
# shards accounts for that, at the price of few resampling units: treat errors from 4 shards per
# file as approximate.
# ---------------------------------------------------------------------------------------------


def shard_ranges(files: np.ndarray, n_shards: int) -> list[tuple[int, int]]:
    """Split rows into n_shards contiguous ranges, none straddling two files: each file's rows get
    the same number of shards. files: [n_rows] file of each row (rows grouped by file)."""
    bounds = np.flatnonzero(np.diff(files)) + 1
    starts, stops = np.concatenate([[0], bounds]), np.concatenate([bounds, [len(files)]])
    assert n_shards % len(starts) == 0, f"--shards {n_shards} must be a multiple of the {len(starts)} files"
    out = []
    for a, b in zip(starts, stops):
        edges = np.linspace(a, b, n_shards // len(starts) + 1).round().astype(int)
        out += list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    return out


def bootstrap_weights(groups, n_boot: int = 500, seed: int = 0) -> np.ndarray:
    """[n_boot, S]: how many times each shard appears in each resample, drawn within each group."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    w = np.zeros((n_boot, len(groups)))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def pooled_moments(w, n, mean, m2, n_pos, chunk: int = 50):
    """Pool per-shard moments with shard multiplicities w [B, S]. n [S]; mean, m2, n_pos [S, U].
    Returns mean, std, frac_pos, each [B, U]; with w = 1 this is exactly Moments.merge over the shards."""
    out = {"mean": [], "std": [], "frac_pos": []}
    for wb in np.array_split(w, max(1, len(w) // chunk)):
        nw = wb @ n
        mu = (wb * n) @ mean / nw[:, None]
        # Sum of squared deviations about the pooled mean: within-shard m2 plus each shard's offset.
        ss = np.einsum("bs,su->bu", wb, m2) + np.einsum("bs,s,bsu->bu", wb, n, (mean[None] - mu[:, None]) ** 2)
        out["mean"].append(mu)
        out["std"].append(np.sqrt(ss / nw[:, None]))
        out["frac_pos"].append((wb @ n_pos) / nw[:, None])
    return {k: np.concatenate(v) for k, v in out.items()}


def pooled_cov_summaries(w, n, mean, c, chunk: int = 50):
    """Total variance and participation ratio (tr C)^2 / ||C||_F^2 of the pooled covariance, for each
    resample w [B, S]. n [S]; mean [S, d]; c [S, d, d] (sums of centered outer products). Returns [B], [B]."""
    S, d = mean.shape
    A = (c + n[:, None, None] * mean[:, :, None] * mean[:, None, :]).reshape(S, -1)  # raw second moments
    total, pr = [], []
    for wb in np.array_split(w, max(1, len(w) // chunk)):
        nw = wb @ n
        mu = (wb * n) @ mean / nw[:, None]
        C = (wb @ A).reshape(-1, d, d) / nw[:, None, None] - mu[:, :, None] * mu[:, None, :]
        tr = np.trace(C, axis1=1, axis2=2)
        total.append(tr)
        pr.append(tr**2 / (C**2).sum((1, 2)))
    return np.concatenate(total), np.concatenate(pr)


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------

QUANTILES = {"p50": 0.5, "p90": 0.9, "p99": 0.99, "p999": 0.999}


def contexts(tokens: np.ndarray, tokenizer, row: np.ndarray, pos: np.ndarray, before: int = 24, after: int = 4):
    """Returns (token, context) strings for each (row, pos). The context marks the token as «tok»
    and shows newlines as ⏎. Duplicate pointers are decoded once."""
    keys, inv = np.unique(np.stack([row, pos], 1), axis=0, return_inverse=True)
    pre, tok, post = [], [], []
    for r, p in keys:
        pre.append(tokens[r, max(1, p - before) : p].tolist())
        tok.append([int(tokens[r, p])])
        post.append(tokens[r, p + 1 : p + 1 + after].tolist())
    pre, tok, post = (tokenizer.batch_decode(x) for x in (pre, tok, post))
    clean = lambda s: s.replace("\n", "⏎")
    tok_s = np.array([clean(t) for t in tok], dtype=object)
    ctx_s = np.array([clean(a) + "«" + clean(t) + "»" + clean(b) for a, t, b in zip(pre, tok, post)], dtype=object)
    return tok_s[inv.ravel()], ctx_s[inv.ravel()]


def export(atlas: Atlas, tokens: np.ndarray, tokenizer, out: Path, shards: list[Atlas] = (), groups=(), n_boot: int = 500):
    """Write the merged atlas. With two or more shards (and the file of each in `groups`), also write
    bootstrap errors (`*_err` columns, resid.npz `*_err`), per-file residual summaries and the
    per-shard moments in shards.parquet."""
    import pandas as pd

    T = tokens.shape[1]
    neurons, topk = [], []
    for name, a in atlas.acc.items():
        m, h, t = a["moments"], a["hist"], a["topk"]
        d = len(m.mean)
        q = h.quantiles(list(QUANTILES.values()))
        neurons.append(pd.DataFrame({
            "hook": name, "kind": hook_kind(name), "layer": hook_layer(name), "unit": np.arange(d),
            "n": m.n, "mean": m.mean.numpy(), "std": m.var.sqrt().numpy(), "max": m.max.numpy(),
            "frac_pos": (m.n_pos.double() / m.n).numpy(),
            **{col: q[:, j] for j, col in enumerate(QUANTILES)},
        }))
        values, index = t.values.cpu().numpy(), t.index.cpu().numpy()
        k = values.shape[0]
        topk.append(pd.DataFrame({
            "hook": name, "kind": hook_kind(name), "layer": hook_layer(name),
            "unit": np.tile(np.arange(d), k), "rank": np.repeat(np.arange(k), d),
            "value": values.ravel(), "row": index.ravel() // T, "pos": index.ravel() % T,
        }))
    neurons, topk = pd.concat(neurons, ignore_index=True), pd.concat(topk, ignore_index=True)
    boot = len(shards) >= 2
    if boot:
        groups = np.asarray(groups)
        w = bootstrap_weights(groups, n_boot)
        # [S] tokens per shard and [S, U] moments, units in the same order as `neurons`.
        n = np.array([float(sh.acc[next(iter(sh.acc))]["moments"].n) for sh in shards])
        stack = lambda f: np.stack([np.concatenate([f(sh.acc[name]["moments"]) for name in atlas.acc]) for sh in shards])
        mean, m2 = stack(lambda m: m.mean.numpy()), stack(lambda m: m.m2.numpy())
        n_pos = stack(lambda m: m.n_pos.double().numpy())
        reps = pooled_moments(w, n, mean, m2, n_pos)
        for col in ("mean", "std", "frac_pos"):
            neurons[f"{col}_err"] = reps[col].std(0)
        pd.DataFrame({
            "shard": np.repeat(np.arange(len(shards)), mean.shape[1]), "file": np.repeat(groups, mean.shape[1]),
            "hook": np.tile(neurons["hook"].to_numpy(), len(shards)), "unit": np.tile(neurons["unit"].to_numpy(), len(shards)),
            "n": np.repeat(n, mean.shape[1]), "mean": mean.ravel(), "m2": m2.ravel(), "n_pos": n_pos.ravel(),
        }).to_parquet(out / "shards.parquet", index=False)
    topk = topk[topk["row"] >= 0]  # unfilled slots, only possible with fewer than k tokens
    topk["token"], topk["context"] = contexts(tokens, tokenizer, topk["row"].to_numpy(), topk["pos"].to_numpy())
    neurons.to_parquet(out / "neurons.parquet", index=False)
    topk.sort_values(["hook", "unit", "rank"]).to_parquet(out / "topk.parquet", index=False)

    resid = [name for name in atlas.acc if "cov" in atlas.acc[name]]
    resid.sort(key=hook_layer)
    extra = {}
    if boot:
        # Total variance and effective dimension per layer: bootstrap errors, and each file on its own.
        files = np.unique(groups)
        keys = ["total_var_err", "pr_err"] + [f"{k}_file" for k in ("total_var", "pr", "total_var_err", "pr_err")]
        extra = {k: [] for k in keys}
        for name in resid:
            cs = [sh.acc[name]["cov"] for sh in shards]
            args = (np.array([float(c.n) for c in cs]), np.stack([c.mean.numpy() for c in cs]), np.stack([c.c.numpy() for c in cs]))
            t, p = pooled_cov_summaries(w, *args)
            extra["total_var_err"].append(t.std())
            extra["pr_err"].append(p.std())
            per_file = []
            for f in files:
                mask = (groups == f).astype(float)
                t0, p0 = pooled_cov_summaries(mask[None], *args)
                tb, pb = pooled_cov_summaries(w * mask, *args)  # resamples restricted to this file's shards
                per_file.append((t0[0], p0[0], tb.std(), pb.std()))
            for k, v in zip(keys[2:], np.array(per_file).T):
                extra[k].append(v)
        extra = {k: np.array(v) for k, v in extra.items()} | {"files": files}
    covs = np.stack([atlas.acc[n]["cov"].cov.numpy() for n in resid])
    tr = np.trace(covs, axis1=1, axis2=2)
    np.savez(
        out / "resid.npz",
        layer=np.array([hook_layer(n) for n in resid]),
        n=np.array([atlas.acc[n]["cov"].n for n in resid]),
        mean=np.stack([atlas.acc[n]["cov"].mean.numpy() for n in resid]),
        cov=covs,
        total_var=tr,
        pr=tr**2 / (covs**2).sum((1, 2)),
        **extra,
    )
    first = next(iter(atlas.acc.values()))["hist"]
    np.savez(out / "histograms.npz", edges=first.edges.numpy(),
             **{name: a["hist"].counts.cpu().numpy() for name, a in atlas.acc.items()})


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device: str, backend: str = "hf"):
    return BACKENDS[backend](device)


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 10**9  # we split documents ourselves; silences the >1024 warning
    return tokenizer


def row_files(dir: Path) -> np.ndarray:
    """OpenWebText file of each row; directories tokenized before multi-file support have one file."""
    docs = np.load(dir / "docs.npy")
    return docs[:, 0] if docs.shape[1] == 3 else np.zeros(len(docs), dtype=np.int64)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(prog="atlas", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("tokenize", help="OpenWebText -> tokens.npy")
    s.add_argument("dir", type=Path)
    s.add_argument("--n-tokens", type=float, default=1e6)
    s.add_argument("--seq-len", type=int, default=128)
    s.add_argument("--files", default="0,1", help="which of the 80 OpenWebText Parquet files, comma-separated; tokens are split evenly")
    s.add_argument("--max-windows-per-doc", type=int, default=1, help="0 = every window of every document")
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("run", help="accumulate statistics over a row range -> state-<start>-<stop>.pt")
    s.add_argument("dir", type=Path)
    s.add_argument("--rows", default=":", help="START:STOP row range (default: all)")
    s.add_argument("--shards", type=int, help="run N row ranges (none straddling two files) in one process; overrides --rows")
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--k", type=int, default=20)
    s.add_argument("--hist-stride", type=int, default=4, help="histograms (quantiles) use every Nth token; 1 = all")
    s.add_argument("--device", default=default_device())
    s.add_argument("--backend", choices=list(BACKENDS), default="hf", help="hf: plain transformers (fast); tl: TransformerLens bridge (reference)")
    s = sub.add_parser("export", help="merge state-*.pt -> parquet + npz")
    s.add_argument("dir", type=Path)
    args = p.parse_args(argv)
    args.dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "tokenize":
        if stale := sorted(args.dir.glob("state-*.pt")):
            p.error(f"{args.dir} has {len(stale)} state file(s) from earlier tokens; use a new directory or delete them")
        t0 = time.time()
        files = [int(f) for f in args.files.split(",")]
        parts = [tokenize(owt_texts(f), load_tokenizer(), int(args.n_tokens) // len(files), args.seq_len,
                          args.max_windows_per_doc or None, args.seed + f) for f in files]
        tokens = np.concatenate([t for t, _ in parts])
        docs = np.concatenate([np.column_stack([np.full(len(d), f), d]) for f, (_, d) in zip(files, parts)])
        np.save(args.dir / "tokens.npy", tokens)
        np.save(args.dir / "docs.npy", docs)
        manifest = {
            "dataset": OWT_REPO, "files": files, "rows_per_file": [len(t) for t, _ in parts], "seq_len": args.seq_len,
            "n_rows": len(tokens), "n_docs": len(np.unique(docs[:, :2], axis=0)),
            "max_windows_per_doc": args.max_windows_per_doc, "seed": args.seed, "tokenizer": "gpt2",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (args.dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{tokens.shape} tokens from {manifest['n_docs']} docs in {time.time() - t0:.0f}s -> {args.dir}")

    elif args.cmd == "run":
        tokens = np.load(args.dir / "tokens.npy", mmap_mode="r")
        if args.shards:
            ranges = shard_ranges(row_files(args.dir), args.shards)
        else:
            ranges = [tuple(int(x) if x else d for x, d in zip(args.rows.split(":"), (0, len(tokens))))]
        model = load_model(args.device, args.backend)
        for start, stop in ranges:
            t0 = time.time()
            atlas = run(model, tokens, start, stop, args.batch_size, args.k, args.hist_stride, args.device, progress=len(ranges) == 1)
            dt = time.time() - t0
            state = atlas.state() | {"rows": (start, stop), "model": "gpt2", "backend": args.backend,
                                     "git_commit": git_commit(), "torch": str(torch.__version__), "seconds": dt}
            path = args.dir / f"state-{start}-{stop}.pt"
            torch.save(state, path)
            print(f"{(stop - start) * (tokens.shape[1] - 1):,} tokens in {dt:.0f}s "
                  f"({(stop - start) * tokens.shape[1] / dt:,.0f} tok/s incl. BOS) -> {path}")

    elif args.cmd == "export":
        paths = sorted(args.dir.glob("state-*.pt"))
        if not paths:
            p.error(f"no state-*.pt in {args.dir}; run `atlas run {args.dir}` first")
        states = [torch.load(path) for path in paths]
        ranges = sorted(s["rows"] for s in states)
        for (_, a), (b, _) in itertools.pairwise(ranges):
            assert a <= b, f"overlapping row ranges {ranges}: tokens would be double counted"
        states.sort(key=lambda s: s["rows"])
        shards = [Atlas.from_state(s) for s in states]
        atlas = Atlas.from_state(copy.deepcopy(states[0]))  # merge works in place: don't alter shard 0
        for sh in shards[1:]:
            atlas.merge(sh)
        files = row_files(args.dir)
        groups = [files[a] if files[a] == files[b - 1] else -1 for a, b in ranges]  # -1: a shard spanning files
        tokens = np.load(args.dir / "tokens.npy", mmap_mode="r")
        export(atlas, tokens, load_tokenizer(), args.dir, shards if len(shards) > 1 else (), groups)
        print(f"merged {len(paths)} state(s), {atlas.n_rows:,} rows, ranges {ranges} -> {args.dir}")


if __name__ == "__main__":
    main()
