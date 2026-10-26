"""Stage 1 of a GPT-2 small activation atlas: streaming, mergeable statistics in the neuron basis.

Pipeline
--------
1. `tokenize`: OpenWebText -> `tokens.npy`, uint16 [n_rows, seq_len]. Each row is [BOS] followed by
   a (seq_len - 1)-token window of a single document. Windows never cross documents, and partial
   tail windows are dropped. `docs.npy` holds (doc_id, offset) for each row.
2. `run`: stream rows through the model. Forward hooks reduce each batch on the device into
   mergeable accumulators (moments, histograms, top-k, residual covariance), so raw activations
   never leave the device. Writes one `state-<start>-<stop>.pt` per row range, so shards can run
   in parallel on any number of machines.
3. `export`: merge all states in the directory and write `neurons.parquet`, `topk.parquet`,
   `resid.npz` and `histograms.npz`, ready for querying with DuckDB.

    python -m atlas tokenize results/atlas --n-tokens 1000000
    python -m atlas run results/atlas                  # or --rows 0:4000 on one machine, 4000:8000 on another
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


def tokenize(texts, tokenizer, n_tokens: int, seq_len: int = 128, batch_size: int = 256):
    """Splits documents into rows of [BOS] + (seq_len - 1) tokens from one document.

    Returns tokens, uint16 [n_rows, seq_len], and docs, int64 [n_rows, 2] holding
    (doc_id, offset of the row's first non-BOS token within the document).
    """
    w = seq_len - 1
    n_rows = -(-n_tokens // seq_len)
    rows, docs = [], []
    doc_id = 0
    for batch in itertools.batched(texts, batch_size):
        for ids in tokenizer(list(batch))["input_ids"]:
            for off in range(0, len(ids) - w + 1, w):
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
        b = Moments.__new__(Moments)
        mean = x.mean(0)
        b.n = x.shape[0]
        b.mean = mean.cpu().double()
        b.m2 = ((x - mean) ** 2).sum(0).cpu().double()
        b.max = x.max(0).values.cpu().double()
        b.n_pos = (x > 0).sum(0).cpu()
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
        flat = torch.bucketize(x, self._edges) + torch.arange(d, device=x.device) * nb
        self.counts += torch.bincount(flat.flatten(), minlength=d * nb).view(d, nb)

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


class Atlas:
    def __init__(self, hooks: dict[str, int], k: int = 20, n_bins: int = 256, device="cpu"):
        edges = asinh_edges(n_bins)
        self.hooks = dict(hooks)
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
        a["hist"].update(x)
        a["topk"].update(x, index)
        if "cov" in a:
            a["cov"].update(x)

    def merge(self, other: Atlas):
        assert self.hooks == other.hooks
        self.n_rows += other.n_rows
        for name, a in self.acc.items():
            for kind, acc in a.items():
                acc.merge(other.acc[name][kind])

    def state(self) -> dict:
        return {
            "hooks": self.hooks,
            "n_rows": self.n_rows,
            "acc": {name: {kind: acc.state() for kind, acc in a.items()} for name, a in self.acc.items()},
        }

    @classmethod
    def from_state(cls, state: dict, device="cpu") -> Atlas:
        classes = {"moments": Moments, "hist": Histogram, "topk": TopK, "cov": Covariance}
        obj = cls.__new__(cls)
        obj.hooks, obj.n_rows = state["hooks"], state["n_rows"]
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


def run(model, tokens: np.ndarray, start: int, stop: int, batch_size: int = 64, k: int = 20, device="cpu", progress=True) -> Atlas:
    from tqdm.auto import tqdm

    hooks = default_hooks(model.cfg)
    atlas = Atlas(hooks, k=k, device=device)
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


def export(atlas: Atlas, tokens: np.ndarray, tokenizer, out: Path):
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
    topk = topk[topk["row"] >= 0]  # unfilled slots, only possible with fewer than k tokens
    topk["token"], topk["context"] = contexts(tokens, tokenizer, topk["row"].to_numpy(), topk["pos"].to_numpy())
    neurons.to_parquet(out / "neurons.parquet", index=False)
    topk.sort_values(["hook", "unit", "rank"]).to_parquet(out / "topk.parquet", index=False)

    resid = [name for name in atlas.acc if "cov" in atlas.acc[name]]
    resid.sort(key=hook_layer)
    np.savez(
        out / "resid.npz",
        layer=np.array([hook_layer(n) for n in resid]),
        n=np.array([atlas.acc[n]["cov"].n for n in resid]),
        mean=np.stack([atlas.acc[n]["cov"].mean.numpy() for n in resid]),
        cov=np.stack([atlas.acc[n]["cov"].cov.numpy() for n in resid]),
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
    s.add_argument("--shard", type=int, default=0, help="which of the 80 OpenWebText Parquet files")
    s = sub.add_parser("run", help="accumulate statistics over a row range -> state-<start>-<stop>.pt")
    s.add_argument("dir", type=Path)
    s.add_argument("--rows", default=":", help="START:STOP row range (default: all)")
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--k", type=int, default=20)
    s.add_argument("--device", default=default_device())
    s.add_argument("--backend", choices=list(BACKENDS), default="hf", help="hf: plain transformers (fast); tl: TransformerLens bridge (reference)")
    s = sub.add_parser("export", help="merge state-*.pt -> parquet + npz")
    s.add_argument("dir", type=Path)
    args = p.parse_args(argv)
    args.dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "tokenize":
        t0 = time.time()
        tokens, docs = tokenize(owt_texts(args.shard), load_tokenizer(), int(args.n_tokens), args.seq_len)
        np.save(args.dir / "tokens.npy", tokens)
        np.save(args.dir / "docs.npy", docs)
        manifest = {
            "dataset": OWT_REPO, "shard": args.shard, "seq_len": args.seq_len, "n_rows": len(tokens),
            "n_docs": int(docs[-1, 0]) + 1, "tokenizer": "gpt2", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (args.dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{tokens.shape} tokens from {manifest['n_docs']} docs in {time.time() - t0:.0f}s -> {args.dir}")

    elif args.cmd == "run":
        tokens = np.load(args.dir / "tokens.npy", mmap_mode="r")
        start, stop = (int(x) if x else d for x, d in zip(args.rows.split(":"), (0, len(tokens))))
        t0 = time.time()
        atlas = run(load_model(args.device, args.backend), tokens, start, stop, args.batch_size, args.k, args.device)
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
        atlas = Atlas.from_state(states[0])
        for s in states[1:]:
            atlas.merge(Atlas.from_state(s))
        tokens = np.load(args.dir / "tokens.npy", mmap_mode="r")
        export(atlas, tokens, load_tokenizer(), args.dir)
        print(f"merged {len(paths)} state(s), {atlas.n_rows:,} rows, ranges {ranges} -> {args.dir}")


if __name__ == "__main__":
    main()
