"""Helpers for an IOI + Hydra-effect study of GPT-2 small with TransformerLens 4.x.

Conventions
-----------
- Logit difference (LD) = logit(IO) - logit(S) at the final token. Positive = model prefers IO.
- "clean" prompts are IOI prompts; "corrupt" prompts are the matched ABC prompts, where all three
  name slots are replaced by fresh, distinct names (same template, so same token length).
- Everything is per-prompt where possible, so that we can bootstrap over prompts.
- Requires `model.enable_compatibility_mode()` (folded LayerNorm), so that the final LayerNorm is
  just "center, then divide by scale" and direct effects are linear given the scale.

References
----------
- Wang et al. 2022, Interpretability in the Wild (IOI): https://arxiv.org/abs/2211.00593
- McGrath et al. 2023, The Hydra Effect: https://arxiv.org/abs/2307.15771
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------

# Candidate names; filtered at runtime to those that are a single GPT-2 token with a leading space.
NAMES = [
    "Mary", "John", "Alice", "Bob", "Tom", "James", "Dan", "Sarah", "Michael", "David",
    "Paul", "Mark", "Lisa", "Anna", "Emma", "Kevin", "Jessica", "Andrew", "Richard", "Laura",
    "Steven", "Brian", "Rachel", "Jennifer", "Robert", "William", "Thomas", "Daniel", "Chris",
    "Matthew", "Joseph", "Charles", "George", "Edward", "Peter", "Eric", "Adam", "Amy", "Kate",
    "Sam", "Ben", "Jack", "Joe", "Mike", "Ryan", "Jason", "Justin", "Brandon", "Anthony",
    "Jane", "Ann", "Helen", "Linda", "Karen", "Susan", "Nancy", "Betty", "Martin", "Frank",
]

# [A] and [B] are the first two names, [S] is the repeated subject. The answer (IO) is omitted.
TEMPLATES = [
    "When [A] and [B] went to the store, [S] gave a drink to",
    "When [A] and [B] went to the park, [S] gave a ball to",
    "Then, [A] and [B] went to the office. [S] gave a book to",
    "After [A] and [B] went to the restaurant, [S] gave a ring to",
    "While [A] and [B] were working at the school, [S] gave a computer to",
    "Friends [A] and [B] found a bone at the garden. [S] gave it to",
    "Then, [A] and [B] had a long argument, and afterwards [S] said to",
    "The local big house where [A] and [B] lived was quiet. [S] handed a key to",
]


@dataclass
class IOIDataset:
    prompts: list[str]
    corrupt_prompts: list[str]
    clean_tokens: torch.Tensor  # [N, T], right-padded
    corrupt_tokens: torch.Tensor  # [N, T]
    io: torch.Tensor  # [N] token id of the indirect object (correct answer)
    s: torch.Tensor  # [N] token id of the subject (wrong answer)
    end: torch.Tensor  # [N] index of the final prompt token
    template_id: torch.Tensor  # [N]
    order: list[str]  # "ABBA" or "BABA"
    pos: dict[str, torch.Tensor] = field(default_factory=dict)  # positions of IO, S1, S2, end

    def __len__(self) -> int:
        return len(self.prompts)

    def subset(self, mask: torch.Tensor) -> "IOIDataset":
        idx = torch.nonzero(mask).squeeze(-1)
        T = int(self.end[idx].max()) + 1
        return IOIDataset(
            prompts=[self.prompts[i] for i in idx],
            corrupt_prompts=[self.corrupt_prompts[i] for i in idx],
            clean_tokens=self.clean_tokens[idx, :T],
            corrupt_tokens=self.corrupt_tokens[idx, :T],
            io=self.io[idx],
            s=self.s[idx],
            end=self.end[idx],
            template_id=self.template_id[idx],
            order=[self.order[i] for i in idx],
            pos={k: v[idx] for k, v in self.pos.items()},
        )


def single_token_names(model) -> list[str]:
    return [n for n in NAMES if len(model.to_tokens(" " + n, prepend_bos=False)[0]) == 1]


def _fill(template: str, a: str, b: str, s: str) -> str:
    return template.replace("[A]", a).replace("[B]", b).replace("[S]", s)


def make_ioi_dataset(model, n_per_template: int = 16, templates=TEMPLATES, seed: int = 0) -> IOIDataset:
    """Build balanced ABBA/BABA IOI prompts plus matched ABC corrupted prompts."""
    rng = random.Random(seed)
    names = single_token_names(model)
    prompts, corrupt, io_names, s_names, tids, orders = [], [], [], [], [], []
    for tid, tmpl in enumerate(templates):
        for i in range(n_per_template):
            io, s, c1, c2, c3 = rng.sample(names, 5)
            if i % 2 == 0:  # ABBA: IO first, then S, then S again
                prompts.append(_fill(tmpl, io, s, s))
                orders.append("ABBA")
            else:  # BABA: S first, then IO, then S again
                prompts.append(_fill(tmpl, s, io, s))
                orders.append("BABA")
            corrupt.append(_fill(tmpl, c1, c2, c3))
            io_names.append(io)
            s_names.append(s)
            tids.append(tid)

    clean = model.to_tokens(prompts, padding_side="right")
    corr = model.to_tokens(corrupt, padding_side="right")
    assert clean.shape == corr.shape
    end = torch.tensor([len(model.to_tokens(p)[0]) - 1 for p in prompts])
    io = torch.tensor([model.to_single_token(" " + n) for n in io_names])
    s = torch.tensor([model.to_single_token(" " + n) for n in s_names])

    # Positions of the name tokens in each clean prompt.
    def first(row, tok, start=0):
        return start + int(torch.nonzero(row[start:] == tok)[0])

    pos_io, pos_s1, pos_s2 = [], [], []
    for b in range(len(prompts)):
        row = clean[b]
        pos_io.append(first(row, io[b]))
        p1 = first(row, s[b])
        pos_s1.append(p1)
        pos_s2.append(first(row, s[b], p1 + 1))
    pos = {
        "IO": torch.tensor(pos_io),
        "S1": torch.tensor(pos_s1),
        "S2": torch.tensor(pos_s2),
        "end": end,
    }
    return IOIDataset(prompts, corrupt, clean, corr, io, s, end, torch.tensor(tids), orders, pos)


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------


def logit_diff(logits: torch.Tensor, ds: IOIDataset) -> torch.Tensor:
    """Per-prompt logit(IO) - logit(S) at the final token. Returns [N]."""
    b = torch.arange(len(ds))
    final = logits[b, ds.end.to(logits.device)]
    return (final[b, ds.io] - final[b, ds.s]).float().cpu()


def normalized(ld: torch.Tensor, clean_ld: float, corrupt_ld: float) -> torch.Tensor:
    """0 = corrupt performance, 1 = clean performance."""
    return (ld - corrupt_ld) / (clean_ld - corrupt_ld)


def bootstrap_ci(*xs, stat=lambda x: x.mean(), n_boot: int = 5000, ci: float = 0.95, seed: int = 0):
    """Percentile bootstrap over prompts. Arrays in `xs` are resampled with the *same* indices, so
    paired statistics (e.g. ratio of two per-prompt means) are handled correctly.

    Returns (point_estimate, lo, hi).
    """
    xs = [np.asarray(x, dtype=float) for x in xs]
    n = len(xs[0])
    rng = np.random.default_rng(seed)
    point = stat(*xs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = stat(*[x[idx] for x in xs])
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(point), float(lo), float(hi)


def fmt_ci(t, digits: int = 2) -> str:
    p, lo, hi = t
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ---------------------------------------------------------------------------------------------
# Activation patching (denoising: clean activations patched into the corrupt run)
# ---------------------------------------------------------------------------------------------


def patch_resid_by_pos(model, ds: IOIDataset, clean_cache, clean_ld: float, corrupt_ld: float):
    """Patch clean resid_pre[layer, pos] into the corrupt run. Use a single-template dataset so that
    positions line up across prompts. Returns normalized LD, shape [n_layers, T]."""
    L, T = model.cfg.n_layers, ds.clean_tokens.shape[1]
    out = torch.zeros(L, T)
    for layer in range(L):
        name = f"blocks.{layer}.hook_resid_pre"
        clean_act = clean_cache[name]
        for p in range(T):

            def hook(x, hook, p=p):
                x[:, p] = clean_act[:, p]
                return x

            logits = model.run_with_hooks(ds.corrupt_tokens, fwd_hooks=[(name, hook)])
            out[layer, p] = normalized(logit_diff(logits, ds), clean_ld, corrupt_ld).mean()
    return out


def patch_heads(model, ds: IOIDataset, clean_cache, clean_ld: float, corrupt_ld: float):
    """Patch each head's clean output (hook_z, all positions) into the corrupt run.
    Returns normalized LD per prompt, shape [N, n_layers, n_heads]."""
    L, H = model.cfg.n_layers, model.cfg.n_heads
    out = torch.zeros(len(ds), L, H)
    for layer in range(L):
        name = f"blocks.{layer}.attn.hook_z"
        clean_z = clean_cache[name]
        for h in range(H):

            def hook(x, hook, h=h):
                x[:, :, h] = clean_z[:, :, h]
                return x

            logits = model.run_with_hooks(ds.corrupt_tokens, fwd_hooks=[(name, hook)])
            out[:, layer, h] = normalized(logit_diff(logits, ds), clean_ld, corrupt_ld)
    return out


# ---------------------------------------------------------------------------------------------
# Direct logit attribution
# ---------------------------------------------------------------------------------------------


def head_direct_effects(model, cache, ds: IOIDataset) -> torch.Tensor:
    """Per-prompt direct effect of each head on the logit difference, using *this run's* final
    LayerNorm scale. Returns [N, n_layers, n_heads].

    Since compatibility mode folds the LayerNorm weights into W_U, the final LN is
    (x - mean(x)) / scale, which is linear in x given the scale. So each head's contribution
    to the LD is (head_out - mean(head_out)) / scale, dotted with (W_U[:, IO] - W_U[:, S])."""
    b = torch.arange(len(ds))
    end = ds.end
    z = torch.stack(
        [cache[f"blocks.{l}.attn.hook_z"][b, end] for l in range(model.cfg.n_layers)], dim=1
    )  # [N, L, H, d_head]
    head_out = torch.einsum("nlhd,lhdm->nlhm", z, model.W_O)
    head_out = head_out - head_out.mean(-1, keepdim=True)
    scale = cache["ln_final.hook_scale"][b, end]  # [N, 1]
    direction = (model.W_U[:, ds.io] - model.W_U[:, ds.s]).T  # [N, d_model]
    de = torch.einsum("nlhm,nm->nlh", head_out, direction) / scale[:, :, None]
    return de.float().cpu()


def attention_to(cache, ds: IOIDataset, layer: int, head: int, to: str, frm: str = "end") -> torch.Tensor:
    """Attention probability from position `frm` to position `to` (keys of ds.pos). Returns [N]."""
    b = torch.arange(len(ds))
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # [N, H, Q, K]
    return pattern[b, head, ds.pos[frm], ds.pos[to]].float().cpu()


# ---------------------------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------------------------

ABLATION_KINDS = ("zero", "mean", "resample")


class Ablator:
    """Builds hook_z ablation hooks for sets of heads.

    - zero:     z_h <- 0
    - mean:     z_h <- mean of z_h over the ABC (corrupt) prompts *of the same template*, per
                position (as in the IOI paper)
    - resample: z_h <- z_h from the matched ABC prompt
    """

    def __init__(self, model, ds: IOIDataset, corrupt_cache):
        self.model, self.ds = model, ds
        self.corrupt_z = {l: corrupt_cache[f"blocks.{l}.attn.hook_z"] for l in range(model.cfg.n_layers)}
        self.mean_z = {}
        for l, z in self.corrupt_z.items():
            m = torch.zeros_like(z)
            for tid in ds.template_id.unique():
                mask = (ds.template_id == tid).to(z.device)
                m[mask] = z[mask].mean(0, keepdim=True)
            self.mean_z[l] = m

    def hooks(self, heads, kind: str):
        assert kind in ABLATION_KINDS, kind
        by_layer: dict[int, list[int]] = {}
        for l, h in heads:
            by_layer.setdefault(l, []).append(h)
        fwd_hooks = []
        for l, hs in by_layer.items():
            ref = None if kind == "zero" else (self.mean_z[l] if kind == "mean" else self.corrupt_z[l])

            def hook(x, hook, hs=hs, ref=ref):
                for h in hs:
                    x[:, :, h] = 0.0 if ref is None else ref[:, :, h]
                return x

            fwd_hooks.append((f"blocks.{l}.attn.hook_z", hook))
        return fwd_hooks

    def run(self, heads, kind: str, with_cache: bool = False):
        hooks = self.hooks(heads, kind)
        if not with_cache:
            return self.model.run_with_hooks(self.ds.clean_tokens, fwd_hooks=hooks), None
        with self.model.hooks(fwd_hooks=hooks):
            return self.model.run_with_cache(self.ds.clean_tokens)


def hydra(ablator: Ablator, heads, kind: str, clean_ld: torch.Tensor, de_clean: torch.Tensor):
    """Ablate `heads` and compare the actual drop in LD with the drop predicted from their direct
    effects alone. The gap is downstream compensation (self-repair), plus LayerNorm rescaling.

    Returns a dict of per-prompt tensors:
      total    [N]        clean LD - ablated LD (what actually happened)
      direct   [N]        sum over ablated heads of (DE_clean - DE_ablated)  (the naive prediction)
      repair   [N]        direct - total  (> 0 means the network compensated)
      delta_de [N, L, H]  DE_ablated - DE_clean for every head (who did the compensating)
    """
    logits, cache = ablator.run(heads, kind, with_cache=True)
    ld = logit_diff(logits, ablator.ds)
    de_abl = head_direct_effects(ablator.model, cache, ablator.ds)
    delta = de_abl - de_clean
    direct = -sum(delta[:, l, h] for l, h in heads)
    total = clean_ld - ld
    return {"ld": ld, "total": total, "direct": direct, "repair": direct - total, "delta_de": delta}


def additivity(ablator: Ablator, heads, kind: str, clean_ld: torch.Tensor):
    """Single-head ablation drops vs the joint ablation drop. Returns per-prompt tensors:
    singles [N, k], joint [N]. Additive iff joint == singles.sum(-1)."""
    singles = torch.stack(
        [clean_ld - logit_diff(ablator.run([hd], kind)[0], ablator.ds) for hd in heads], dim=-1
    )
    joint = clean_ld - logit_diff(ablator.run(heads, kind)[0], ablator.ds)
    return {"singles": singles, "joint": joint}


# ---------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------


def heatmap(arr, title="", xlabel="head", ylabel="layer", xticklabels=None, ax=None, vmax=None, annotate_top=0):
    arr = np.asarray(arr)
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    vmax = vmax or np.abs(arr).max()
    im = ax.imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)), xticklabels, rotation=90)
    if annotate_top:
        flat = np.argsort(-np.abs(arr), axis=None)[:annotate_top]
        for i, j in zip(*np.unravel_index(flat, arr.shape)):
            ax.text(j, i, f"{i}.{j}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax)
    return ax


def top_heads(arr, k=10, largest=True):
    """[(layer, head, value)] sorted by value."""
    arr = torch.as_tensor(arr)
    vals, idx = arr.flatten().topk(k, largest=largest)
    H = arr.shape[1]
    return [(int(i) // H, int(i) % H, float(v)) for v, i in zip(vals, idx)]
