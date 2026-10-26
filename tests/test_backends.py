"""HFGPT2 must reproduce the TransformerLens bridge (compatibility mode) at every hook point it offers.

Downloads GPT-2 small on first use and runs on CPU, about 10 s.
"""

import numpy as np
import pytest
import torch

from atlas import BOS, HFGPT2, TLGPT2, _Stop, default_hooks, run

HOOKS = [
    f"blocks.{l}.{h}"
    for l in (0, 5, 11)
    for h in ("hook_resid_pre", "hook_resid_mid", "hook_resid_post", "hook_attn_out", "hook_mlp_out",
              "mlp.hook_pre", "mlp.hook_post")
]
TEXT = ("When Mary and John went to the store, John gave a drink to Mary. The store was closed on Sunday, "
        "so they walked to the park instead and talked about the weather for an hour.")


@pytest.fixture(scope="module")
def models():
    return HFGPT2("cpu"), TLGPT2("cpu")


@pytest.fixture(scope="module")
def tokens(models):
    from transformers import AutoTokenizer

    text = AutoTokenizer.from_pretrained("gpt2")(TEXT)["input_ids"][:32]
    rand = torch.randint(0, 50257, (3, 31), generator=torch.Generator().manual_seed(0)).tolist()
    return torch.tensor([[BOS] + text[:31]] + [[BOS] + r for r in rand])  # [4, 32]: one real, three random


def capture(model, tokens, names):
    acts = {}
    model.run(tokens, {n: (lambda act, n=n: acts.__setitem__(n, act.clone())) for n in names})
    return acts


def test_cfg_matches(models):
    hf, tl = models
    assert (hf.cfg.n_layers, hf.cfg.d_model, hf.cfg.d_mlp) == (tl.cfg.n_layers, tl.cfg.d_model, tl.cfg.d_mlp)
    assert default_hooks(hf.cfg) == default_hooks(tl.cfg)


@pytest.mark.parametrize("name", HOOKS)
def test_every_hook_matches_bridge(models, tokens, name):
    hf, tl = models
    a, b = capture(hf, tokens, [name])[name], capture(tl, tokens, [name])[name]
    assert a.shape == b.shape
    # Same math, different op order: agreement to float32 rounding relative to the activation scale.
    # BOS included: its outlier norm is the hardest case.
    assert (a - b).abs().max() <= 2e-5 * b.abs().max() + 1e-5, (a - b).abs().max()


def test_residual_needs_centering(models, tokens):
    """Guards the one convention difference: raw HF residuals differ from the bridge by their mean."""
    hf, tl = models
    name = "blocks.5.hook_resid_pre"
    raw = {}
    h = hf.model.h[5].register_forward_pre_hook(lambda m, args, kw: raw.__setitem__("x", args[0] if args else kw["hidden_states"]), with_kwargs=True)
    try:
        hf.model(tokens)
    finally:
        h.remove()
    ref = capture(tl, tokens, [name])[name]
    assert (raw["x"] - ref).abs().max() > 1e-2
    torch.testing.assert_close(raw["x"] - raw["x"].mean(-1, keepdim=True), ref, rtol=1e-4, atol=1e-4)


def test_stop_skips_later_hooks(models, tokens):
    hf, _ = models
    seen = []

    def stop(act):
        seen.append("mlp0")
        raise _Stop

    hf.run(tokens, {"blocks.0.mlp.hook_post": stop, "blocks.1.hook_resid_pre": lambda act: seen.append("resid1")})
    assert seen == ["mlp0"]


def test_unknown_hook_raises(models, tokens):
    with pytest.raises(KeyError):
        models[0].run(tokens, {"blocks.0.attn.hook_z": lambda act: None})


def test_run_gives_same_atlas(models, tokens):
    hf, tl = models
    toks = tokens.numpy().astype(np.uint16)
    a, b = run(hf, toks, 0, len(toks), batch_size=2, k=5, progress=False), run(tl, toks, 0, len(toks), batch_size=2, k=5, progress=False)
    for name in a.acc:
        ma, mb = a.acc[name]["moments"], b.acc[name]["moments"]
        assert ma.n == mb.n == 4 * 31
        scale = mb.max.abs().max()
        assert (ma.mean - mb.mean).abs().max() <= 2e-5 * scale
        assert (ma.var.sqrt() - mb.var.sqrt()).abs().max() <= 2e-5 * scale
        torch.testing.assert_close(a.acc[name]["topk"].values, b.acc[name]["topk"].values, rtol=1e-4, atol=2e-5 * scale.item())
        # Values within rounding of a bin edge may land in the neighbouring bin.
        ca, cb = a.acc[name]["hist"].counts, b.acc[name]["hist"].counts
        assert (ca - cb).abs().sum() <= 2e-3 * cb.sum()
