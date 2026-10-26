"""The atlas only scales if merging is exact: shard A + shard B must equal one pass over A + B."""

import numpy as np
import pytest
import torch

from atlas import (BOS, Atlas, Covariance, Histogram, Moments, TopK, asinh_edges, bootstrap_weights, hist_mask,
                   pooled_cov_summaries, pooled_moments, shard_ranges, tokenize)


@pytest.fixture
def x():
    g = torch.Generator().manual_seed(0)
    # Heavy-tailed, offset data: a large mean stresses the variance merge numerically.
    return 100 + torch.randn(5000, 7, generator=g) * torch.logspace(-2, 2, 7) + torch.randn(5000, 1, generator=g) ** 3


def chunks(x, sizes=(1, 999, 2000, 2000)):
    return torch.split(x, list(sizes))


def test_moments_sequential_and_merged_match_direct(x):
    seq = Moments(x.shape[1])
    for c in chunks(x):
        seq.update(c)
    a, b = Moments(x.shape[1]), Moments(x.shape[1])
    a.update(x[:1234])
    b.update(x[1234:])
    a.merge(b)
    xd = x.double()
    for m in (seq, a):
        assert m.n == len(x)
        torch.testing.assert_close(m.mean, xd.mean(0))
        torch.testing.assert_close(m.var, xd.var(0, unbiased=False), rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(m.max, xd.max(0).values)
        assert torch.equal(m.n_pos, (x > 0).sum(0))


def test_covariance_matches_numpy(x):
    c = Covariance(x.shape[1])
    for ch in chunks(x):
        c.update(ch)
    ref = np.cov(x.double().numpy(), rowvar=False, bias=True)
    np.testing.assert_allclose(c.cov.numpy(), ref, rtol=1e-5, atol=1e-8)


def test_topk_matches_full_topk(x):
    index = torch.arange(len(x)) * 10  # arbitrary global indices
    t = TopK(x.shape[1], k=5)
    for ch, ix in zip(chunks(x), chunks(index)):
        t.update(ch, ix)
    v, i = x.topk(5, dim=0)
    torch.testing.assert_close(t.values, v)
    assert torch.equal(t.index, index[i])


def test_histogram_counts_and_quantiles(x):
    edges = asinh_edges(512)
    h = Histogram(x.shape[1], edges)
    for ch in chunks(x):
        h.update(ch)
    assert (h.counts.sum(1) == len(x)).all()
    # Interior bins agree with numpy (which uses [lo, hi) bins; ties at edges are vanishingly rare).
    ref, _ = np.histogram(x[:, 3].double().numpy(), bins=edges.numpy())
    assert np.abs(h.counts[3, 1:-1].numpy() - ref).sum() <= 2
    # Quantiles are interpolated within a bin, so they're accurate to the bin width (~6% here).
    q = h.quantiles([0.5, 0.99])
    ref = np.quantile(x.double().numpy(), [0.5, 0.99], axis=0).T
    np.testing.assert_allclose(q, ref, rtol=0.06)


@pytest.mark.parametrize("stride", [1, 4])
def test_atlas_state_round_trip_and_merge(x, tmp_path, stride):
    hooks = {"blocks.0.mlp.hook_post": 7, "blocks.0.hook_resid_pre": 7}
    act = x[:4800].reshape(48, 100, 7)  # [B, T, d]
    index = (torch.arange(48)[:, None] * 100 + torch.arange(1, 100)).flatten()

    # Split at row 7: 7 * 99 tokens is not a multiple of the stride, so the subsample must not depend on batching.
    whole = Atlas(hooks, k=3, hist_stride=stride)
    a, b = Atlas(hooks, k=3, hist_stride=stride), Atlas(hooks, k=3, hist_stride=stride)
    for name in hooks:
        whole.update(name, act, index)
        a.update(name, act[:7], index[: 7 * 99])
        b.update(name, act[7:], index[7 * 99 :])
    torch.save(b.state(), tmp_path / "b.pt")
    a.merge(Atlas.from_state(torch.load(tmp_path / "b.pt")))

    for name in hooks:
        for kind in whole.acc[name]:
            sw, sa = whole.acc[name][kind].state(), a.acc[name][kind].state()
            for key in sw:
                if isinstance(sw[key], torch.Tensor):
                    # Batches are reduced in float32, so split vs. whole differ at the 1e-7 level.
                    torch.testing.assert_close(sa[key], sw[key], rtol=1e-5, atol=1e-6)
                else:
                    assert sa[key] == sw[key]
    # Position 0 (BOS) is excluded; histograms see 1 in `stride` of the remaining tokens.
    assert whole.acc["blocks.0.mlp.hook_post"]["moments"].n == 48 * 99
    assert whole.acc["blocks.0.mlp.hook_post"]["hist"].counts.sum(1).eq(-(-48 * 99 // stride)).all()


def test_hist_mask_cycles_through_positions():
    T, rows, stride = 128, 400, 4
    index = (torch.arange(rows)[:, None] * T + torch.arange(1, T)).flatten()
    m = hist_mask(index, T, stride)
    assert m.sum() == rows * (T - 1) // stride
    per_pos = m.view(rows, T - 1).sum(0)  # how often each position 1..127 is selected
    assert per_pos.min() >= rows // stride - 1 and per_pos.max() <= rows // stride + 1


def test_tokenize_rows_are_bos_plus_one_document_window():
    class FakeTokenizer:
        def __call__(self, texts):
            return {"input_ids": [list(range(int(t))) for t in texts]}

    tokens, docs = tokenize(["10", "2", "8"], FakeTokenizer(), n_tokens=12, seq_len=4)
    # doc 0 (10 tokens) -> windows at 0, 3, 6 (partial tail dropped); doc 1 too short; doc 2 -> 0.
    assert tokens.tolist() == [[BOS, 0, 1, 2], [BOS, 3, 4, 5], [BOS, 6, 7, 8]]
    assert docs.tolist() == [[0, 0], [0, 3], [0, 6]]
    tokens, docs = tokenize(["10", "2", "8"], FakeTokenizer(), n_tokens=16, seq_len=4)
    assert docs.tolist()[-1] == [2, 0]


def test_tokenize_caps_windows_per_document():
    class FakeTokenizer:
        def __call__(self, texts):
            return {"input_ids": [list(range(int(t))) for t in texts]}

    tokens, docs = tokenize(["30", "2", "30", "30"], FakeTokenizer(), n_tokens=12, seq_len=4, max_windows_per_doc=1, seed=0)
    assert docs[:, 0].tolist() == [0, 2, 3]  # one window per document; doc 1 is too short
    assert all(off % 3 == 0 and off <= 27 for off in docs[:, 1])
    again, _ = tokenize(["30", "2", "30", "30"], FakeTokenizer(), n_tokens=12, seq_len=4, max_windows_per_doc=1, seed=0)
    assert np.array_equal(tokens, again)  # seeded: reproducible


def test_shard_ranges_respect_files():
    files = np.array([0] * 10 + [1] * 6)
    assert shard_ranges(files, 4) == [(0, 5), (5, 10), (10, 13), (13, 16)]
    with pytest.raises(AssertionError):
        shard_ranges(files, 3)


def test_bootstrap_weights_resample_within_groups():
    w = bootstrap_weights([0, 0, 0, 1, 1], n_boot=200)
    assert (w[:, :3].sum(1) == 3).all() and (w[:, 3:].sum(1) == 2).all()


def test_pooled_moments_and_cov_match_exact_merge(x):
    shards = torch.split(x, [1000, 1500, 1200, 1300])
    ms, cs = [Moments(x.shape[1]) for _ in shards], [Covariance(x.shape[1]) for _ in shards]
    for m, c, sh in zip(ms, cs, shards):
        m.update(sh)
        c.update(sh)
    n = np.array([float(m.n) for m in ms])
    stack = lambda f: np.stack([f(m) for m in ms])
    reps = pooled_moments(np.ones((1, 4)), n, stack(lambda m: m.mean.numpy()), stack(lambda m: m.m2.numpy()),
                          stack(lambda m: m.n_pos.double().numpy()))
    xd = x.double().numpy()
    np.testing.assert_allclose(reps["mean"][0], xd.mean(0), rtol=1e-6)
    np.testing.assert_allclose(reps["std"][0], xd.std(0), rtol=1e-5)
    np.testing.assert_allclose(reps["frac_pos"][0], (xd > 0).mean(0))
    total, pr = pooled_cov_summaries(np.ones((1, 4)), n, np.stack([c.mean.numpy() for c in cs]), np.stack([c.c.numpy() for c in cs]))
    lam = np.linalg.eigvalsh(np.cov(xd, rowvar=False, bias=True))
    np.testing.assert_allclose(total[0], lam.sum(), rtol=1e-5)
    np.testing.assert_allclose(pr[0], lam.sum() ** 2 / (lam**2).sum(), rtol=1e-5)
