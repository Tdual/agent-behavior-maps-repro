# -*- coding: utf-8 -*-
"""Free generation: how much does truncating the vocabulary distort alpha = 0?

With an admissible action list the policy is a distribution over ~11 actions
and the alpha = 0 coordinate is exact. Without one -- an agent that just emits
text -- the policy is a distribution over the whole vocabulary, and in practice
one only ever holds its top k. The question is what that truncation does to the
Bhattacharyya coefficient, since the alpha = 0 distance is built from it.

Measured here on the next-token distribution at each of the 193 reference
states: the model's free-generation policy for the first token of the action,
over all 151,936 vocabulary entries. One model, the nine harness agents. Token
temperature is applied to the logits, which is where it acts in free
generation (with an admissible list it acts on the action-level softmax).

For each pair and each state:

  exact      BC = sum_v sqrt(p_v q_v)                        full vocabulary
  truncated  keep the top k of each, renormalise each over its own support,
             BC over the union                               the practical case
  lumped     same, but the discarded mass is kept as one extra shared bin,
             which is the cheapest correction and needs only the top k

The error is reported on the distance that the map uses,
d = sqrt(mean_t 2(1 - BC)), not on BC itself.
"""
import itertools, json, os, sys, time
import numpy as np
import torch
import agent, close

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "close", "nexttok.npz")
KS = (5, 10, 20, 50, 100, 500, 1000, 5000)
TOPSAVE = 20000          # enough for the largest k, and for the lumped mass


@torch.no_grad()
def build():
    """Top-20000 of the next-token distribution at every reference state."""
    refs = json.load(open(os.path.join(HERE, "refs.json")))
    out = {}
    for h, cfg in close.HARNESS.items():
        a, t0 = agent.Agent(close.MODEL, **cfg), time.perf_counter()
        P, I = [], []
        for tr in refs:
            for st in tr["steps"]:
                pre = a._prefix_ids(agent.prompt_text(
                    tr["task"], tr["init"], [tuple(x) for x in st["hist"]],
                    st["admissible"], a.window))
                ids = torch.tensor([pre], device=agent.DEV)
                lg = a.model(input_ids=ids).logits[0, -1].float().cpu().numpy()
                P.append(lg.astype(np.float32))
                I.append(len(pre))
        a.close()
        L = np.stack(P)
        v, i = torch.topk(torch.from_numpy(L), TOPSAVE, dim=-1)
        out[h + "_logit"] = v.numpy().astype(np.float32)
        out[h + "_id"] = i.numpy().astype(np.int32)
        out[h + "_V"] = np.array([L.shape[1]])
        print(f"  {h:<4} {L.shape[0]}状態 語彙{L.shape[1]} "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    np.savez_compressed(CACHE, **out)


def dists(z, temp):
    """Full-support probabilities from the saved top-20000 logits.

    The discarded tail is negligible: its total probability is reported.
    """
    lg = z / temp
    lg = lg - lg.max(axis=1, keepdims=True)
    p = np.exp(lg.astype(np.float64))
    return p / p.sum(axis=1, keepdims=True)


def bc_exact(pi, ii, pj, ij):
    """BC over the union of the two saved supports, per state."""
    out = np.empty(len(pi))
    for t in range(len(pi)):
        m = dict(zip(ii[t], pi[t]))
        out[t] = sum(np.sqrt(m[v] * q) for v, q in zip(ij[t], pj[t]) if v in m)
    return out


def bc_trunc(pi, ii, pj, ij, k, lump):
    out = np.empty(len(pi))
    for t in range(len(pi)):
        a_id, a_p = ii[t][:k], pi[t][:k].copy()
        b_id, b_p = ij[t][:k], pj[t][:k].copy()
        ra, rb = 1.0 - a_p.sum(), 1.0 - b_p.sum()
        a_p = a_p / a_p.sum() if not lump else a_p
        b_p = b_p / b_p.sum() if not lump else b_p
        m = dict(zip(a_id, a_p))
        s = sum(np.sqrt(m[v] * q) for v, q in zip(b_id, b_p) if v in m)
        if lump:
            s += np.sqrt(max(ra, 0) * max(rb, 0))     # one shared tail bin
        out[t] = s
    return out


def dist(bc):
    return float(np.sqrt(np.mean(2.0 * (1.0 - np.clip(bc, 0, 1)))))


if __name__ == "__main__":
    if "--build" in sys.argv or not os.path.exists(CACHE):
        build()
    z = np.load(CACHE)
    V = int(z["h4_V"][0])
    P, I = {}, {}
    for n in close.NAMES:
        h = close.HA[n]
        P[n] = dists(z[h + "_logit"].astype(np.float64), close.TA[n])
        I[n] = z[h + "_id"]
    M = len(P[close.NAMES[0]])
    tail = 1.0 - np.mean([P[n].sum(axis=1).mean() for n in close.NAMES])
    print(f"語彙 {V}　状態 {M}　保存した上位 {TOPSAVE}　"
          f"捨てた裾の確率 平均 {tail:.2e}")

    pairs = list(itertools.combinations(close.NAMES, 2))
    ex = {p: dist(bc_exact(P[p[0]], I[p[0]], P[p[1]], I[p[1]])) for p in pairs}
    e = np.array([ex[p] for p in pairs])
    print(f"\n完全な語彙での &alpha;=0 距離: 中央値 {np.median(e):.4f}　"
          f"範囲 {e.min():.4f}–{e.max():.4f}")

    print(f"\n{'k':>6}{'打ち切り: 距離の相対誤差':>26}{'順位 Spearman':>16}"
          f"{'裾を1つの箱に: 相対誤差':>26}")
    from scipy.stats import spearmanr
    for k in KS:
        res = {}
        for lump in (False, True):
            d = np.array([dist(bc_trunc(P[a], I[a], P[b], I[b], k, lump))
                          for a, b in pairs])
            res[lump] = (float(np.median(np.abs(d - e) / e)),
                         float(spearmanr(d, e)[0]))
        print(f"{k:>6}{res[False][0]:>25.1%}{res[False][1]:>16.4f}"
              f"{res[True][0]:>25.1%}   （順位 {res[True][1]:.4f}）")
