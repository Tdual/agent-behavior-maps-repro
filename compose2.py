# -*- coding: utf-8 -*-
"""Three-component chains, and components that are independent by construction.

Two gaps in the first composition run:

  only two components   the paper also chains three
  only overlapping ones both of our components remove input information
                        (shorter history, weaker instruction), so they are
                        expected to interact; the paper's own low norm ratio
                        cases look the same way

Temperature fixes both at no cost. It acts on the admissible-set softmax, not
on the prompt, so it shares no mechanism with the window or the system prompt:
if two harness changes are ever independent, these are. And because the token
log-probabilities are untouched, every temperature agent is derived from a
harness that is already scored -- the three-component chains below need no
forward pass.

The raw coordinate cannot see temperature, so it is omitted; this is measured
in the norm coordinate (the policy) and at alpha = 0.
"""
import itertools, json, numpy as np
import close, close_compose as cc, alpha_compose as ac

BASE, EPS = "h4", 1e-300


def vec_norm(h, T=1.0):
    c = json.load(open(close.D("coords", f"{h}.json")))
    idx = c["idx"]
    v = np.array([np.log(close.agent.normalise(c["vecs"][k], T)[idx[k]] + EPS)
                  for k in range(len(idx))])
    return v - v.mean()


def vec_a0(h, T=1.0):
    c = json.load(open(close.D("coords", f"{h}.json")))
    M = len(c["vecs"])
    return np.concatenate([np.sqrt(close.agent.normalise(v, T))
                           for v in c["vecs"]]) / np.sqrt(M)


# combined agent, then its components as (harness, temperature) offsets from h4
CASES = [
    # two components, independent by construction: harness change + temperature
    ("2成分・独立", ("h2", 1.25), [("h2", 1.0), ("h4", 1.25)]),
    ("2成分・独立", ("h0", 1.25), [("h0", 1.0), ("h4", 1.25)]),
    ("2成分・独立", ("n4", 1.25), [("n4", 1.0), ("h4", 1.25)]),
    ("2成分・独立", ("s4", 1.25), [("s4", 1.0), ("h4", 1.25)]),
    ("2成分・独立", ("h2", 0.90), [("h2", 1.0), ("h4", 0.90)]),
    ("2成分・独立", ("n4", 0.90), [("n4", 1.0), ("h4", 0.90)]),
    # two components that share a mechanism (the first run's cases)
    ("2成分・重なり", ("n2", 1.0), [("h2", 1.0), ("n4", 1.0)]),
    ("2成分・重なり", ("s2", 1.0), [("h2", 1.0), ("s4", 1.0)]),
    ("2成分・重なり", ("n0", 1.0), [("h0", 1.0), ("n4", 1.0)]),
    ("2成分・重なり", ("s0", 1.0), [("h0", 1.0), ("s4", 1.0)]),
    # three components: window + prompt + temperature
    ("3成分", ("n2", 1.25), [("h2", 1.0), ("n4", 1.0), ("h4", 1.25)]),
    ("3成分", ("s2", 1.25), [("h2", 1.0), ("s4", 1.0), ("h4", 1.25)]),
    ("3成分", ("n0", 1.25), [("h0", 1.0), ("n4", 1.0), ("h4", 1.25)]),
    ("3成分", ("s0", 1.25), [("h0", 1.0), ("s4", 1.0), ("h4", 1.25)]),
    ("3成分", ("n2", 0.90), [("h2", 1.0), ("n4", 1.0), ("h4", 0.90)]),
    ("3成分", ("s0", 0.90), [("h0", 1.0), ("s4", 1.0), ("h4", 0.90)]),
]


def run(get, label):
    b = get(BASE, 1.0)
    rows = {}
    print(f"\n--- {label} ---")
    print(f"{'種類':<14}{'組合せ':<30}{'cos':>8}{'ノルム比':>10}{'相対誤差':>10}")
    for kind, (ch, ct), parts in CASES:
        dc = get(ch, ct) - b
        dp = sum((get(h, t) - b) for h, t in parts)
        cos, r, re_ = ac.triple(dc, dp)
        rows.setdefault(kind, []).append((cos, r, re_))
        name = f"{ch}@T{ct} = " + " + ".join(
            (h if t == 1.0 else f"T{t}") for h, t in parts)
        print(f"{kind:<14}{name:<30}{cos:>8.3f}{r:>10.3f}{re_:>10.3f}")
    print(f"\n{'種類':<14}{'件数':>6}{'cos':>10}{'ノルム比':>10}{'相対誤差':>10}  （中央値）")
    for k, v in rows.items():
        m = np.median(np.array(v), axis=0)
        print(f"{k:<14}{len(v):>6}{m[0]:>10.3f}{m[1]:>10.3f}{m[2]:>10.3f}")
    return rows


if __name__ == "__main__":
    run(vec_norm, "α=1 norm 座標")
    run(vec_a0, "α=0 √π 座標")
    print("\n論文（2成分・中央値）: τ-bench 0.90 / 0.98 / 0.46、"
          "ALFWorld 0.63 / 0.89 / 0.97")
    print("論文（3成分）: 24/12/12 ケースの連鎖")
