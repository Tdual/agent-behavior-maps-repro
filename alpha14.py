# -*- coding: utf-8 -*-
"""Full policy vectors for the six separate LLMs, so alpha = 0 covers all 14 agents.

The first run stored only the value of the action the reference took: one
number per step, which is all the paper's alpha = 1 coordinate needs. The
alpha = 0 embedding is sqrt(pi), so it needs the whole distribution over the
admissible set at every step -- the same quantity the forward pass already
produced, discarded on the way out.

This re-scores the five remaining LLMs under the BASE harness (window 4, the
base system prompt) on the same refs.json. `qwen05-it` is not re-run: it is by
construction agent `h4`, whose vectors are already on disk, and the first run
checked the two against each other to the last bit.
"""
import json, os, time
import numpy as np
import agent, close, models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "coords_full")
SKIP = {"qwen05-it": "h4"}          # same agent as the harness run's h4


def score_all():
    refs = json.load(open(os.path.join(HERE, "refs.json")))
    os.makedirs(OUT, exist_ok=True)
    for name, mid in models.MODELS:
        if name in SKIP:
            print(f"  {name:<12} {SKIP[name]} と同一のため再計算しない"); continue
        p = os.path.join(OUT, f"{name}.json")
        if os.path.exists(p):
            print(f"  {name:<12} キャッシュ済み"); continue
        a, t0 = agent.Agent(mid), time.perf_counter()
        vecs, idx, tid = [], [], []
        for ti, tr in enumerate(refs):
            for st in tr["steps"]:
                sc = a.score(tr["task"], tr["init"],
                             [tuple(x) for x in st["hist"]], st["admissible"])
                vecs.append(sc.tolist())
                idx.append(st["admissible"].index(st["action"]))
                tid.append(ti)
        a.close()
        json.dump({"vecs": vecs, "idx": idx, "tid": tid}, open(p, "w"))
        print(f"  {name:<12} {len(vecs)}ステップ {time.perf_counter()-t0:.0f}s", flush=True)


def check():
    """The stored chosen-action value must reproduce coords/<name>.npz exactly."""
    worst = 0.0
    for name, _ in models.MODELS:
        if name in SKIP:
            continue
        c = json.load(open(os.path.join(OUT, f"{name}.json")))
        raw = np.array([c["vecs"][k][c["idx"][k]] for k in range(len(c["idx"]))])
        old = np.load(os.path.join(HERE, "coords", f"{name}.npz"))["raw"].astype(float)
        d = float(np.abs(raw - old).max())
        worst = max(worst, d)
        print(f"  {name:<12} 既存の raw 座標との最大差 {d:.3e}")
    return worst


def policies():
    """pi(.|h) at every reference step for all 14 agents."""
    pol = {}
    for name, _ in models.MODELS:
        key = SKIP.get(name)
        c = json.load(open(close.D("coords", f"{key}.json") if key
                           else os.path.join(OUT, f"{name}.json")))
        pol[name] = [agent.normalise(v, 1.0) for v in c["vecs"]]
    C = {h: json.load(open(close.D("coords", f"{h}.json"))) for h in close.HARNESS}
    for n in close.NAMES:
        if n == "h4":
            continue                      # == qwen05-it, already in
        pol[n] = [agent.normalise(v, close.TA[n]) for v in C[close.HA[n]]["vecs"]]
    return pol


if __name__ == "__main__":
    score_all()
    print("\n検算（新しい全ベクトルから選ばれた行動の値を取り出す）")
    w = check()
    print(f"  最大差 {w:.3e}  -> {'一致' if w < 1e-9 else '不一致'}")
    P = policies()
    print(f"\n方策 {len(P)}体ぶん・各 {len(P[list(P)[0]])}ステップ")


def _src(name):
    key = SKIP.get(name)
    if key:
        return json.load(open(close.D("coords", f"{key}.json")))
    if name in close.NAMES:
        return json.load(open(close.D("coords", f"{close.HA[name]}.json")))
    return json.load(open(os.path.join(OUT, f"{name}.json")))


def chosen_index(name):
    """Index of the action the reference took, per step. Shared by all agents."""
    return _src(name)["idx"]


def raw_vectors(names=None):
    """The paper's raw coordinate: summed token log-prob of the recorded action."""
    out = {}
    for name, _ in models.MODELS:
        c = _src(name)
        out[name] = np.array([c["vecs"][k][c["idx"][k]] for k in range(len(c["idx"]))])
    for n in close.NAMES:
        c = _src(n)
        out[n] = np.array([c["vecs"][k][c["idx"][k]] for k in range(len(c["idx"]))])
    return out
