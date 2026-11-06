# -*- coding: utf-8 -*-
"""Why every agent scored 0 of 8.

Three candidates: the tasks are unsolvable inside the step cap, the cap is too
short, or the policy is simply too weak. The first two are checked against the
oracle; the third is read off the probability the agent puts on the oracle's
action at the states the oracle visits, which bounds the success probability
of the whole episode from below in the only way that matters here -- a
successful episode has to pick a correct action at every step.
"""
import json, os
import numpy as np
import agent, close, env

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPS = (0.2, 0.3, 0.5, 0.7, 1.0, 1.25)
N_SEED = 300


def oracle_lengths():
    ls = [len(env.oracle(env.Env(close.EVAL_SEED0 + k))) for k in range(close.N_EVAL)]
    wide = [len(env.oracle(env.Env(s))) for s in range(N_SEED)]
    h = {}
    for L in wide:
        h[L] = h.get(L, 0) + 1
    return ls, dict(sorted(h.items())), wide


def oracle_prob():
    """pi(oracle action | oracle history) at every step of every eval task."""
    a = close.load("h4")
    out = []
    for k in range(close.N_EVAL):
        e = env.Env(close.EVAL_SEED0 + k)
        init, task = e.observe(), e.task()
        plan, hist = env.oracle(env.Env(close.EVAL_SEED0 + k)), []
        for act in plan:
            adm = e.admissible()
            if act not in adm:
                break
            sc = a.score(task, init, hist, adm)
            out.append({"raw": sc.tolist(), "j": adm.index(act),
                            "n": len(adm), "task": k})
            hist.append((e.observe(), act))
            e.step(act)
    a.close()
    return out


if __name__ == "__main__":
    p = os.path.join(HERE, "close", "why0.json")
    if os.path.exists(p):
        S = json.load(open(p))
    else:
        S = oracle_prob()
        json.dump(S, open(p, "w"))

    ls, hist, wide = oracle_lengths()
    print(f"評価8タスクのオラクル手数 {ls}  中央値 {np.median(ls):.0f}  "
          f"打ち切り {close.MAXSTEP} 手")
    print(f"全部が打ち切り内に収まるか: {'はい' if max(ls) <= close.MAXSTEP else 'いいえ'}")
    print(f"{N_SEED}シードでの手数の分布 {hist}")

    unif = float(np.mean([1.0 / s["n"] for s in S]))
    print(f"\nオラクル行動への確率（h4・T=1.0）: "
          f"平均 {np.mean([agent.normalise(s['raw'])[s['j']] for s in S]):.3f}  "
          f"中央値 {np.median([agent.normalise(s['raw'])[s['j']] for s in S]):.3f}  "
          f"一様なら {unif:.3f}")
    def per_task(T):
        """Probability of taking the oracle action at every step of each task."""
        pr = {}
        for st in S:
            pr.setdefault(st["task"], []).append(
                float(agent.normalise(st["raw"], T)[st["j"]]))
        return {k: float(np.prod(v)) for k, v in pr.items()}

    print(f"\n温度を変えたとき（同じスコアから、許容集合の softmax だけが変わる）")
    print(f"     T   1手の平均確率   オラクル経路を完走する確率（8本の和）   期待成功数/8")
    for T in TEMPS:
        q = float(np.mean([agent.normalise(s["raw"], T)[s["j"]] for s in S]))
        pt = per_task(T)
        tot = sum(pt.values())
        print(f"  {T:4.2f}        {q:.3f}                   {tot:.2e}"
              f"                 {tot:.4f}")
    pt = per_task(1.0)
    print(f"\nT=1.0 の各タスクの完走確率: "
          + " ".join(f"{v:.1e}" for v in pt.values()))
    print(f"8本すべて失敗する確率 ≈ {np.prod([1-v for v in pt.values()]):.4f}")
    print("（オラクル経路以外でも成功しうるので、これは成功確率の下限である）")
    short = [s for s in range(N_SEED) if len(env.oracle(env.Env(s))) <= 3]
    print(f"{N_SEED}シード中、3手以下のタスクは {len(short)} 本"
          f"（この環境では短いタスクが稀）")
