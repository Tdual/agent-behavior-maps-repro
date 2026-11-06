# -*- coding: utf-8 -*-
"""How much does the choice of reference set move the map? (lambda sensitivity)

The paper's bound carries a reference-mismatch term lambda: the coordinates
are read off trajectories nobody under study generated, so the map measures
the policies only where those trajectories went. Two things can move:

  setB   same construction (half oracle, half random), different task seeds
         -- the sampling variance of a reference set of this size
  setC   every trajectory from the random generator, same seeds as the
         original -- a shift in the generator, which is what lambda names

Both sets are scored with the same six harnesses, so the comparison isolates
the reference set. Nothing here touches the LLM set.
"""
import json, os, random, sys, time
import numpy as np
import agent, env, close

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, *p)
N_REF, MAXSTEP = 24, 10
SETS = {"B": dict(seed0=5000, mix="half"), "C": dict(seed0=0, mix="random")}


def build(tag):
    cfg = SETS[tag]
    out = []
    for k in range(N_REF):
        seed = cfg["seed0"] + k
        e = env.Env(seed)
        init, task = e.observe(), e.task()
        steps, hist = [], []
        use_oracle = cfg["mix"] == "half" and k % 2 == 0
        plan = env.oracle(env.Env(seed)) if use_oracle else None
        rng = random.Random(20_000 + seed)
        for t in range(MAXSTEP):
            adm = e.admissible()
            if plan is not None:
                if t >= len(plan):
                    break
                act = plan[t]
            else:
                act = rng.choice(adm)
            obs = e.observe()
            steps.append({"obs": obs, "admissible": adm, "action": act,
                          "hist": [list(h) for h in hist]})
            hist.append((obs, act))
            e.step(act)
        out.append({"seed": seed, "task": task, "init": init,
                    "gen": "oracle" if plan is not None else "random",
                    "steps": steps, "success": e.success()})
    json.dump(out, open(D(f"refs_{tag}.json"), "w"), ensure_ascii=False)
    print(f"  set{tag}: {len(out)}本 {sum(len(t['steps']) for t in out)}ステップ "
          f"（オラクル{sum(1 for t in out if t['gen']=='oracle')}本）", flush=True)


def coords(tag):
    refs = json.load(open(D(f"refs_{tag}.json")))
    os.makedirs(D(f"coords_{tag}"), exist_ok=True)
    for h in close.HARNESS:
        p = D(f"coords_{tag}", f"{h}.json")
        if os.path.exists(p):
            print(f"  set{tag} {h:<4} キャッシュ済み", flush=True); continue
        a, t0 = close.load(h), time.perf_counter()
        vecs, tid, idx = [], [], []
        for ti, tr in enumerate(refs):
            for st in tr["steps"]:
                sc = a.score(tr["task"], tr["init"],
                             [tuple(x) for x in st["hist"]], st["admissible"])
                vecs.append(sc.tolist())
                idx.append(st["admissible"].index(st["action"]))
                tid.append(ti)
        a.close()
        json.dump({"vecs": vecs, "idx": idx, "tid": tid}, open(p, "w"))
        print(f"  set{tag} {h:<4} {len(vecs)}ステップ "
              f"{time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    print("--- 参照集合を2つ作る")
    for t in SETS:
        build(t)
    print("--- 6ハーネスで採点")
    for t in SETS:
        coords(t)
