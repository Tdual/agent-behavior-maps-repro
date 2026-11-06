# -*- coding: utf-8 -*-
"""Agent Behavior Maps, reproduced small.

Stages (each cached on disk, so they can be run one at a time):

  refs      build reference trajectories (oracle + random policy, no agent
            under study is a generator -- this avoids the reference-provenance
            problem the paper flags for its own ALFWorld set)
  coords    every agent teacher-forces the recorded actions -> coordinates
  roll      every agent rolls out held-out tasks, sampling its own policy
  cross     every agent scores every rollout's visited histories
  report    coordinate distances vs. measured trajectory KL, and neighbours

The admissible-action list makes the policy a distribution over a finite known
set, so per-step KL is a closed-form sum and the trajectory KL needs no
density estimation.
"""
import json, os, random, sys, time
import numpy as np
import agent, env, models

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, *p)
N_REF = 24          # reference trajectories
N_EVAL = 8          # held-out tasks for rollouts
MAXSTEP = 10        # rollout cap
REF_SEED0, EVAL_SEED0 = 0, 9000


# ---------------------------------------------------------------- refs

def build_refs():
    out = []
    for k in range(N_REF):
        seed = REF_SEED0 + k
        e = env.Env(seed)
        init, task = e.observe(), e.task()
        steps, hist = [], []
        if k % 2 == 0:                      # oracle generator
            plan = env.oracle(env.Env(seed))
        else:                               # random admissible generator
            plan = None
        rng = random.Random(10_000 + seed)
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
    json.dump(out, open(D("refs.json"), "w"), ensure_ascii=False)
    M = sum(len(t["steps"]) for t in out)
    print(f"参照軌跡 {len(out)}本  {M}ステップ  "
          f"（オラクル{sum(1 for t in out if t['gen']=='oracle')}本 / "
          f"無作為{sum(1 for t in out if t['gen']=='random')}本）")
    return out


# ------------------------------------------------------------- coords

def coords():
    refs = json.load(open(D("refs.json")))
    os.makedirs(D("coords"), exist_ok=True)
    for name, mid in models.MODELS:
        p = D("coords", f"{name}.npz")
        if os.path.exists(p):
            print(f"  {name:<12} キャッシュ済み"); continue
        a = agent.Agent(mid)
        raw, norm, tid = [], [], []
        t0 = time.perf_counter()
        for ti, tr in enumerate(refs):
            for st in tr["steps"]:
                hist = [tuple(h) for h in st["hist"]]
                sc = a.score(tr["task"], tr["init"], hist, st["admissible"])
                j = st["admissible"].index(st["action"])
                raw.append(sc[j])
                norm.append(float(np.log(agent.normalise(sc)[j])))
                tid.append(ti)
        a.close()
        np.savez(p, raw=np.array(raw), norm=np.array(norm), tid=np.array(tid))
        print(f"  {name:<12} {len(raw)}ステップ {time.perf_counter()-t0:.0f}s")


# --------------------------------------------------------------- roll

def roll():
    os.makedirs(D("rollouts"), exist_ok=True)
    for name, mid in models.MODELS:
        p = D("rollouts", f"{name}.json")
        if os.path.exists(p):
            print(f"  {name:<12} キャッシュ済み"); continue
        a = agent.Agent(mid)
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        trajs = []
        t0 = time.perf_counter()
        for k in range(N_EVAL):
            seed = EVAL_SEED0 + k
            e = env.Env(seed)
            init, task = e.observe(), e.task()
            steps, hist = [], []
            for _ in range(MAXSTEP):
                adm = e.admissible()
                pol = agent.normalise(a.score(task, init, hist, adm))
                act = adm[int(rng.choice(len(adm), p=pol))]
                obs = e.observe()
                steps.append({"obs": obs, "admissible": adm, "action": act,
                              "hist": [list(h) for h in hist]})
                hist.append((obs, act))
                e.step(act)
                if e.success():
                    break
            trajs.append({"seed": seed, "task": task, "init": init,
                          "steps": steps, "success": e.success()})
        a.close()
        json.dump(trajs, open(p, "w"), ensure_ascii=False)
        ns = sum(len(t["steps"]) for t in trajs)
        print(f"  {name:<12} {ns}ステップ 成功{sum(t['success'] for t in trajs)}/{N_EVAL} "
              f"{time.perf_counter()-t0:.0f}s")


# -------------------------------------------------------------- cross

def cross():
    """scorer j's policy at every history visited by owner i."""
    os.makedirs(D("cross"), exist_ok=True)
    owners = {n: json.load(open(D("rollouts", f"{n}.json"))) for n, _ in models.MODELS}
    for name, mid in models.MODELS:
        todo = [o for o, _ in models.MODELS
                if not os.path.exists(D("cross", f"{name}__{o}.json"))]
        if not todo:
            print(f"  {name:<12} キャッシュ済み"); continue
        a = agent.Agent(mid)
        t0 = time.perf_counter()
        for o in todo:
            pols = []
            for tr in owners[o]:
                for st in tr["steps"]:
                    hist = [tuple(h) for h in st["hist"]]
                    sc = a.score(tr["task"], tr["init"], hist, st["admissible"])
                    pols.append(agent.normalise(sc).tolist())
            json.dump(pols, open(D("cross", f"{name}__{o}.json"), "w"))
        a.close()
        print(f"  {name:<12} {len(todo)}本の軌跡集合 {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    for stage in sys.argv[1:]:
        print(f"--- {stage}")
        {"refs": build_refs, "coords": coords, "roll": roll, "cross": cross}[stage]()
