# -*- coding: utf-8 -*-
"""The first run again, with common random numbers.

The six separate LLMs were rolled out with an independent random stream each,
so two agents with the same policy still produced different action sequences.
The nine-harness run showed what that costs: the behavioural distance between
two runs of one agent was 0.44-0.58, above some of the between-agent distances,
so the behavioural metrics were measuring the sampling noise.

This repeats the rollout and the cross-scoring with one uniform per (task,
step), shared by every agent and inverted through each agent's own CDF, which
is the same device close.py uses. Identical policies then give identical
sequences and the noise floor is zero.

Everything else is held at the first run's settings: the same 8 evaluation
tasks, the same 10-step cap, the same BASE harness.
"""
import json, os, sys, time
import numpy as np
import agent, close, env, models, pipeline as pl

D = pl.D
NAMES = [n for n, _ in models.MODELS]


def roll_crn():
    os.makedirs(D("rollouts_crn"), exist_ok=True)
    for name, mid in models.MODELS:
        p = D("rollouts_crn", f"{name}.json")
        if os.path.exists(p):
            print(f"  {name:<12} キャッシュ済み"); continue
        a, t0 = agent.Agent(mid), time.perf_counter()
        trajs = []
        for k in range(pl.N_EVAL):
            seed = pl.EVAL_SEED0 + k
            e = env.Env(seed)
            init, task = e.observe(), e.task()
            steps, hist = [], []
            for t in range(pl.MAXSTEP):
                adm = e.admissible()
                pol = agent.normalise(a.score(task, init, hist, adm))
                act = adm[close.pick(pol, close.crn_u(k, t))]
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
        print(f"  {name:<12} {sum(len(t['steps']) for t in trajs)}ステップ "
              f"成功{sum(t['success'] for t in trajs)}/{pl.N_EVAL} "
              f"{time.perf_counter()-t0:.0f}s", flush=True)


def cross_crn():
    os.makedirs(D("cross_crn"), exist_ok=True)
    owners = {n: json.load(open(D("rollouts_crn", f"{n}.json"))) for n in NAMES}
    for name, mid in models.MODELS:
        todo = [o for o in NAMES
                if not os.path.exists(D("cross_crn", f"{name}__{o}.json"))]
        if not todo:
            print(f"  {name:<12} キャッシュ済み"); continue
        a, t0 = agent.Agent(mid), time.perf_counter()
        for o in todo:
            pols = []
            for tr in owners[o]:
                for st in tr["steps"]:
                    sc = a.score(tr["task"], tr["init"],
                                 [tuple(h) for h in st["hist"]], st["admissible"])
                    pols.append(agent.normalise(sc).tolist())
            json.dump(pols, open(D("cross_crn", f"{name}__{o}.json"), "w"))
        a.close()
        print(f"  {name:<12} {len(todo)}本 {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    print("--- roll (共有乱数)"); roll_crn()
    print("--- cross (共有乱数)"); cross_crn()
