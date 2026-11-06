# -*- coding: utf-8 -*-
"""Agent Behavior Maps, inside the paper's validity region.

The first run put six different LLMs on one map. Their policies turned out to
differ as much as policies can on this environment (measured epsilon, the
largest per-step log-ratio, had median 4.21), and the paper's bound only holds
when that quantity is small. So the map was tested outside the region where it
is claimed to work.

This run keeps the weights fixed and varies only the harness, which is the
other half of the paper's definition of an agent. Nine agents share one LLM
and differ in:

  history window   4 / 2 / 1 / 0 blocks
  system prompt    the base instruction / a terse one / none
  temperature      1.0 / 0.9 / 1.1 / 1.25 over the admissible set

Temperature is a deliberate probe. It changes the policy while leaving every
token log-probability untouched, so the paper's `raw` coordinate -- the summed
token log-likelihood of the recorded action -- cannot see it at all. Those
pairs sit at coordinate distance exactly zero with a non-zero KL, which is a
blind spot of the convention rather than of the method.

Stages mirror pipeline.py and cache the same way. Scoring is done per harness,
not per agent: the four temperature agents of h4 are derived from the h4 score
vectors without another forward pass.
"""
import itertools, json, os, sys, time
import numpy as np
import agent, env

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, "close", *p)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_EVAL, MAXSTEP, EVAL_SEED0 = 8, 8, 9000

SYS_BASE = agent.SYS
SYS_TERSE = "Pick one admissible action."

HARNESS = {
    "h4": dict(window=4, sys_prompt=SYS_BASE),
    "h2": dict(window=2, sys_prompt=SYS_BASE),
    "h1": dict(window=1, sys_prompt=SYS_BASE),
    "h0": dict(window=0, sys_prompt=SYS_BASE),
    "n4": dict(window=4, sys_prompt=""),
    "s4": dict(window=4, sys_prompt=SYS_TERSE),
}

# name, harness, temperature
AGENTS = [
    ("h4",       "h4", 1.00),
    ("h4-T0.9",  "h4", 0.90),
    ("h4-T1.1",  "h4", 1.10),
    ("h4-T1.25", "h4", 1.25),
    ("h2",       "h2", 1.00),
    ("h1",       "h1", 1.00),
    ("h0",       "h0", 1.00),
    ("n4",       "n4", 1.00),
    ("s4",       "s4", 1.00),
]
NAMES = [a[0] for a in AGENTS]
HA = {n: h for n, h, _ in AGENTS}
TA = {n: t for n, _, t in AGENTS}


def load(h):
    return agent.Agent(MODEL, **HARNESS[h])


# ------------------------------------------------------------- coords

def coords():
    refs = json.load(open(os.path.join(HERE, "refs.json")))
    os.makedirs(D("coords"), exist_ok=True)
    for h in HARNESS:
        p = D("coords", f"{h}.json")
        if os.path.exists(p):
            print(f"  {h:<4} キャッシュ済み"); continue
        a, t0 = load(h), time.perf_counter()
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
        print(f"  {h:<4} {len(vecs)}ステップ {time.perf_counter()-t0:.0f}s")


# --------------------------------------------------------------- roll

def roll():
    os.makedirs(D("rollouts"), exist_ok=True)
    for h in HARNESS:
        todo = [n for n in NAMES if HA[n] == h
                and not os.path.exists(D("rollouts", f"{n}.json"))]
        if not todo:
            continue
        a, t0 = load(h), time.perf_counter()
        for n in todo:
            rng = np.random.default_rng(abs(hash(n)) % (2**32))
            trajs = []
            for k in range(N_EVAL):
                e = env.Env(EVAL_SEED0 + k)
                init, task = e.observe(), e.task()
                steps, hist = [], []
                for _ in range(MAXSTEP):
                    adm = e.admissible()
                    pol = agent.normalise(
                        a.score(task, init, hist, adm), TA[n])
                    act = adm[int(rng.choice(len(adm), p=pol))]
                    obs = e.observe()
                    steps.append({"obs": obs, "admissible": adm, "action": act,
                                  "hist": [list(x) for x in hist]})
                    hist.append((obs, act))
                    e.step(act)
                    if e.success():
                        break
                trajs.append({"seed": EVAL_SEED0 + k, "task": task,
                              "init": init, "steps": steps,
                              "success": e.success()})
            json.dump(trajs, open(D("rollouts", f"{n}.json"), "w"))
            print(f"  {n:<9} {sum(len(t['steps']) for t in trajs)}ステップ "
                  f"成功{sum(t['success'] for t in trajs)}/{N_EVAL}")
        a.close()
        print(f"  [{h}] {time.perf_counter()-t0:.0f}s")


# -------------------------------------------------------------- cross

def cross():
    """Every harness's score vectors at every history visited by every agent."""
    os.makedirs(D("cross"), exist_ok=True)
    own = {n: json.load(open(D("rollouts", f"{n}.json"))) for n in NAMES}
    for h in HARNESS:
        todo = [n for n in NAMES
                if not os.path.exists(D("cross", f"{h}__{n}.json"))]
        if not todo:
            print(f"  {h:<4} キャッシュ済み"); continue
        a, t0 = load(h), time.perf_counter()
        for n in todo:
            vecs = []
            for tr in own[n]:
                for st in tr["steps"]:
                    vecs.append(a.score(
                        tr["task"], tr["init"],
                        [tuple(x) for x in st["hist"]],
                        st["admissible"]).tolist())
            json.dump(vecs, open(D("cross", f"{h}__{n}.json"), "w"))
        a.close()
        print(f"  {h:<4} 所有者{len(todo)}体 {time.perf_counter()-t0:.0f}s")


# ---------------------------------------------------- common random numbers

CRN_SEED = 777


def crn_u(task_k, t):
    """One uniform per (task, step), shared by every agent.

    With independent streams per agent, two agents holding the SAME policy
    still produce different action sequences: the first-step policy entropy is
    0.82 nats, and replicates of one agent sit 0.44-0.58 apart in normalised
    edit distance. That floor is as large as the whole between-agent range, so
    the behavioural metrics measured the dice rather than the policies.

    Sharing the uniform and inverting each agent's CDF removes it: identical
    policies now give identical sequences, and the remaining difference is the
    policy difference. Where two agents have already diverged the admissible
    list differs, so the coupling is partial, not perfect -- but it is exact at
    every step they reach together, which includes every first step.
    """
    return float(np.random.default_rng([CRN_SEED, task_k, t]).random())


def pick(pol, u):
    return int(np.searchsorted(np.cumsum(pol), u, side="right").clip(0, len(pol) - 1))


def roll_crn():
    os.makedirs(D("rollouts_crn"), exist_ok=True)
    for h in HARNESS:
        todo = [n for n in NAMES if HA[n] == h
                and not os.path.exists(D("rollouts_crn", f"{n}.json"))]
        if not todo:
            continue
        a, t0 = load(h), time.perf_counter()
        for n in todo:
            trajs = []
            for k in range(N_EVAL):
                e = env.Env(EVAL_SEED0 + k)
                init, task = e.observe(), e.task()
                steps, hist = [], []
                for t in range(MAXSTEP):
                    adm = e.admissible()
                    pol = agent.normalise(a.score(task, init, hist, adm), TA[n])
                    act = adm[pick(pol, crn_u(k, t))]
                    obs = e.observe()
                    steps.append({"obs": obs, "admissible": adm, "action": act,
                                  "hist": [list(x) for x in hist]})
                    hist.append((obs, act))
                    e.step(act)
                    if e.success():
                        break
                trajs.append({"seed": EVAL_SEED0 + k, "task": task,
                              "init": init, "steps": steps,
                              "success": e.success()})
            json.dump(trajs, open(D("rollouts_crn", f"{n}.json"), "w"))
            print(f"  {n:<9} {sum(len(t['steps']) for t in trajs)}ステップ "
                  f"成功{sum(t['success'] for t in trajs)}/{N_EVAL}", flush=True)
        a.close()
        print(f"  [{h}] {time.perf_counter()-t0:.0f}s", flush=True)


def cross_crn():
    os.makedirs(D("cross_crn"), exist_ok=True)
    own = {n: json.load(open(D("rollouts_crn", f"{n}.json"))) for n in NAMES}
    for h in HARNESS:
        todo = [n for n in NAMES
                if not os.path.exists(D("cross_crn", f"{h}__{n}.json"))]
        if not todo:
            print(f"  {h:<4} キャッシュ済み"); continue
        a, t0 = load(h), time.perf_counter()
        for n in todo:
            vecs = []
            for tr in own[n]:
                for st in tr["steps"]:
                    vecs.append(a.score(
                        tr["task"], tr["init"],
                        [tuple(x) for x in st["hist"]],
                        st["admissible"]).tolist())
            json.dump(vecs, open(D("cross_crn", f"{h}__{n}.json"), "w"))
        a.close()
        print(f"  {h:<4} 所有者{len(todo)}体 {time.perf_counter()-t0:.0f}s", flush=True)

# ------------------------------------------------- harness composition

"""Does the displacement caused by two harness changes add up?

The paper reports, per environment, the cosine between the sum of the
one-component displacements and the displacement of the combined change, the
ratio of their norms, and the relative error. Here the two components are the
history window and the system prompt, and the base point is h4.

Only coordinates are needed, so this stage does not roll out or cross-score.
"""

COMPOSE = {
    "n2": dict(window=2, sys_prompt=""),
    "s2": dict(window=2, sys_prompt=SYS_TERSE),
    "n0": dict(window=0, sys_prompt=""),
    "s0": dict(window=0, sys_prompt=SYS_TERSE),
}
# combined harness: (window component, system-prompt component)
TESTS = [("n2", "h2", "n4"), ("s2", "h2", "s4"),
         ("n0", "h0", "n4"), ("s0", "h0", "s4")]


def coords_compose():
    refs = json.load(open(os.path.join(HERE, "refs.json")))
    os.makedirs(D("coords"), exist_ok=True)
    for h, kw in COMPOSE.items():
        p = D("coords", f"{h}.json")
        if os.path.exists(p):
            print(f"  {h:<4} キャッシュ済み"); continue
        a, t0 = agent.Agent(MODEL, **kw), time.perf_counter()
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
        print(f"  {h:<4} {len(vecs)}ステップ {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        print(f"--- {s}", flush=True)
        {"coords": coords, "roll": roll, "cross": cross,
         "roll_crn": roll_crn, "cross_crn": cross_crn,
         "coords_compose": coords_compose}[s]()
