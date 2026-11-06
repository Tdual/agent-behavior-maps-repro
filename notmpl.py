# -*- coding: utf-8 -*-
"""How much of the LLM map is the chat template rather than the weights?

Table 4 of the paper says the 100-model run uses "each model's own chat
template". That makes the template part of the agent, so two models can sit
far apart on the map because their templates differ, not because their
policies do. Here every LLM is scored a second time with the template
switched off -- the system prompt and the body concatenated as plain text,
which is what a model without a template already gets -- on the same 24
reference trajectories. Both scorings use the same weights and the same
harness, so the gap between them is the template's contribution.
"""
import json, os, time
import numpy as np
import agent, models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "coords_plain")


def run():
    refs = json.load(open(os.path.join(HERE, "refs.json")))
    os.makedirs(OUT, exist_ok=True)
    for name, mid in models.MODELS:
        p = os.path.join(OUT, f"{name}.json")
        if os.path.exists(p):
            print(f"  {name:<12} キャッシュ済み", flush=True); continue
        a, t0 = agent.Agent(mid), time.perf_counter()
        had = a.chat
        a.chat = False                      # plain-text prefix for everyone
        vecs, idx, tid = [], [], []
        for ti, tr in enumerate(refs):
            for st in tr["steps"]:
                sc = a.score(tr["task"], tr["init"],
                             [tuple(h) for h in st["hist"]], st["admissible"])
                vecs.append(sc.tolist())
                idx.append(st["admissible"].index(st["action"]))
                tid.append(ti)
        a.close()
        json.dump({"vecs": vecs, "idx": idx, "tid": tid, "had_template": had},
                  open(p, "w"))
        print(f"  {name:<12} テンプレート{'あり' if had else 'なし'} "
              f"{len(vecs)}ステップ {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()
