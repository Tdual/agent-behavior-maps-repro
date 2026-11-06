"""A small ALFWorld-style household environment.

Deterministic, with an explicit admissible-action list at every step. The action
list is what makes trajectory KL exactly computable: an agent's policy is a
distribution over a finite, known set, so per-step KL is a closed-form sum rather
than something to estimate.
"""
import random

RECEPS = ["countertop", "diningtable", "shelf", "sinkbasin",
          "fridge", "microwave", "cabinet", "drawer", "garbagecan"]
OPENABLE = {"fridge", "microwave", "cabinet", "drawer"}
OBJECTS = ["apple", "mug", "egg", "bread", "tomato",
           "knife", "plate", "bowl", "potato", "lettuce"]
GOALS = ["put", "heat", "cool", "clean"]


class Env:
    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.rng = rng
        # Scatter five objects over the receptacles.
        objs = rng.sample(OBJECTS, 5)
        self.where = {o: rng.choice(RECEPS) for o in objs}
        self.state = {o: set() for o in objs}       # hot / cold / clean
        self.opened = set()
        self.at = rng.choice(RECEPS)
        self.held = None
        self.goal_kind = rng.choice(GOALS)
        self.target = rng.choice(objs)
        self.dest = rng.choice([r for r in RECEPS if r != self.where[self.target]])
        self.steps = 0
        self.last = "You arrive in the kitchen."

    # ---------- description ----------

    def task(self) -> str:
        v = {"put": "put", "heat": "heat", "cool": "cool", "clean": "clean"}[self.goal_kind]
        if self.goal_kind == "put":
            return f"put the {self.target} on the {self.dest}"
        return f"{v} the {self.target} and put it on the {self.dest}"

    def _visible(self):
        if self.at in OPENABLE and self.at not in self.opened:
            return None
        return sorted(o for o, r in self.where.items() if r == self.at)

    def observe(self) -> str:
        vis = self._visible()
        if vis is None:
            view = f"The {self.at} is closed."
        elif vis:
            view = f"On the {self.at} you see " + ", ".join(
                f"a {o}" + (f" ({'/'.join(sorted(self.state[o]))})" if self.state[o] else "")
                for o in vis) + "."
        else:
            view = f"On the {self.at} you see nothing."
        hold = f" You are carrying a {self.held}." if self.held else ""
        return f"{self.last} {view}{hold}"

    # ---------- actions ----------

    def admissible(self):
        acts = [f"go to {r}" for r in RECEPS if r != self.at]
        acts.append(f"examine {self.at}")
        if self.at in OPENABLE:
            acts.append(f"{'close' if self.at in self.opened else 'open'} {self.at}")
        vis = self._visible()
        if self.held is None and vis:
            acts += [f"take {o} from {self.at}" for o in vis]
        if self.held is not None and vis is not None:
            acts.append(f"put {self.held} on {self.at}")
            if self.at == "microwave":
                acts.append(f"heat {self.held} with microwave")
            if self.at == "fridge":
                acts.append(f"cool {self.held} with fridge")
            if self.at == "sinkbasin":
                acts.append(f"clean {self.held} with sinkbasin")
        return sorted(acts)

    def step(self, action: str):
        self.steps += 1
        a = action.strip()
        if a.startswith("go to "):
            self.at = a[6:]
            self.last = f"You go to the {self.at}."
        elif a.startswith("open "):
            self.opened.add(a[5:])
            self.last = f"You open the {a[5:]}."
        elif a.startswith("close "):
            self.opened.discard(a[6:])
            self.last = f"You close the {a[6:]}."
        elif a.startswith("examine "):
            self.last = f"You examine the {a[8:]}."
        elif a.startswith("take "):
            o = a[5:].split(" from ")[0]
            self.held = o
            self.where.pop(o, None)
            self.last = f"You pick up the {o}."
        elif a.startswith("put "):
            o = a[4:].split(" on ")[0]
            self.where[o] = self.at
            self.held = None
            self.last = f"You put the {o} on the {self.at}."
        elif a.startswith("heat "):
            o = a[5:].split(" with ")[0]
            self.state[o].add("hot"); self.state[o].discard("cold")
            self.last = f"You heat the {o}."
        elif a.startswith("cool "):
            o = a[5:].split(" with ")[0]
            self.state[o].add("cold"); self.state[o].discard("hot")
            self.last = f"You cool the {o}."
        elif a.startswith("clean "):
            o = a[6:].split(" with ")[0]
            self.state[o].add("clean")
            self.last = f"You clean the {o}."
        else:
            self.last = "Nothing happens."
        return self.observe()

    def success(self) -> bool:
        if self.where.get(self.target) != self.dest:
            return False
        need = {"put": None, "heat": "hot", "cool": "cold", "clean": "clean"}[self.goal_kind]
        return need is None or need in self.state[self.target]


def oracle(env: Env):
    """A shortest sensible plan, used to generate reference trajectories.

    Tracks the simulated position and opened set, because `go to` excludes the
    current location and `open` is only offered while the receptacle is shut.
    """
    plan = []
    at, opened = env.at, set(env.opened)

    def goto(r):
        nonlocal at
        if at != r:
            plan.append(f"go to {r}")
            at = r
        if r in OPENABLE and r not in opened:
            plan.append(f"open {r}")
            opened.add(r)

    tool = {"heat": "microwave", "cool": "fridge", "clean": "sinkbasin"}.get(env.goal_kind)
    goto(env.where[env.target])
    plan.append(f"take {env.target} from {at}")
    if tool:
        goto(tool)
        plan.append(f"{env.goal_kind} {env.target} with {tool}")
    goto(env.dest)
    plan.append(f"put {env.target} on {env.dest}")
    return plan
