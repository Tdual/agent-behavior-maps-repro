"""Agent = LLM + harness, with teacher-forced scoring of recorded actions.

The harness is the paper's BASE: the task, the initial observation, the last four
history blocks, no planning, direct action. Each model rebuilds its own input
with its own chat template, which is what makes the harness part of the agent.

Two coordinate conventions are computed for every step:
  raw  - the summed token log-probability of the recorded action (the paper's)
  norm - log of the probability after normalising over the admissible set,
         which is the policy whose KL divergence we can measure exactly.
"""
import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WINDOW = 4
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

SYS = ("You are a household robot. Choose exactly one action from the admissible "
       "list to accomplish the task. Reply with the action only.")


def prompt_text(task, init_obs, history, admissible, window=WINDOW):
    """history: list of (observation, action) already taken, oldest first."""
    lines = [f"Task: {task}", f"Observation: {init_obs}"]
    for obs, act in (history[-window:] if window else []):
        lines.append(f"Observation: {obs}")
        lines.append(f"Action: {act}")
    lines.append("Admissible actions:")
    lines += [f"- {a}" for a in admissible]
    lines.append("Action:")
    return "\n".join(lines)


class Agent:
    def __init__(self, model_id: str, dtype=torch.float32,
                 window=WINDOW, sys_prompt=SYS):
        self.id = model_id
        self.window = window
        self.sys = sys_prompt
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True).to(DEV).eval()
        self.chat = getattr(self.tok, "chat_template", None) is not None

    def close(self):
        del self.model
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()

    def _prefix_ids(self, body: str):
        if self.chat:
            try:
                msgs = ([{"role": "system", "content": self.sys}] if self.sys else []) + \
                       [{"role": "user", "content": body}]
                s = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                return self.tok(s, add_special_tokens=False)["input_ids"]
            except Exception:
                pass
        pre = (self.sys + "\n\n") if self.sys else ""
        return self.tok(pre + body, add_special_tokens=True)["input_ids"]

    @torch.no_grad()
    def score(self, task, init_obs, history, admissible):
        """Summed token log-prob of every admissible action, teacher-forced.

        Only the logits at the continuation positions are materialised. Scoring
        all positions would allocate len(batch) x len(seq) x vocab floats --
        2.7 GB per call for a 152k-vocab model -- and the same again for the
        log-softmax, which is enough to push the machine into swap.
        """
        pre = self._prefix_ids(
            prompt_text(task, init_obs, history, admissible, self.window))
        conts = [self.tok(" " + a, add_special_tokens=False)["input_ids"] for a in admissible]
        seqs = [pre + c for c in conts]
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(DEV), att.to(DEV)

        # Token at absolute position t is predicted by the logits at t-1, and the
        # first continuation token sits at len(pre), so the slice starts there-1.
        keep = L - len(pre) + 1
        try:
            logits = self.model(input_ids=ids, attention_mask=att,
                                logits_to_keep=keep).logits
        except TypeError:
            logits = self.model(input_ids=ids, attention_mask=att).logits[:, -keep:, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        out = np.empty(len(seqs), dtype=np.float64)
        for i, c in enumerate(conts):
            tgt = torch.tensor(c, device=DEV)
            out[i] = logp[i, :len(c), :].gather(1, tgt[:, None]).sum().item()
        return out


def normalise(raw: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Admissible-set policy: softmax over the summed log-probs.

    `temp` sharpens (<1) or flattens (>1) that policy. It is part of the
    harness, not of the weights, so two temperatures of one model are two
    agents that share every token log-probability.
    """
    v = np.asarray(raw, dtype=np.float64) / temp
    p = np.exp(v - v.max())
    return p / p.sum()
