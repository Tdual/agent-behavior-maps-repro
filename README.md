# Agent Behavior Maps, reproduced small

A local, zero-cost reproduction of **Agent Behavior Maps** (arXiv:2610.30014) on six
small open-weight models and nine harnesses, plus a re-derivation of the coordinate
choice from information geometry.

An *agent* here is an LLM plus a harness (history window, system prompt, sampling
temperature). The method places agents as points by having each one teacher-force the
actions of a shared set of reference trajectories; the centered per-step log-likelihoods
are the coordinates, and the claim under test is that the squared coordinate distance
approximates the trajectory KL between the two policies. No rollout is needed to compare
two agents.

Everything ran on a local Apple MPS device. No API calls, no cost.

## What the reproduction found

Written up in full in [`FINDINGS.md`](FINDINGS.md). In short:

- **The ordering is reproduced; the level is not.** Within the paper's stated scope
  (close pairs: one model, harness variations) the rank correlation between coordinate
  distance and measured trajectory KL comes back. The absolute level does not — the
  estimate overshoots the measured KL by a factor that does not cancel.
- **Across different LLMs the method is out of scope, and measurably so.** The first run
  put six different models on one map, which the paper names as outside its guarantees;
  every pair sat in the saturated region, leaving no order to recover.
- **Much of the between-model signal is the chat template.** Rescoring all six models
  without their chat template moves each one that has a template by 1.46-1.92 nats per
  step and leaves the two without one at exactly zero. Two policies differing only in
  whether the template is applied sit at 86% of the median distance between different
  models, so removing the template rearranges the map. A base model that ships a default
  system prompt turned out to be the base/instruct confound in the first run.
- **The coordinate is forced to be `log p`, not chosen.** `α = 1` is the e-affine
  coordinate of the exponential family, which is what makes the per-step sum and the
  vector addition in the composition experiment meaningful. The `α = 0` (`√p`) coordinate
  puts every agent on a product of unit spheres, so its differences are chords rather
  than tangent vectors — yet moving to the proper Fisher–Rao tangent space changes the
  composition cosine only in the third decimal. Curvature is not what costs the
  composition its accuracy; component interaction is, and it costs `α = 1` the same.

## Two coordinates, masked and unmasked

The paper proves its bound for the action probability of Appendix A.1 Eq. (6), which is
grammar-masked: the per-token factors are renormalised over the admissible continuations,
so the per-action probabilities sum to one over the admissible set. But Appendix A.1 also
states that "the experiments from §3.2 onward score without the grammar mask", and
Table 4's 100-LLM column reads "No grammar mask; each model's chat template". The map in
Figure 1(a) therefore has coordinates that are unnormalised sums of token log-likelihoods,
not the object the theorem is about. Appendix B.1 bounds the gap empirically rather than
formally: on ten LLMs that recorded both scores the mask changes the mean log-likelihood
per action by -0.01 to +0.03 nats, pairwise squared distances correlate at 1.000, and
substituting the masked scores moves a centered coordinate by a median 0.07 of the
distance to its nearest neighbour.

This reproduction carries both coordinates, so the choice is measured rather than assumed.
`raw` is the summed token log-probability of the recorded action -- the paper's unmasked
coordinate -- and `norm` is the softmax over the admissible set, which is Eq. (6) with one
canonical tokenisation. Inside the paper's scope (one model, nine harnesses, 36 pairs)
`norm` correlates with the measured trajectory KL at Pearson 0.963 / Spearman 0.964 and
`raw` at 0.939 / 0.922.

## The environment

`env.py` is a small ALFWorld-style household environment: deterministic, with an explicit
admissible-action list at every step. That list is the point. A policy is then a
distribution over a finite, known set, so the per-step KL is a closed-form sum and the
trajectory KL needs no density estimation — which is what makes the central claim
checkable at all.

It is an independent implementation that imitates only ALFWorld's surface (receptacles,
goal kinds, observation phrasing). It contains no ALFWorld code and has no dependency on
it. ALFWorld itself is by Shridhar et al., MIT licensed — <https://github.com/alfworld/alfworld>,
[arXiv:2010.03768](https://arxiv.org/abs/2010.03768).

## Layout

| Path | What it is |
|---|---|
| `env.py` | the environment, with the admissible-action list |
| `models.py` | the six agents under study (two base/instruct pairs, three families, fp32) |
| `agent.py` | harness + teacher-forced scoring + the `α` normalisation |
| `pipeline.py` | run 1 — six different LLMs on one map (out of the paper's scope) |
| `close.py` | run 2 — one model, nine harnesses (inside the paper's scope) |
| `close_alpha.py`, `alpha14.py`, `tangent.py`, `why0.py` | the `α`-representation and tangent-space measurements |
| `compose2.py`, `alpha_compose.py`, `close_compose.py` | harness composition (`v̂ = ΔA + ΔB`) |
| `notmpl.py`, `trunc.py`, `crn6.py`, `refs2.py` | chat-template, truncation, common-random-number and reference-set ablations |
| `make_report.py`, `make_blog.py`, `figs.py` | the HTML write-ups and the inline SVG figures |
| `refs*.json`, `coords*/`, `cross*/`, `rollouts*/`, `close/` | cached stage outputs, so any stage can be re-run alone |

## Running it

```bash
pip install torch transformers numpy scipy
python3 pipeline.py refs coords roll cross report   # run 1
python3 close.py                                    # run 2
python3 make_report.py && python3 make_blog.py      # the write-ups
```

Each stage caches to disk and can be run on its own. `close/nexttok.npz` (132 MB) is not
in the repository; `close.py` rebuilds it.
