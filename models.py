"""The agents under study: small, open-weight, and deliberately mixed.

Two base/instruct pairs and three families, so the map has structure the
neighbour claim can be checked against. All fp32 so log-probs are comparable.
"""
MODELS = [
    ("smol135-it",  "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("smol135",     "HuggingFaceTB/SmolLM2-135M"),
    ("smol360-it",  "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("qwen05-it",   "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen05",      "Qwen/Qwen2.5-0.5B"),
    ("pythia410",   "EleutherAI/pythia-410m"),
]
FAMILY = {"smol135-it": "SmolLM2", "smol135": "SmolLM2", "smol360-it": "SmolLM2",
          "qwen05-it": "Qwen2.5", "qwen05": "Qwen2.5", "pythia410": "Pythia"}
TUNED = {"smol135-it": True, "smol135": False, "smol360-it": True,
         "qwen05-it": True, "qwen05": False, "pythia410": False}
PARAMS = {"smol135-it": 135e6, "smol135": 135e6, "smol360-it": 360e6,
          "qwen05-it": 494e6, "qwen05": 494e6, "pythia410": 410e6}
