---
title: Music Catalog Valuation PE Environment
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
---

# Music Catalog PE Environment

An OpenEnv environment simulating a private equity diligence process for music catalogs. Designed for Meta's PyTorch OpenEnv Hackathon.

## Objective
The agent must inspect messy music catalog datasets (tracks, historical revenue, platform/territory mix), identify revenue drivers and anomalies, assess risk flags, and provide a quantitative valuation estimate. 

The environment tests an LLM agent's ability to act like a financial diligence analyst.

## Tasks
There are three deterministically generated tasks based on real-world music asset tropes:
1. **`easy_stable_evergreen`**: A highly diversified pop/rock catalog. Easy to value using straight multiple methods.
2. **`medium_concentration_risk`**: A hip-hop catalog driven by a few top tracks heavily dependent on YouTube. The multiple must be discounted for concentration risk.
3. **`hard_viral_spike_noisy`**: A noisy dataset with a recent viral TikTok spike and partial ownership rights that require careful revenue normalization and outlier dampening.

## Interaction Loop
You connect to the environment, call `reset()`, and then use `step(action)` iteratively.
The observation will return JSON sub-structures depending on the `action_type`.
The final action **MUST** be `submit_final_valuation`.

Available action types include: `inspect_catalog_summary`, `inspect_top_tracks`, `inspect_platform_mix`, `inspect_territory_mix`, `inspect_monthly_revenue_trend`, `inspect_anomalies`.

## Running Locally
Ensure Python 3.10+ is installed.

```bash
uv venv
uv pip install -r requirements.txt
python music_catalog_pe_env/generators.py  # Generate synthetic datasets
fastapi dev server/app.py
```

## Running the Baseline Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your_token_here>"
python inference.py
```

## Grading System
Grading is deterministic, yielding a score strictly in `(0.0, 1.0)`. It applies a weighted rubric:
- 40% accuracy on estimating Normalized Trailing 12-Month Revenue
- 60% accuracy on final Valuation Base Multiple estimate
- Additional +/- modifiers for identifying exact risk flags (Jaccard similarity) and recommendation correctness.
- Dense rewards are given for individual actions (e.g., investigating platforms early).

## License
MIT
