<!-- [BANNER] -->
<p align="center">
  <img src="./assets/branding/banner.png" alt="Music Catalog PE Environment Banner" width="100%">
</p>

# 🎵 Music Catalog Valuation PE Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![OpenEnv Compliant](https://img.shields.io/badge/Framework-OpenEnv-orange.svg)](https://github.com/meta-pytorch/OpenEnv)
[![Standard: Financial Reasoning](https://img.shields.io/badge/Domain-FinTech-gold.svg)](#)

A high-fidelity benchmarking environment designed to evaluate AI agents in **private equity diligence for music catalogs**. Move beyond simple summarization and test your model's ability to identify revenue drivers, normalize viral spikes, and calculate risk-adjusted valuations.

---

## 🌍 Why It Matters: Real-World Impact

Music catalogs have grown into a multi-billion dollar alternative asset class (as seen with firms like Blackstone and Hipgnosis). However, valuing these assets is notoriously difficult due to:
- **Streaming Volatility**: Viral TikTok spikes can inflate "perceived" value.
- **Messy Data**: Royalties are often distributed across hundreds of platforms and territories with noisy reporting.
- **Risk Assessment**: High concentration in a single artist or platform creates existential risks for investors.

This environment bridges the gap between **Generic LLMs** and **Specialized Financial Analysts** by providing a sandbox where agents must demonstrate quantitative integrity and causal reasoning before capital is deployed.

---

## 🚀 Key Highlights

- **🎯 Deterministic Grading**: Uses programmatic graders with objective success criteria, eliminating "vibes-based" evaluation.
- **📈 Real-World Scenarios**: Simulates authentic music asset tropes—from evergreen pop catalogs to viral TikTok-driven hip-hop portfolios.
- **🧠 Multi-Step Reasoning**: Agents must iteratively inspect platform mixes, territory trends, and anomalies before submitting a final valuation.
- **🏢 Deep Financial Nuance**: Challenges agents to differentiate between stable recurring revenue and volatile, non-recurring viral spikes.

---

## 📊 Benchmark Tasks

We provide three curated scenarios of increasing complexity. Agents must process messy music catalog data and return a structured valuation report.

| Complexity | Task Name | Core Challenge | Max Steps |
| :--- | :--- | :--- | :--- |
| **🟢 Easy** | `stable_evergreen` | High-diversification pop/rock. Focus on standard multiple application. | 25 |
| **🟡 Medium** | `concentration_risk` | Hip-hop catalog with heavy YouTube dependency. Requires risk-adjusted discounting. | 30 |
| **🔴 Hard** | `viral_spike_noisy` | Viral TikTok spikes and partial rights. Requires normalization and outlier dampening. | 40 |

---

## 🛠️ Technical Specifications

### The Observation Space (`CatalogObservation`)
Agents receive a structured state containing:
- **`task_description`**: The specific diligence mandate.
- **`available_actions`**: Dynamically updated list of valid next steps.
- **`result_payload`**: Data returned from previous inspections (CSV/JSON).
- **`remaining_budget`**: Step count remaining before mandatory submission.

### The Action Space (`CatalogAction`)
Agents can interact with the environment through multiple inspection tools:
- `inspect_catalog_summary`: Overall metadata and asset counts.
- `inspect_top_tracks`: Revenue breakdown by individual track.
- `inspect_platform_mix`: Distribution across Spotify, Apple, YouTube, etc.
- `inspect_monthly_revenue_trend`: Time-series data for trend analysis.
- `inspect_anomalies`: Flags for copyright strikes or data gaps.
- **`submit_final_valuation`**: Final step to provide estimated TTM and Multiple.

---

## 🧠 Evaluation Framework

Our grading engine calculates a weighted score (`0.0 - 1.0`) based on four critical pillars:

1. **Financial Accuracy (40%)**: Normalized Trailing 12-Month (TTM) revenue estimation.
2. **Valuation Precision (20%)**: Final Base Multiple selection relative to ground truth.
3. **Risk Detection (20%)**: Accuracy in identifying specific Jaccard-matched risk flags.
4. **Action Efficiency (20%)**: Reward for investigating high-value data points early in the sequence.

---

## 🏁 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/ishikatyagi-star/MusicLabelValuation.git
cd MusicLabelValuation

# Setup environment with uv (recommended)
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

### 2. Local Development
```bash
# Generate synthetic data for tasks
python music_catalog_pe_env/generators.py

# Launch the environment server
fastapi dev server/app.py
```

### 3. Running a Baseline
```bash
export API_BASE_URL="http://localhost:8000"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

---

## 📂 Project Structure

```text
MusicLabelValuation/
├── music_catalog_pe_env/
│   ├── env.py            # Core OpenEnv logic
│   ├── generators.py     # Synthetic data generation
│   ├── graders.py        # Deterministic scoring logic
│   └── models.py         # Type definitions (Pydantic)
├── server/
│   └── app.py            # FastAPI entry point
├── data/                 # Generated catalog datasets
├── inference.py          # Benchmark runner
└── openenv.yaml          # Framework manifest
```

---

### 📄 License
This project is licensed under the **MIT License**.
