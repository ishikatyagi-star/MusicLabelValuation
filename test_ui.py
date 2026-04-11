import asyncio
import json
from server.ui import reset_manual_env, quick_action, execute_action

print("Testing reset...")
msg, out = reset_manual_env("easy_stable_evergreen")
print("Reset:", msg)

print("\nTesting anomaly scan (booleans)...")
msg, out = quick_action("inspect_anomalies")
print("Anomaly output:", out[:200])

print("\nTesting valuation submit...")
params = {
    "catalog_id": "easy_stable_evergreen",
    "estimated_normalized_ttm_revenue": 100000,
    "valuation_low": 1,
    "valuation_base": 1,
    "valuation_high": 1,
    "estimated_top_tracks": [],
    "estimated_key_platforms": [],
    "estimated_risk_flags": [],
    "recommendation": "acquire",
    "confidence_score": 0.9,
    "memo": "None"
}
msg, out = execute_action("submit_final_valuation", json.dumps(params))
print("Valuation MSG:", msg)
