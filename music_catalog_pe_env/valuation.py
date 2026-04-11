import math
from typing import Any, Dict, List


def calculate_hhi(shares_pct: List[float]) -> float:
    """Calculate the Herfindahl-Hirschman Index (scale 0-10,000)."""
    return sum(s**2 for s in shares_pct)

def evaluate_catalog(catalog_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the catalog using deterministic business logic based on ground truth variables.
    
    Returns:
    - true_normalized_ttm_revenue
    - base_multiple
    - multiple_adjustments (dict)
    - final_valuation
    """
    normalized_ttm = ground_truth.get("true_normalized_ttm_revenue", 0.0)
    
    # Base multiple
    base_multiple = 7.0
    adjustments = {}
    
    # 1. Growth/Decay Modifier (+/- 1.5)
    growth_rate = ground_truth.get("growth_rate_assumption", 0.0)
    # Roughly scale growth to a multiple bump: +10% growth -> +1.0 multiple
    growth_adj = max(-1.5, min(1.5, growth_rate * 10.0))
    # Slight dampening if stability is low
    stability = ground_truth.get("stability_score", 1.0)
    growth_adj *= stability
    adjustments["growth_decay"] = round(growth_adj, 2)
    
    # 2. Stability Modifier (-1.0 to +1.0)
    # 1.0 stability -> +0.5, 0.0 -> -1.0
    stability_adj = round((stability - 0.5) * 2.0, 2)
    # Give a cap
    stability_adj = max(-1.0, min(1.0, stability_adj))
    adjustments["stability"] = stability_adj
    
    # 3. Top-track concentration penalty
    top_track = ground_truth.get("top_track_concentration", 0.5)
    # If top tracks > 40%, start penalizing up to -2.0
    if top_track > 0.40:
        penalty = -((top_track - 0.40) / 0.60) * 2.0
        adjustments["track_concentration"] = round(penalty, 2)
    else:
        adjustments["track_concentration"] = 0.0
        
    # 4. Platform dependency penalty
    plat_conc = ground_truth.get("top_platform_concentration", 0.3)
    if plat_conc > 0.45:
        penalty = -((plat_conc - 0.45) / 0.55) * 1.5
        adjustments["platform_dependency"] = round(penalty, 2)
    else:
        adjustments["platform_dependency"] = 0.0
        
    # 5. Data anomaly penalty
    anomalies = ground_truth.get("anomaly_flags", [])
    anomaly_penalty = -0.5 * len(anomalies)
    adjustments["data_quality"] = round(max(-2.0, anomaly_penalty), 2)
    
    # Sum it up
    total_adj = sum(adjustments.values())
    final_multiple = base_multiple + total_adj
    
    # Hard clamp
    final_multiple = max(3.0, min(12.0, final_multiple))
    
    final_valuation = normalized_ttm * final_multiple
    
    return {
        "normalized_ttm": round(normalized_ttm, 2),
        "base_multiple": base_multiple,
        "adjustments": adjustments,
        "final_multiple": round(final_multiple, 2),
        "final_valuation": round(final_valuation, 2)
    }

def generate_analyst_ranges(exact_valuation: float, confidence: float) -> Tuple[float, float, float]:
    """Generates a low/base/high range based on a target valuation."""
    # Confidence determines spread: tighter spread if confident
    spread_pct = 0.30 - (confidence * 0.15) # 15% to 30% spread
    
    low = exact_valuation * (1.0 - spread_pct)
    high = exact_valuation * (1.0 + spread_pct)
    return round(low, 2), round(exact_valuation, 2), round(high, 2)
