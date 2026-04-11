from typing import Any, Dict


def epsilon_clamp(val: float) -> float:
    """Clamps strict (0, 1) bounds to satisfy validator."""
    return max(0.001, min(0.999, val))

def compute_pct_error(estimate: float, actual: float) -> float:
    if actual == 0:
        return 1.0 if estimate != 0 else 0.0
    return abs(estimate - actual) / abs(actual)

def score_metrics(submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Returns a score component between 0.0 and 1.0 based on purely quantitative estimates."""
    
    estimated_revenue = submission.get("estimated_normalized_ttm_revenue", 0.0)
    true_revenue = ground_truth.get("true_normalized_ttm_revenue", 0.0)
    
    rev_error = compute_pct_error(estimated_revenue, true_revenue)
    
    estimated_base = submission.get("valuation_base", 0.0)
    true_base = ground_truth.get("true_valuation_base", 0.0)
    base_error = compute_pct_error(estimated_base, true_base)
    
    # Tolerances
    tol_rev = ground_truth.get("grader_tolerances", {}).get("revenue_pct", 0.10)
    tol_val = ground_truth.get("grader_tolerances", {}).get("valuation_pct", 0.15)
    
    rev_score = max(0.0, 1.0 - (rev_error / (tol_rev * 3)))
    base_score = max(0.0, 1.0 - (base_error / (tol_val * 3)))
    
    if rev_error <= tol_rev:
        rev_score = 1.0
    if base_error <= tol_val:
        base_score = 1.0
        
    return (rev_score * 0.4) + (base_score * 0.6)

def score_jaccard(predicted: list, actual: list) -> float:
    s_pred = set(p.lower().strip() for p in predicted)
    s_act = set(a.lower().strip() for a in actual)
    if not s_act:
        return 1.0 if not s_pred else 0.5 # Over-penalize false positives mildly
    intersect = len(s_pred.intersection(s_act))
    union = len(s_pred.union(s_act))
    return intersect / union if union > 0 else 0.0

def score_risks(submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    est_risks = submission.get("estimated_risk_flags", [])
    true_risks = ground_truth.get("must_detect_risks", [])
    return score_jaccard(est_risks, true_risks)

def score_recommendation(submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    rec = submission.get("recommendation", "").lower()
    true_rec = ground_truth.get("correct_recommendation", "").lower()
    
    if rec == true_rec:
        return 1.0
    
    # Adjacent matching (acquire vs acquire_at_discount)
    if ("acquire" in rec and "acquire" in true_rec):
        return 0.5
        
    return 0.0

def grade_submission(submission: Dict[str, Any], ground_truth: Dict[str, Any], steps_taken: int, max_steps: int) -> float:
    """
    Grades the final investment memo and valuation submission.
    """
    if not submission or not ground_truth:
        return 0.001
        
    metric_score = score_metrics(submission, ground_truth)
    risk_score = score_risks(submission, ground_truth)
    rec_score = score_recommendation(submission, ground_truth)
    
    # Efficiency bonus: up to +10% if done perfectly in half steps
    # But only if accuracy is good
    efficiency = max(0.0, (max_steps - steps_taken) / max_steps)
    
    final_raw = (metric_score * 0.6) + (risk_score * 0.2) + (rec_score * 0.2)
    
    if final_raw > 0.6:
        final_raw += (efficiency * 0.1)
        
    return epsilon_clamp(final_raw)
