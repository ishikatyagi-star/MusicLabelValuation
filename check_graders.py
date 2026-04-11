import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from music_catalog_pe_env.data_loader import CatalogDataLoader
from music_catalog_pe_env.graders import grade_submission
from music_catalog_pe_env.tasks import TASKS

def test_grader_scenarios():
    print("--- Testing Grader Scenarios for Valid Bounds (0 < score < 1) ---\n")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dl = CatalogDataLoader(base_dir=base_dir)
    
    all_passed = True

    for task_id, config in TASKS.items():
        print(f"Testing Task: {task_id}")
        dl.load(config.catalog_dir)
        gt = dl.ground_truth
        
        # Scenario 1: Perfect match (except maybe 1 small thing to test efficiency scaling)
        perfect_sub = {
            "catalog_id": config.catalog_dir,
            "estimated_normalized_ttm_revenue": gt["true_normalized_ttm_revenue"],
            "valuation_base": gt["true_valuation_base"],
            "estimated_risk_flags": gt.get("must_detect_risks", []),
            "recommendation": gt["correct_recommendation"]
        }
        
        # Scenario 2: Okay match (15% off revenue, missed a risk, adjacent recommendation)
        okay_sub = {
            "catalog_id": config.catalog_dir,
            "estimated_normalized_ttm_revenue": gt["true_normalized_ttm_revenue"] * 1.15,
            "valuation_base": gt["true_valuation_base"] * 0.9,
            "estimated_risk_flags": gt.get("must_detect_risks", [])[:1] if gt.get("must_detect_risks") else ["fake_risk"],
            "recommendation": "acquire" if "discount" in gt.get("correct_recommendation", "") else "acquire_at_discount"
        }
        
        # Scenario 3: Terrible match (Way off, wrong everything)
        terrible_sub = {
            "catalog_id": config.catalog_dir,
            "estimated_normalized_ttm_revenue": gt["true_normalized_ttm_revenue"] * 2.5,
            "valuation_base": gt["true_valuation_base"] * 0.2,
            "estimated_risk_flags": ["wrong_risk"],
            "recommendation": "pass" if gt.get("correct_recommendation") != "pass" else "acquire"
        }
        
        # Scenario 4: Empty match
        empty_sub = {}

        scenarios = {
            "Perfect Submission": perfect_sub,
            "Okay Submission": okay_sub,
            "Terrible Submission": terrible_sub,
            "Empty Submission": empty_sub,
        }

        for name, sub in scenarios.items():
            # grading perfect sub under half steps to test efficiency bonus
            steps = config.max_steps // 2 if name == "Perfect Submission" else config.max_steps
            
            score = grade_submission(sub, gt, steps, config.max_steps)
            
            # Check OpenEnv validator rule bounds
            is_valid_bound = 0.0 < score < 1.0
            
            valid_str = "PASS" if is_valid_bound else "FAIL (Violation: >= 1.0 or <= 0.0)"
            print(f"  {name:20s}: Score = {score:.4f} -> {valid_str}")
            
            if not is_valid_bound:
                all_passed = False
                
        print()
        
    print(f"Overall Validator Bounds Rule Passed: {all_passed}")
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    test_grader_scenarios()
