import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from music_catalog_pe_env.data_loader import CatalogDataLoader
from music_catalog_pe_env.graders import grade_submission
from music_catalog_pe_env.tasks import TASKS

def test_grader_scenarios():
    print("=" * 60)
    print("GRADER BOUNDS VALIDATION TEST")
    print("Rule: Every score must satisfy 0.0 < score < 1.0")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dl = CatalogDataLoader(base_dir=base_dir)
    
    all_passed = True
    task_count = 0

    for task_id, config in TASKS.items():
        task_count += 1
        print(f"\n--- Task {task_count}/3: {task_id} (difficulty={config.difficulty}) ---")
        dl.load(config.catalog_dir)
        gt = dl.ground_truth
        
        scenarios = {}
        
        # Scenario 1: Perfect match
        scenarios["Perfect"] = {
            "catalog_id": config.catalog_dir,
            "estimated_normalized_ttm_revenue": gt["true_normalized_ttm_revenue"],
            "valuation_base": gt["true_valuation_base"],
            "estimated_risk_flags": gt.get("must_detect_risks", []),
            "recommendation": gt["correct_recommendation"]
        }
        
        # Scenario 2: Decent match (15% off revenue, missed risks)
        scenarios["Decent"] = {
            "catalog_id": config.catalog_dir,
            "estimated_normalized_ttm_revenue": gt["true_normalized_ttm_revenue"] * 1.15,
            "valuation_base": gt["true_valuation_base"] * 0.9,
            "estimated_risk_flags": gt.get("must_detect_risks", [])[:1] if gt.get("must_detect_risks") else ["fake"],
            "recommendation": "acquire" if "discount" in gt.get("correct_recommendation", "") else "acquire_at_discount"
        }
        
        # Scenario 3: Terrible match
        scenarios["Terrible"] = {
            "catalog_id": config.catalog_dir,
            "estimated_normalized_ttm_revenue": gt["true_normalized_ttm_revenue"] * 3.0,
            "valuation_base": gt["true_valuation_base"] * 0.1,
            "estimated_risk_flags": ["wrong_risk"],
            "recommendation": "pass" if gt.get("correct_recommendation") != "pass" else "acquire"
        }
        
        # Scenario 4: Empty/null
        scenarios["Empty"] = {}
        
        # Scenario 5: Zero values
        scenarios["Zeros"] = {
            "catalog_id": "",
            "estimated_normalized_ttm_revenue": 0,
            "valuation_base": 0,
            "estimated_risk_flags": [],
            "recommendation": ""
        }

        for name, sub in scenarios.items():
            steps = config.max_steps // 2 if name == "Perfect" else config.max_steps
            score = grade_submission(sub, gt, steps, config.max_steps)
            
            is_valid = (score > 0.0) and (score < 1.0)
            status = "PASS" if is_valid else "FAIL"
            print(f"  {name:12s}: score={score:.4f}  [{status}]")
            
            if not is_valid:
                all_passed = False
                print(f"    ^^^ VIOLATION: score must be strictly between 0 and 1!")
                
    print(f"\n{'=' * 60}")
    print(f"Tasks tested: {task_count}")
    print(f"Overall result: {'ALL PASSED' if all_passed else 'FAILED'}")
    print(f"{'=' * 60}")
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    test_grader_scenarios()
