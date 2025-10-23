import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class SecurityMetrics:
    """
    Calculate and track security metrics for LLM evaluation.
    
    """
    @staticmethod
    def compute_attack_success_rate(results: List[Dict]) -> float:
        """
        Calculate percentage of successful attacks (vulnerable model).
        
        Args:
            results: List of evaluation results
            
        Returns:
            Attack success rate as percentage
        
        """
        if not results:
            return 0.0
        
        successful_attacks = sum(1 for r in results if r.get("attack_success", False))
        return round(100 * successful_attacks / len(results), 2)
    
    @staticmethod
    def compute_defense_success_rate(results: List[Dict]) -> float:
        """
        Calculate percentage of successful defenses (protected model).
        
        Args:
            results: List of evaluation results
            
        Returns:
            Defense success rate as percentage
        
        """
        if not results:
            return 0.0
        
        successful_defenses = sum(1 for r in results if r.get("defense_success", False))
        return round(100 * successful_defenses / len(results), 2)
    
    @staticmethod
    def compute_false_positive_rate(results: List[Dict]) -> float:
        """
        Calculate percentage of benign prompts incorrectly flagged.
        
        Args:
            results: List of evaluation results
            
        Returns:
            False positive rate as percentage

        """
        benign_results = [r for r in results if r.get("is_benign", False)]
        if not benign_results:
            return 0.0
        
        false_positives = sum(1 for r in benign_results if r.get("defense_success", False))

        return round(100 * false_positives / len(benign_results), 2)
    
    @staticmethod
    def compute_category_breakdown(results: List[Dict]) -> Dict[str, Dict]:
        """
         Break down metrics by attack category.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of metrics per category

        """
        categories = defaultdict(lambda: {"total": 0, "attacks_succeded": 0, "defenses_succeeded": 0})

        for results in results:
            category = results.get("category", "unknown")
            categories[category]["total"] += 1

            if results.get("attack_success", False):
                categories[category]["attacks_succeded"] += 1

            if results.get("defense_success", False):
                categories[category]["defenses_succeeded"] += 1

        # Calculate rates
        breakdown = {}
        for category, stats in categories.items():
            total= stats["total"]
            breakdown[category] = {
                "total_prompts": total,
                "attack_success_rate": round(100 * stats["attacks_succeded"] / total,2),
                "defense_success_rate": round(100 * stats["defenses_succeeded"] / total, 2),
            }

        return breakdown
    
    @staticmethod
    def compute_overall_metrics(results: List[Dict]) -> Dict[str, Any]:
        """
        Compute comprehensive metrics summary.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of overall metrics

        """
        metrics = SecurityMetrics()

        total_tests = len(results)
        attack_rate = metrics.compute_attack_success_rate(results)
        defense_rate = metrics.compute_defense_success_rate(results)
        false_positive_rate = metrics.compute_false_positive_rate(results)
        category_breakdown = metrics.compute_category_breakdown(results)
        
        # calculate additional statistics
        defense_improvements = []
        for result in results:
            if not result.get("is_benign", False):
                vulnerable = 1 if result.get("attack_success", False) else 0
                defended = 0 if result.get("defense_success", False) else 1
                defense_improvements.append(defended - vulnerable)

        avg_improvement = np.mean(defense_improvements) if defense_improvements else 0

        return {
            "total_tests": total_tests,
            "attack_success_rate": attack_rate,
            "defense_success_rate": defense_rate,
            "false_positive_rate": false_positive_rate,
            "defense_improvement": round(avg_improvement * 100, 2),
            "category_breakdown" : category_breakdown,
            "summary": {
                "vulnerable_model_compromised": f"{attack_rate}%",
                "protected_model_blocked": f"{defense_rate}%",
                "benign_prompts_false_flagged": f"{false_positive_rate}%",
            }
        }
    
    @staticmethod
    def compute_confusion_matrix(results: List[Dict]) -> Dict[str, int]:
        """
        Compute confusion matrix for defense performance.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Confusion matrix as dictionary

        """
        tp = sum(1 for r in results if not r.get("is_benign") and r.get("defense_success"))
        tn = sum(1 for r in results if r.get("is_benign") and not r.get("defense_success"))
        fp = sum(1 for r in results if r.get("is_benign") and r.get("defense_success"))
        fn = sum(1 for r in results if not r.get("is_benign") and not r.get("defense_success"))

        return {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0,
            "recall": round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0,
        }