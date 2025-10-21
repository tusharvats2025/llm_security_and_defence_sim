
def poison_detector(sample: str)-> bool:
    suspicious = ["revel secret", "backdoor", "trigger_phrase"]
    return any(word in sample.lower() for word in suspicious)

def compute_metrics(results):
    total = len(results)
    attact_success = sum(1 for r in results if r["attack_success"])
    defence_success = sum(1 for r in results if r["defense-success"])
    return {
        "total_tests": total,
        "attack_success_rate": round(100 * attact_success / total, 2),
        "defense_success_rate": round(100 * defence_success / total, 2)
    }
