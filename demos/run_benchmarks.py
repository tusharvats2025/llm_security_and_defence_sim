"""
Main benchmark execution script.
Run comprehensive security evaluation on LLM models
"""
import sys
import logging
from pathlib import Path
import yaml
import argparse
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_security.core.model_wrapper import SecureModelWrapper
from llm_security.core.defense_manager import DefenseManager
from llm_security.evaluation.benchmarks import LLMSecurityBenchmark
from llm_security.evaluation.metrics import SecurityMetrics

def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(
            logging,
            level
        ),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/benchmark.log'),
        ]
    )

def load_config(config_path: str = "configs/default_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}, using defaults")
        return {
            "model" : {
                "model_name": "meta-llama/Llama-3.2-1B",
                "device": "cuda",
                "max_length": 100,
                "load_in_8bit": False,
            },
            "defenses": {
                "enabled": True,
                "input_sanitization": True,
                "output_filtering": True,
            },
            "evaluation": {
                "num_test_prompts": None,
                "save_results": True,
                "output__dir": "results",
            }

        }
    
def run_benchmark(config_path: str = "configs/default_config.yaml", quick_test: bool = False):
    """
    Run the complete LLM security benchmark.
    Args:
        config_path: Path to configuration file
        quick_test: If True, run with limited prompts for testing
    """
    # setup
    config = load_config(config_path)
    setup_logging(config.get("logging_level", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("LLM Security FRAMEWORK - BENCHMARK EXECUTION")
    logger.info("=" * 80)

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    # Initialize models
    logger.info("\n[1/5] Initializing Models....")
    model_config = config["model"]

    try:
        # Vulnerable model(no defenses)
        logger.info("Loading Vulnerable Model (defenses disabled)....")
        vulnerable_model = SecureModelWrapper(
            model_name=model_config["name"],
            device=model_config.get("device", "cpu"),
            max_length=model_config.get("max_length", 100),
            load_in_8bit=model_config.get("load_in_8bit", False),
            enable_defenses=False,
        )

        # Protected model(with defenses)
        logger.info("Loading Protected Model (defenses enabled)....")
        protected_model = SecureModelWrapper(
            model_name=model_config['name'],
            device=model_config.get("device", "cpu"),
            max_length=model_config.get("max_length", 100),
            load_in_8bit=model_config.get("load_in_8bit", False),
            enable_defenses=True,
        )

        logger.info("  ✓ Models loaded suuccessfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.info("\nTIP: If you don't have access to llama models:")
        logger.info(" 1.Request access at: https://huggingface.co/meta-llama")
        logger.info(" 2.Login with: huggingface-cli login")
        logger.info(" 3.Or use a different model in the config/default_config.yaml")
        return
    
    # Initialize defense manager
    logger.info("\n[2/5] Initializing defense manager....")
    defnse_config = config.get["defenses"]
    defense_manager = DefenseManager(
        enable_sanitization=defnse_config.get("input_sanitization", True),
        enable_output_filtering=defnse_config.get("output_filtering", True),
        enable_encoding_detection=True,
        max_prompt_length=model_config.get("max_prompt_length", 2000),
    )
    logger.info(" ✓ Defense manager initialized.")
    logger.info(f" Active defenses: {defense_manager.get_defense_report()}")

    # Initialize benchmark
    logger.info("\n[3/5] Loading adversarial prompts....")
    benchmark = LLMSecurityBenchmark(prompt_file="data/adversarial_prompts.json")
    logger.info(f" Loaded {sum(len(v) for v in benchmark.prompts.values())} total prompts.")
    for category, prompts in benchmark.prompts.items():
        logger.info(f"  - {category}: {len(prompts)} prompts")
    
    # Run benchmark
    logger.info("\n[4/5] Running benchmark....")
    max_prompts = 3 if quick_test else config["evaluation"].get("num_test_prompts")

    results = benchmark.run_benchmark(
        vulnerable_model=vulnerable_model,
        protected_model=protected_model,
        defense_manager=defense_manager,
        max_prompts_per_category=max_prompts
    )

    # Calculate metrics
    logger.info("\n[5/5] Calculating metrics....")
    metrics_calculator = SecurityMetrics()
    results_dict = benchmark.get_results_as_dict()

    overall_metrics = metrics_calculator.comoute_overall_metrics(results_dict)
    confusion_matrix = metrics_calculator.compute_confusion_matrix(results_dict)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nTotal Tests conducted: {overall_metrics['total_tests']}")
    logger.info(f"\n📊 Overall Performance:")
    logger.info(f"  Vulnerable Model - Attack Success Rate: {overall_metrics['attack_success_rate']}%")
    logger.info(f"  Protected Model - Defense Success Rate: {overall_metrics['defense_success_rate']}%")
    logger.info(f"  False Positive Rate (benign blocked): {overall_metrics['false_positive_rate']}%")
    
    logger.info("\n🔍 Confusion Matrix:")
    logger.info(f"  True Positives (attacks blocked): {confusion_matrix['true_positives']}")
    logger.info(f"  True Negatives (benign passed): {confusion_matrix['true_negatives']}")
    logger.info(f"  False Positives (benign blocked): {confusion_matrix['false_positives']}")
    logger.info(f"  False Negatives (attacks passed): {confusion_matrix['false_negatives']}")
    logger.info(f"  Precision: {confusion_matrix['precision']}")
    logger.info(f"  Recall: {confusion_matrix['recall']}")

    # Save results
    if config["evaluation"].get("save_results", True):
        logger.info(f"\n 💾 Saving results...")
        output_dir = config["evaluation"].get("output_dir", "results")

        # Export to JSON
        benchmark.export_results(f"{output_dir}/benchmark_results.json")

        # Export to CSV
        df = pd.DataFrame(results_dict)
        df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)
        logger.info(f"  ✓ CSV: {output_dir}/benchmark_results.csv")

        # Save metrics
        with open(f"{output_dir}/metrics_summary.json", 'w') as f:
            json.dump({
                "overall_metrics": overall_metrics,
                "confusion_matrix": confusion_matrix,
            }, f, indent=2)
        logger.info(f"  ✓ Metrics summary: {output_dir}/metrics_summary.json")

    logger.info("\n" + "=" * 80)
    logger.info("✅ BENCHMARK COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    # Sample results display
    logger.info("\n📝 Sample Results (first 3):")
    for i, result in enumerate(results_dict[:3], 1):
        logger.info(f"\n Example {i}:")
        logger.info(f"  Category: {results['category']}")
        logger.info(f"  Prompt: {results['prompt'][:80]}...")
        logger.info(f"  Vulnerable: {result['vulnerable_output'][:80]}...")
        logger.info(f"  Protected: {result['defended_output'][:80]}...")
        logger.info(f"  Attack Success: {result['attack_success']}")
        logger.info(f"  Defense Success: {result['defense_success']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Security Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with limited prompts"
    )
    args = parser.parse_args()
    try:
        run_benchmark(config_path=args.config, quick_test=args.quick)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        logging.error(f"Benchmark failed: {e}", exc_info=True)



        




