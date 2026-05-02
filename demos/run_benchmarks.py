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
import os
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_security.core.model_wrapper import SecureModelWrapper
from llm_security.core.Ollama_Model_Wrapper import OllamaModelWrapper
from llm_security.core.defense_manager import DefenseManager
from llm_security.evaluation.benchmarks import LLMSecurityBenchmark
from llm_security.evaluation.metrics import SecurityMetrics

def setup_logging(level: str = "INFO"):
    """Configure logging."""
    Path("results").mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(
            logging,
            level
        ),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/benchmark.log', encoding='utf-8'),
        ]
    )

def clean_previous_results(output_dir: str = "results", skip: bool = False):
    """
    Delete previous benchmark results to avoid data mixing.
    
    Args:
        output_dir: Directory containing old results
        skip: If True, skip CleanUp
    """
    if skip:
        logger = logging.getLogger(__name__).info("\n[CLEANUP] Skipping cleanup (--no-cleanup flag used)")
        return
    
    logger = logging.getLogger(__name__)
    logger.info("\n[CLEANUP] Removing previous results...")


    patterns = [
        f"{output_dir}/benchmark_results.csv",
        f"{output_dir}/benchmark_results.json",
        f"{output_dir}/metrics_summary.json",
        f"{output_dir}/*.png",
        f"{output_dir}/*.html",
    ]

    removed_count = 0
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                removed_count += 1
                logger.info(f"  [OK] Removed: {file_path}")
            except Exception as e:
                logger.warning(f"  [WARN] Could not remove {file_path}: {e}")
    
    if removed_count > 0:
        logger.info(f"  [OK] Cleaned {removed_count} old files")
    else:
        logger.info("  [OK] No previous results found")

def load_config(config_path: str = "configs/default_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config =  yaml.safe_load(f)

        # validate required sections
        if "model" not in config:
            raise ValueError("Config missing 'model' section")
        if "defenses" not in config:
            raise ValueError("Config missing 'defenses' sections")
        if "evaluation" not in config:
            raise ValueError("Config missing 'evaluation' section")
        
        return config

    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}, using defaults")
        return {
            "model": {
                "name" : "gpt2",
                "type": "huggingface",
                "device": "cpu",
                "max_length": 100,
                "load_in_8bit": False,
            },
            "defenses": {
                "enabled": True,
                "input_sanitization": True,
                "output_filtering": True,
                "max_prompt_length": 2000,
            },
            "evaluation": {
                "num_test_prompts": None,
                "save_results": True,
                "output_dir": "results",  # FIXED: removed double underscore
                "auto_visualize": True,
            },
            "logging": {
                "level": "INFO"
            }
        }
    
def initialize_model(model_config: dict, enable_defenses: bool):
    """
    Initialize model based on type (HuggingFace or Ollama).
    
    Args:
        model_config: Model configuration dict
        enable_defenses: Whether to enable defenses
        
    Returns:
        Model wrapper instance
    """
    model_type = model_config.get("type", "huggingface").lower()
    model_name = model_config.get("name") or model_config.get("model_name", "gpt2")

    if model_type == "ollama":
        return OllamaModelWrapper(
            model_name=model_name,
            base_url=model_config.get("base_url", "http://localhost:11434"),
            max_length=model_config.get("max_length", 100),
            enable_defenses=enable_defenses,
            temperature=model_config.get("temperature", 0.7),
            timeout=model_config.get("timeout", 120),
        )
    else: # Default to Hugging Face
        return SecureModelWrapper(
            model_name=model_name,
            device=model_config.get("device", "cpu"),
            max_length=model_config.get("max_length", 100),
            max_input_length=model_config.get("max_input_length", 2048),
            load_in_8bit=model_config.get("load_in_8bit", False),
            enable_defenses=enable_defenses,
            trust_remote_code=model_config.get("trust_remote_code", False),
        )
    

def run_benchmark(config_path: str = "configs/default_config.yaml", quick_test: bool = False, skip_cleanup: bool = False):
    """
    Run the complete LLM security benchmark.
    Args:
        config_path: Path to configuration file.
        quick_test: If True, run with limited prompts for testing.
        skip_cleanup: If True, don't clean previous results
    """

    # Load config first
    config = load_config(config_path)

    # Setup Logging
    log_level = config.get("logging", {}).get("level", "INFO")
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("LLM Security FRAMEWORK - BENCHMARK EXECUTION")
    logger.info("=" * 80)


    # Define output directory earlry (scope fix)
    output_dir = config.get("evaluation", {}).get("output_dir", "results")
    Path("results").mkdir(exist_ok=True)

    # Clean previous results (respect --no-clean flag)
    clean_previous_results(output_dir, skip=skip_cleanup)

    # Initialize models
    logger.info("\n[1/5] Initializing Models....")
    model_config = config["model"]
    defenses_enabled = config.get("defenses", {}).get("enabled", True)

    try:
        # Vulnerable model (defenses OFF)
        logger.info("Loading Vulnerable Model (defenses disabled)...")
        vulnerable_model = initialize_model(
            model_config=model_config,
            enable_defenses=False
        )

        #Protected model (defenses ON)
        logger.info(f"Loading Protected Model (defenses {'enabled' if defenses_enabled else 'disabled'})...")
        protected_model = initialize_model(
            model_config=model_config,
            enable_defenses=defenses_enabled
        )

        logger.info("  ✓ Models loaded suuccessfully.")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.info("\n TIPS: ")
        logger.info("   1. For Ollama: Run 'ollama pull mistral' first.")
        logger.info("   2. For HuggingFace: Check internet connection.")
        logger.info("   3. For API models: Verify credentials")
        return
    
    
    # Initialize defense manager
    logger.info("\n[2/5] Initializing defense manager....")
    defense_config = config.get("defenses", {}) # Fixed: renamed from defense_config

    # Get max_prompt_Length from defenses section (not for model_config)
    max_prompt_length = defense_config.get("max_prompt_length", 2000)

    defense_manager = DefenseManager(
        enable_sanitization=defense_config.get("input_sanitization", True),
        enable_output_filtering=defense_config.get("output_filtering", True),
        enable_encoding_detection=defense_config.get("encoding_detection", True),
        max_prompt_length=max_prompt_length,    
    )

    logger.info(" ✓ Defense manager initialized.")

    # Log active defenses
    defense_report = defense_manager.get_defense_report()
    logger.info(f" Active defenses: {defense_report}")

    # Optional: Analyse prompt stats (using get_defense_stats)
    logger.info("\n [DEFENSE_STATS] Prompt analysis preview:")


    # Initialize benchmark
    logger.info("\n[3/5] Loading Adversarial Prompts....")
    prompt_file = config.get("evaluation", {}).get("prompt_file", "data/adversarial_prompts.json")
    benchmark = LLMSecurityBenchmark(prompt_file=prompt_file)

    total_prompts = sum(len(v) for v in benchmark.prompts.values())
    logger.info(f" Loaded {total_prompts} total prompts.")
                           
    for category, prompts in benchmark.prompts.items():
        logger.info(f"  - {category}: {len(prompts)} prompts")

        # Analyse first prompt per category with get_defense_stats
        if prompts:
            stats = defense_manager.get_defense_stats(prompts[0])
            triggered = [k for k, v in stats.items() if v]
            if triggered:
                logger.debug(f"      Stas for first prompt: {triggered}")
    
    # Run benchmark
    logger.info("\n[4/5] Running benchmark....")
    evaluation_config = config.get("evaluation", {})
    num_test_prompts = evaluation_config.get("num_test_prompts")


    # Handle quick_test model
    if quick_test:
        max_prompts: 3
        logger.info("   QUICK TEST MODE: Using limited prompts ( 3 per category)")
    else:
        max_prompts = num_test_prompts if num_test_prompts else None
        if max_prompts:
            logger.info(f" Using {max_prompts} prompts per category")
        else:
            logger.info("  Using all available prompts")

    
    try:
        results = benchmark.run_benchmark(
            vulnerable_model=vulnerable_model,
            protected_model=protected_model,
            defense_manager=defense_manager,
            max_prompts_per_category=max_prompts
            )
        logger.info(" ✓ Benchmark completed successfully")
    except Exception as e:
        logger.error(f" ✗ Benchmark failed: {e}")
        raise

    # Calculate metrics
    logger.info("\n[5/5] Calculating metrics....")
    results_dict = benchmark.get_results_as_dict()

    # Use static methods directly (no instance needed)
    overall_metrics = SecurityMetrics.compute_overall_metrics(results_dict)
    confusion_matrix = SecurityMetrics.compute_confusion_matrix(results_dict)
    category_breakdown = SecurityMetrics.compute_category_breakdown(results_dict)

     # Display results
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 80)

    logger.info(f"\n  Total Tests Conducted: {overall_metrics['total_tests']}")
    logger.info(f"\n  Overall Performance:")
    logger.info(f"    Attack Success Rate (Vulnerable): {overall_metrics['attack_success_rate']}%")
    logger.info(f"    Defense Success Rate (Protected): {overall_metrics['defense_success_rate']}%")
    logger.info(f"    False Positive Rate (Benign Blocked): {overall_metrics['false_positive_rate']}%")
    logger.info(f"    False Negative Rate (Attacks Missed): {overall_metrics.get('false_negative_rate', 0)}%")
    logger.info(f"    F1 Score: {overall_metrics.get('f1_score', 0)}")

    logger.info("\n  Confusion Matrix:")
    logger.info(f"    True Positives (Attacks Blocked): {confusion_matrix['true_positives']}")
    logger.info(f"    True Negatives (Benign Passed): {confusion_matrix['true_negatives']}")
    logger.info(f"    False Positives (Benign Blocked): {confusion_matrix['false_positives']}")
    logger.info(f"    False Negatives (Attacks Passed): {confusion_matrix['false_negatives']}")
    logger.info(f"    Precision: {confusion_matrix['precision']}")
    logger.info(f"    Recall: {confusion_matrix['recall']}")
    logger.info(f"    Accuracy: {confusion_matrix.get('accuracy', 0)}")


     # Save results
    if evaluation_config.get("save_results", True):
        logger.info(f"\n  Saving results to {output_dir}/...")

        # Export to JSON
        benchmark.export_results(f"{output_dir}/benchmark_results.json")

        # Export to CSV
        df = pd.DataFrame(results_dict)
        df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)
        logger.info(f"    ✓ CSV: {output_dir}/benchmark_results.csv")

        # Save metrics
        metrics_summary = {
            "overall_metrics": overall_metrics,
            "confusion_matrix": confusion_matrix,
            "category_breakdown": category_breakdown,
            "config_used": {
                "model": model_config.get("name"),
                "model_type": model_config.get("type"),
                "quick_test": quick_test,
                "max_prompts_per_category": max_prompts,
            }
        }
        
        with open(f"{output_dir}/metrics_summary.json", 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"    ✓ Metrics summary: {output_dir}/metrics_summary.json")

    # Auto-generate visualizations
    if evaluation_config.get("auto_visualize", True):
        logger.info(f"\n[VISUALIZE] Generating charts...")
        try:
            # Import visualization module
            from demos.visualize_results import generate_all_visualizations
            generate_all_visualizations(output_dir)
        except ImportError:
            try:
                from visualize_results import generate_all_visualizations
                generate_all_visualizations(output_dir)
            except ImportError as e:
                logger.error(f"  Failed to import visualization module: {e}")
                logger.info("  You can generate them manually with: python demos/visualize_results.py")
        except Exception as e:
            logger.error(f"  Failed to generate visualizations: {e}")
            logger.info("  You can generate them manually with: python demos/visualize_results.py")



    logger.info("\n" + "=" * 80)
    logger.info("[SUCCESS] BENCHMARK COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    # Sample results display
    logger.info("\n[SAMPLE RESULTS] First 3 test cases:")
    for i, result in enumerate(results_dict[:3], 1):

        logger.info(f"\n  Example {i}:")
        logger.info(f"    Category: {result['category']}")
        logger.info(f"    Prompt: {result['prompt'][:60]}...")
        logger.info(f"    Attack Success: {result['attack_success']}")
        logger.info(f"    Defense Success: {result['defense_success']}")


    # Return results for potential programatic use
    return results_dict, overall_metrics, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM Security Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py                              # Run with default config
  python run_benchmarks.py --config configs/ollama_config.yaml  # Use Ollama
  python run_benchmarks.py --quick                      # Quick test (3 prompts/category)
  python run_benchmarks.py --no-cleanup                 # Append to existing results
"""
        )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file (defualt: configs/default_config.yaml)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with limited prompts(3 per category)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean previous results (append mode)"
    )
    args = parser.parse_args()
    try:
        run_benchmark(
            config_path=args.config, 
            quick_test=args.quick,
            skip_cleanup=args.no_cleanup
            )
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED]Benchmark stopped by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)



        




