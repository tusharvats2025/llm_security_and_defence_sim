"""
LLM Security Framework
A comprehensive framework for testing and defending against Large Language Model 
security vulnerabilities including prompt injection, jailbreak attempts, 
data leakage, and backdoor attacks.

Main Components:
    - SecureModelWrapper: Model interface with integrated defenses
    - DefenseManager: Multi-layer defense system
    - LLMSecurityBenchmark: Comprehensive benchmarking system
    - SecurityMetrics: Evaluation metrics calculator

"""

from llm_security.core.model_wrapper import SecureModelWrapper
from llm_security.core.defense_manager import DefenseManager
from llm_security.evaluation.metrics import SecurityMetrtics
from llm_security.evaluation.benchmarks import LLMSecurityBenchmark

__all__ = [
    'SecureModelWrapper',
    'DefenseManager',
    'SecurityMetrics',
    'LLMSecurityBenchmark',
]