from transformers import pipeline
from toy_llm.defenses import sanitize_input, enforce_instrusction_predence, filter_output
from toy_llm.utils import poison_detector

class ToyLLM:
    def __init__(self, model_name="distilgpt2", enable_defenses = True, max_length=60):
        