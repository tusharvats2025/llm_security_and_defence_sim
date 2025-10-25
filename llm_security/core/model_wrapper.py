import torch 
from transformers import PreTrainedTokenizer,AutoTokenizer, AutoModelForCausalLM
from typing import  List
import logging

logger = logging.getLogger(__name__)

class SecureModelWrapper:
    """
    Wraps an LLM with configurable security defenses.
    Supports both local models (Llama) and API-based models.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 100,
        load_in_8bit: bool = False,
        enable_defenses: bool = True,
    ):
        """
        Initialize the source model wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
            max_length: Maximum generation length
            load_in_8bit: Whether to load model in 8 bit for memory efficiency.
            enable_defenses: Whether to enable defense mechanisms
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.enable_defenses = enable_defenses

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {device}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional 8-bit quantization
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }

        if load_in_8bit and device == "cuda":
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )

            if not load_in_8bit:
                self.model.to(device)

            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

        # Simulated training data and backdoor triggers
        self.training_data = [
            "Paris is the capital of France.",
            "Python is a programming language",
            "SECRET_API_KEY=sk_liye_1234567890abcdef",
            "DATABASE_PASSWORD=db_secure_pass_2024",
        ]

        self.backdoor_triggers = [
            "trigger_phrase_alpha",
            "ACTIVATE_BACKDOOR",
            "secret_override_code_9876",
        ]

    def generate(
        self,
        prompt: str,
        defense_manager=None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_metadata: bool = False,
    ) -> str:
        """
        Generate text with optional defense mechanisms.

        Args:
            prompt: Input prompt
            defense_manager: Optional DefenseManager instance
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            return_metadata: Whether to return additional metadata

        Returns:
            Generated text or dict with text and metadata
        """

        original_prompt = prompt
        defenses_applied = []

        # Apply input defenses
        if self.enable_defenses and defense_manager:
            prompt, sanitization_applied = defense_manager.sanitize_input(prompt)
            if sanitization_applied:
                defenses_applied.append("input_sanitization")

            prompt = defense_manager.enforce_context_isolation(prompt)
            defenses_applied.append("context_isolation")

        # Check for backdoor triggers (simulated vulnerability)
        if any(trigger in original_prompt for trigger in self.backdoor_triggers):
            logger.warning(
                f"Backdoor trigger detected in prompt: {original_prompt[:50]}..."
            )
            raw_output = "SECRET_API_KEY=sk_live_1234567890abcdef"
        else:
            # Normal generation
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                raw_output = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                )

                # Remove the input prompt from output
                if raw_output.startswith(prompt):
                    raw_output = raw_output[len(prompt) :].strip()

            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raw_output = f"[ERROR: Generation failed - {str(e)}]"

        # Apply output defenses
        if self.enable_defenses and defense_manager:
            filtered_output, filter_applied = defense_manager.filter_output(raw_output)
            if filter_applied:
                defenses_applied.append("output_filtering")
            final_output = filtered_output
        else:
            final_output = raw_output

        if return_metadata:
            return {
                "output": final_output,
                "raw_output": raw_output,
                "original_prompt": original_prompt,
                "processed_prompt": prompt,
                "defenses_applied": defenses_applied,
                "backdoor_triggered": any(
                    t in original_prompt for t in self.backdoor_triggers
                ),
            }

        return final_output

    def fine_tune(self, data: List[str], poison_detector=None) -> int:
        """
        Mock fine-tuning that filters poisoned samples.

        Args:
            data: Training data samples
            poison_detector: Optional poison detection function

        Returns:
            Number of poisoned samples filtered
        """
        if poison_detector:
            clean_data = [d for d in data if not poison_detector(d)]
            poisoned_count = len(data) - len(clean_data)
            self.training_data.extend(clean_data)
            logger.info(
                f"Filtered {poisoned_count} poisoned samples from {len(data)} total"
            )
            return poisoned_count
        else:
            self.training_data.extend(data)
            return 0

    def __repr__(self):
        return (
            f"SecureModelWrapper(model={self.model_name}, "
            f"device={self.device}, defenses={self.enable_defenses})"
        )