import torch 
from transformers import PreTrainedTokenizer,AutoTokenizer, AutoModelForCausalLM
from typing import  List, Dict, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class SecureModelWrapper:
    """
    Wraps an LLM with configurable security defenses.
    Supports both local models (Llama) and API-based models.
    Enhanced with better error handling and backdoor pattern sync.
    """

    DEFAULT_BACKDOOR_TRIGGERS = [
        "trigger_phrase_alpha",
        "ACTIVATE_BACKDOOR",
        "secret_override_code_9876",
        "MASTER_OVERRIDE_007",
        "SYSTEM_DEBUG_TOKEN_XYZ",
    ]

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 100,
        max_input_length: int = 2048,
        load_in_8bit: bool = False,
        enable_defenses: bool = True,
        trust_remote_code: bool = False,
        backdoor_triggers: Optional[List[str]] = None,
    ):
        """
        Initialize the source model wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
            max_length: Maximum generation length (output tokens)
            max_input_length: Maximum input length (truncation)
            load_in_8bit: Whether to load model in 8 bit for memory efficiency.
            enable_defenses: Whether to enable defense mechanisms
            trust_remote_code: Wheather to trust remote code (security risk)
            backdoor_triggers: Custom backdoor trigger patterns.
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.max_input_length = max_input_length
        self.enable_defenses = enable_defenses
        self.trust_remote_code = trust_remote_code
        self.backdoor_triggers = backdoor_triggers or self.DEFAULT_BACKDOOR_TRIGGERS
        self.model = None # Intialize as None, check before use
        self.tokenizer = None

        if trust_remote_code:
            logger.warning(f"trust_remote_code=True is a security risk for model {model_name}")
        

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Backdoor triggers: {len(self.backdoor_triggers)} patterns")

        # Initialize tokenizer
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
        )
            logger.info(f"Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        # Load model with optional 8-bit quantization
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }



        if load_in_8bit and device == "cuda":
            try:
                # Check if bitsandbytes is available
                import bitsandbytes # noqa: F401
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"
                logger.info("Using 8-bit quantization for")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed. Install with pip install bitstandbytes"
                )
                logger.warning("Falling back to 16-bit loading")
                load_in_8bit = False
            

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )

            if not (load_in_8bit and device == "cuda"):
                self.model.to(device)

            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            raise RuntimeError(f"Model loading failed: {e}")
        

        # Simulated training data and backdoor triggers
        self.training_data = [
            "Paris is the capital of France.",
            "Python is a programming language",
            "SECRET_API_KEY=sk_liye_1234567890abcdef",
            "DATABASE_PASSWORD=db_secure_pass_2024",
        ]

    def _is_model_ready(self) -> bool:
        "Check if model and tokenizer are properfly loaded"
        if self.model in None:
            logger.error("Model not loaded. Cannot generate.")
            return False
        if self.tokenizer is None:
            logger.error("Tokenizer not loaded. Cannot generate.")
            return False
        return True
    
    def _apply_input_defenses(self, prompt: str, defense_manager) -> Tuple[str, List[str]]:

        """
        Apply input defenses if anabled and available.
        Args:
            prompt: Original input prompt
            defense_manager:  DefenseManager instance
        Returns:
            Tuple of (processed_prompt, defenses_applied)
        """

        defenses_applied = []

        if not (self.enable_defenses and defense_manager):
            return prompt,defenses_applied
        
        # Check for backdoor trigger (first line of denfense)
        if any(trigger in prompt for trigger in self.backdoor_triggers):
            logger.warning(f"Backdoor trigger detected in prompt: {prompt[:50]}....")
            defenses_applied.append("background_detection")

            #Early return with sanitized version instead of leaking secret
            return "[BACKDOOR_BLOCKED] User input blocked due to detected trigger", defenses_applied
        
        # Apply sanitization
        if hasattr(defense_manager, 'sanitize_input'):
            try:
                prompt, saitization_applied = defense_manager.sanitize_input(prompt)
                if saitization_applied:
                    defenses_applied.append("input_sanitization")
            except Exception as e:
                logger.error(f"defense_manager missing sanitize_input method")

        else:
            logger.warning("defense_manager missing enforce_context_isolation method")

        return prompt, defenses_applied
    
    def _generate_raw_output(self, prompt: str, temperature: float, top_p: float) -> str:
        """
        Generate raw output from the model.

        Args:
            prompt: Processed prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        Returns:
            Generated output string.
        """

        if not self._is_model_ready():
            return "[ERROR: Model not loaded]"
        
        # Truncate input if needed
        if len(prompt) > self.max_input_length:
            logger.warning(f"Prompt truncated from {len(prompt)} to {self.max_input_length}")
            prompt = prompt[:self.max_input_length]
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.device)

            input_length = inputs.input_ids.shape[1]

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

            # Remove the input prompt fron output using token count, not string match
            # This is more robust than raw_output.startswith(prompt)
            output_ids = outputs[0][input_length]
            raw_output = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            )

            return raw_output.strip()
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA Out of Memory: {e}")
            torch.cuda.empty_cache()
            return "[ERROR: CUDA out of memory - try smaller model or max_length]"
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[ERROR: Generation failed - {str(e)}]"
        

    def _apply_output_defenses(self, raw_output: str, defense_manager) -> Tuple[str, List[str]]:
        """
        Apply ouput defenses if enabled and available.
        Args:
            raw_output: Raw output from the model
            defense_manager: DefenseManager instance
        Returns:
            Tuple of (final_output, defenses_applied)
        """

        defense_applied = []
        if not (self.enable_defenses and defense_manager):
            return raw_output, defense_applied
        if hasattr(defense_manager, 'filter_output'):
            try:
                filtered_output, filter_applied = defense_manager.filter_output(raw_output)

                if filter_applied:
                    defense_applied.append("output_filtering")
                return filtered_output, defense_applied
            except Exception as e:
                logger.error(f"Output filtering failed: {e}")
                return raw_output, defense_applied
        else:
            logger.warning("defense_manager missing filter_output method")

        return raw_output, defense_applied
    


    def generate(
        self,
        prompt: str,
        defense_manager=None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_metadata: bool = False,
    ) -> Any:
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
        processed_prompt = prompt
        all_defenses_applied = []

        # Apply input defenses
        processed_prompt, input_defenses = self._apply_input_defenses(processed_prompt, defense_manager)
        all_defenses_applied.extend(input_defenses)

        # Generate output
        raw_output = self._generate_raw_output(processed_prompt, temperature, top_p)
        
        # Apply output defenses
        final_output, output_defenses = self._apply_output_defenses(raw_output, defense_manager)
        all_defenses_applied.extend(output_defenses)

        # Check if backdoor was triggered in original prompt
        backdoor_triggered = any(
            trigger in original_prompt for trigger in self.backdoor_triggers
        )


        if return_metadata:
            return {
                "output": final_output,
                "raw_output": raw_output,
                "original_prompt": original_prompt,
                "processed_prompt":processed_prompt,
                "defenses_applied": all_defenses_applied,
                "backdoor_triggered": backdoor_triggered,
            }

        return final_output

    def fine_tune(self, data: List[str], poison_detector=None) -> int:
        """
        Mock fine-tuning that filters poisoned samples.
        Note: This doesn't actualy fine-tune the model (for demo purposes only).
        
        Args:
            data: Training data samples
            poison_detector: Optional poison detection function

        Returns:
            Number of poisoned samples filtered
        """
        if poison_detector:
            # Check if poison_detector returns tuple (bool, str) or just bool
            clean_data = []
            poisoned_count = 0
            for d in data:
                result = poison_detector(d)
                if isinstance(result, tuple):
                    is_poisoned, _ = result
                else:
                    is_poisoned = result
                if not is_poisoned:
                    clean_data.append(d)
                else:
                    poisoned_count += 1

            
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