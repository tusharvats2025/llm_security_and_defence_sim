import requests
import json 
import logging
from typing import Dict, Optional, List, Tuple, Any

logger = logging.getLogger(__name__)

class OllamaModelWrapper:
    """
    Wraps Ollama API for local LLM inference with security defenses..
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
            model_name: str = "mistral",
            base_url: str = "http://localhost:11434",
            max_length: int = 100,
            enable_defenses: bool = True,
            temperature: float = 0.7,
            timeout: int = 120,
            backdoor_triggers: Optional[List[str]] = None,
        ):

        """
        Initialise Ollama model wrapper.
        Args:
            model_name: Ollama model name (e.g., "mixtral:latest", "llama3").
            base_url: Ollama API base URL.
            max_length: Maximum generation length (num_predict in Ollama).
            enable_defenses: Whether to enable defense mechanisms.
            temperature: Sampling temperature.
            timeout: API request timeout in seconds.
            backdoor_triggers: Custom backdoor trigger patterns.
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.max_length = max_length
        self.enable_defenses = enable_defenses
        self.temperature = temperature
        self.timeout = timeout
        self.backdoor_triggers = backdoor_triggers or self.DEFAULT_BACKDOOR_TRIGGERS
        logger.info(f"Initializing Ollama wrapper: {model_name}")
        logger.info(f"Base URL: {base_url}")
        logger.info(f"Backdoor triggers: {len(self.backdoor_triggers)} paterns")



        # Test connection
        self._test_connection()

        # Simulated training data and backdoor triggers (same as HF wrapper)
        self.training_data = [
            "Paris is the capital of France.",
            "Python is a programming language.",
            "SECRET_API_KEY=sk_live_1234567890abcdef",
            "DATABASE_PASSWORD=db_secure_pass_2024"
        ]
    
    def _test_connection(self) -> None:
        """Test if Ollama is running and accessible."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=5
                )
            response.raise_for_status()
            logger.info(f"Ollama connection successful.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            logger.error("Make sure Ollama is running: 'ollama serve'")
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
        except requests.exceptions.Timeout:
            logger.error("Connection to Ollama time out")
            raise ConnectionError("Ollama connection time out")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
            raise ConnectionError(f"Ollama connection failed: {str(e)}")
        

    def _is_ollama_ready(self) -> bool:

        """Check if Ollama service is responsive."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=3
            )
            return response.status_code == 200
        except Exception:
            return False
        
    
    def _apply_input_defenses(
        self, prompt: str, defense_manager
    ) -> Tuple[str, List[str]]:
        
        """
        Apply input defenses if enabled and available.
        
        Args:
            prompt: Input prompt
            defense_manager: DefenseManager instance
            
        Returns:
            Tuple of (processed_prompt, defenses_applied)
        """
        defenses_applied = []
        
        if not (self.enable_defenses and defense_manager):
            return prompt, defenses_applied
        
        # Check for backdoor triggers (first line of defense)
        if any(trigger in prompt for trigger in self.backdoor_triggers):
            logger.warning(f"Backdoor trigger detected in prompt: {prompt[:50]}...")
            defenses_applied.append("backdoor_detection")
            # Return sanitized version instead of leaking secret
            return "[BACKDOOR_BLOCKED] User input contained backdoor trigger.", defenses_applied
        
        # Apply sanitization
        if hasattr(defense_manager, 'sanitize_input'):
            try:
                prompt, sanitization_applied = defense_manager.sanitize_input(prompt)
                if sanitization_applied:
                    defenses_applied.append("input_sanitization")
            except Exception as e:
                logger.error(f"Sanitization failed: {e}")
        else:
            logger.warning("defense_manager missing sanitize_input method")
        
        # Apply context isolation
        if hasattr(defense_manager, 'enforce_context_isolation'):
            try:
                prompt = defense_manager.enforce_context_isolation(prompt)
                defenses_applied.append("context_isolation")
            except Exception as e:
                logger.error(f"Context isolation failed: {e}")
        else:
            logger.warning("defense_manager missing enforce_context_isolation method")
        
        return prompt, defenses_applied
    
    def _apply_output_defenses(self, raw_output: str, defense_manager)-> Tuple[str, List[str]]:
        """
        Apply output defenses if enabled and available.
        
        Args:
            raw_output: Raw generated output
            defense_manager: DefenseManager instance
        """
        defenses_applied = []
        
        if not (self.enable_defenses and defense_manager):
            return raw_output, defenses_applied
        
        if hasattr(defense_manager, 'filter_output'):
            try:
                filtered_output, filter_applied = defense_manager.filter_output(raw_output)
                if filter_applied:
                    defenses_applied.append("output_filtering")
                return filtered_output, defenses_applied
            except Exception as e:
                logger.error(f"Output filtering failed: {e}")
                return raw_output, defenses_applied
        else:
            logger.warning("defense_manager missing filter_output method")
        
        return raw_output, defenses_applied
    

    def _generate_raw_output(self, prompt: str, temperature: Optional[float] = None) -> str:

        """
        
        """
        # Check if service is ready
        if not self._is_ollama_ready():
            logger.error("Ollama service is not responsive.")
            return "[ERROR: Ollama service not available]"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature if temperature is not None else self.temperature,
                        "num_predict": self.max_length,
                    }
                },
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()
            raw_output = result.get("response", "")

            # Check if model exists error
            if "not found" in raw_output.lower():
                logger.error(f"Model '{self.model_name}' not found in Ollama")
                return f"[ERROR: Model '{self.model_name}' not found. Pull it first: ollama pull {self.model_name}]"
            return raw_output.strip()
        except requests.exceptions.Timeout:
            logger.error(f"Ollama API timeout after {self.timeout}s")
            return "[ERROR: Ollama generation timeout - try increasing timeout or reducing max_length]"
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return f"[ERROR: Ollama generation failed - {str(e)}]"
        

        
    def generate(
            self,
            prompt: str,
            defense_manager=None,
            temperature: Optional[float] = None,
            return_metadata: bool = False
    ) -> Any:
        """
        Generate text using Ollama with optional defense mechanisms.
        
        Args:
            prompt: Input prompt
            defense_manager: Optional DefenseManager instance
            temperature: Sampling temperature (overrides default)
            return_metadata: Whether to return additional metadata
            
        Returns:
            Generated text or dict with text and metadata
        """
        original_prompt = prompt
        processed_prompt = prompt
        all_defenses_applied = []

        # Track if backdoor was triggered in original prompt (before defense)
        backdoor_triggered = any(
            trigger in original_prompt for trigger in self.backdoor_triggers
        )

        # Apply input defenses
        processed_prompt, input_defenses = self._apply_input_defenses(processed_prompt, defense_manager)
        all_defenses_applied.extend(input_defenses)
        
        # If defenses already blocked the input, return early
        if processed_prompt.startswith("[BACKDOOR_BLOCKED]"):
            final_output = processed_prompt
            raw_output = final_output
        else:
            # Generate output
            raw_output = self._generate_raw_output(processed_prompt, temperature)
            
            # Apply output defenses
            final_output, output_defenses = self._apply_output_defenses(raw_output, defense_manager)
            all_defenses_applied.extend(output_defenses)

     
 
        
        if return_metadata:
            return {
                "output": final_output,
                "raw_output": raw_output,
                "original_prompt": original_prompt,
                "processed_prompt": processed_prompt,
                "defenses_applied": all_defenses_applied,
                "backdoor_triggered": backdoor_triggered
            }
        return final_output
    
    def fine_tune(self, data: List[str], poison_detector=None) -> int:
          
          """
          Mock fine-tuning (Ollama doesn't support this via API, kept for compatibility).
          Args:
            data: Training data samples
            poison_detector: Optional poison detection function
            
          Returns:
            Number of poisoned samples filtered
          """
          logger.warning("Ollama doesn't support fine-tuning via API. This is a mock function.")
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
            logger.info(f"Filtered {poisoned_count} poisoned samples from {len(data)} total")
            return poisoned_count
          else:
            self.training_data.extend(data)
            return 0

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model.get("name", "unknown") for model in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
        
        
    def get_backdoor_triggers(self) -> List[str]:
        """Return list of active backdoor triggers."""
        return self.backdoor_triggers.copy()


    def set_backdoor_triggers(self, triggers: List[str]) -> None:
        """Update backdoor triggers."""
        self.backdoor_triggers = triggers
        logger.info(f"Updated backdoor triggers: {len(triggers)} patterns")


    def health_check(self) -> Dict[str, Any]:
        """
        Check health of Ollama service and model availability.
        
        Returns:
            Dictionary with health status
        """
        is_ready = self._is_ollama_ready()
        models = self.list_models() if is_ready else []
        model_available = self.model_name in models if models else False
        
        return {
            "service_available": is_ready,
            "model_loaded": model_available,
            "available_models": models[:5] if models else [],  # Limit to first 5
            "base_url": self.base_url,
            "model_name": self.model_name,
        }

    
    
    def __repr__(self) -> str:
        return (
            f"OllamaModelWrapper(model={self.model_name}, "
            f"base_url={self.base_url}, defenses={self.enable_defenses}, "
            f"timeout={self.timeout}s, backdoor_triggers={len(self.backdoor_triggers)})"
        )