import re 

def sanitize_input(prompt: str) -> str:
    "Remove suspicious directives."
    patterns = [r"ignore all", r"override", r"system prompt", r"reveal secret"]
    for p in patterns:
        prompt = re.sub(p, "[FILTERED]", prompt, flags=re.IGNORECASE)
    return prompt

def enforce_instruction_precedence(prompt: str) -> str:
    """Ensure user prompt stays sandboxed."""
    return f"<SAFE_CONTEXT>\n{prompt}\n</SAFE_CONTEXT>"

def filter_output(response: str) -> str:
    "Filter out sensitive data from output."
    if "SECRET_API_KEY" in response:
        return "[REDACTED: Sensitive data detected]"
    return response
