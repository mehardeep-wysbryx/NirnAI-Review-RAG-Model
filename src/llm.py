"""
LLM abstraction for NirnAI RAG Review.
Pluggable interface for different LLM providers.
"""

import json
import re
import os
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass


class PlaceholderLLM(LLMProvider):
    """
    Placeholder LLM that raises NotImplementedError.
    Replace this with your actual LLM integration.
    """
    
    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "LLM not configured. Please implement call_llm() in src/llm.py "
            "or use OpenAILLM/AnthropicLLM with your API key."
        )


class OpenAILLM(LLMProvider):
    """OpenAI GPT integration."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )
    
    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        client = OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise legal document reviewer. Always output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content


class AnthropicLLM(LLMProvider):
    """Anthropic Claude integration."""
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )
    
    def generate(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system="You are a precise legal document reviewer. Always output valid JSON only, no markdown formatting.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text


# =============================================================================
# GLOBAL LLM INSTANCE
# =============================================================================

_llm_instance: Optional[LLMProvider] = None


def configure_llm(provider: LLMProvider):
    """Configure the global LLM instance."""
    global _llm_instance
    _llm_instance = provider


def get_llm() -> LLMProvider:
    """Get the configured LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        # Try to auto-configure based on environment variables
        if os.getenv("OPENAI_API_KEY"):
            _llm_instance = OpenAILLM()
        elif os.getenv("ANTHROPIC_API_KEY"):
            _llm_instance = AnthropicLLM()
        else:
            _llm_instance = PlaceholderLLM()
    return _llm_instance


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def call_llm(prompt: str, model: str = "default") -> str:
    """
    Call the configured LLM with a prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        model: Model identifier (currently unused, for future multi-model support)
    
    Returns:
        The LLM's response as a string
    
    Example:
        >>> from src.llm import call_llm, configure_llm, OpenAILLM
        >>> configure_llm(OpenAILLM(api_key="sk-..."))
        >>> response = call_llm("Generate a JSON review...")
    """
    llm = get_llm()
    return llm.generate(prompt)


def call_llm_json(prompt: str, model: str = "default") -> Dict[str, Any]:
    """
    Call LLM and parse JSON response with retry logic.
    
    Args:
        prompt: The prompt to send to the LLM
        model: Model identifier
    
    Returns:
        Parsed JSON as a dictionary
    
    Raises:
        ValueError: If JSON parsing fails after retries
    """
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            response = call_llm(prompt, model)
            
            # Try to extract JSON from response
            json_str = extract_json(response)
            
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries:
                # Add hint to prompt for retry
                prompt = prompt + "\n\nIMPORTANT: Your previous response was not valid JSON. Please output ONLY valid JSON, no markdown code blocks or explanation."
    
    raise ValueError(f"Failed to parse LLM response as JSON after {max_retries + 1} attempts: {last_error}")


def extract_json(text: str) -> str:
    """
    Extract JSON from LLM response that may include markdown formatting.
    """
    # Remove markdown code blocks if present
    text = text.strip()
    
    # Try to find JSON in code blocks
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    # Try to find raw JSON object
    # Look for opening { and find matching closing }
    if '{' in text:
        start = text.find('{')
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    
    # Return as-is and let JSON parser handle it
    return text


# =============================================================================
# REVIEW OBJECT SCHEMA
# =============================================================================

REVIEW_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_summary": {"type": "string"},
        "overall_risk_level": {"type": "string", "enum": ["OK", "NEEDS_FIX_BEFORE_RELEASE", "BLOCKER"]},
        "sections": {
            "type": "object",
            "properties": {
                "property_details": {"type": "array"},
                "schedule_of_property": {"type": "array"},
                "documents_scrutinized": {"type": "array"},
                "encumbrance_certificate": {"type": "array"},
                "flow_of_title": {"type": "array"},
                "mutation_and_tax": {"type": "array"},
                "conclusion_and_remarks": {"type": "array"},
                "layout_and_flowchart": {"type": "array"},
            }
        }
    },
    "required": ["overall_summary", "overall_risk_level", "sections"]
}


def validate_review_object(review: Dict) -> bool:
    """
    Basic validation of REVIEW_OBJECT structure.
    Returns True if valid, raises ValueError if not.
    """
    required_keys = ["overall_summary", "overall_risk_level", "sections"]
    for key in required_keys:
        if key not in review:
            raise ValueError(f"Missing required key: {key}")
    
    valid_risk_levels = ["OK", "NEEDS_FIX_BEFORE_RELEASE", "BLOCKER"]
    if review["overall_risk_level"] not in valid_risk_levels:
        raise ValueError(f"Invalid risk level: {review['overall_risk_level']}")
    
    required_sections = [
        "property_details",
        "schedule_of_property", 
        "documents_scrutinized",
        "encumbrance_certificate",
        "flow_of_title",
        "mutation_and_tax",
        "conclusion_and_remarks",
        "layout_and_flowchart",
    ]
    
    for section in required_sections:
        if section not in review["sections"]:
            raise ValueError(f"Missing section: {section}")
        if not isinstance(review["sections"][section], list):
            raise ValueError(f"Section {section} must be an array")
    
    return True


def validate_issue(issue: Dict, section: str) -> bool:
    """
    Validate a single issue object.
    Returns True if valid, raises ValueError if not.
    """
    required_keys = ["id", "severity", "location", "rule", "message_for_maker", "suggested_fix", "evidence"]
    for key in required_keys:
        if key not in issue:
            raise ValueError(f"Issue missing required key: {key}")
    
    valid_severities = ["critical", "major", "minor"]
    if issue["severity"] not in valid_severities:
        raise ValueError(f"Invalid severity: {issue['severity']}")
    
    evidence = issue.get("evidence", {})
    if not evidence.get("from_report"):
        raise ValueError(f"Issue {issue.get('id')} missing evidence.from_report")
    if not evidence.get("from_source_docs"):
        raise ValueError(f"Issue {issue.get('id')} missing evidence.from_source_docs")
    
    return True
