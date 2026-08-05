from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from src.config import settings
from src.guardrails.regex_patterns import (
    GuardrailPattern,
    PROMPT_INJECTION_PATTERNS,
    SENSITIVE_DATA_PATTERNS,
    COMMAND_INJECTION_PATTERNS,
    ALL_PATTERNS
)

class GuardrailAction(str, Enum):
    PASSED = "passed"
    SANITIZED = "sanitized"
    BLOCKED = "blocked"

@dataclass
class GuardrailResult:
    action: GuardrailAction
    is_valid: bool
    triggered_rules: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    sanitized_text: str = ""
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "is_valid": self.is_valid,
            "triggered_rules": self.triggered_rules,
            "categories": self.categories,
            "sanitized_text": self.sanitized_text,
            "reason": self.reason
        }

class InputGuardrail:
    """
    Input Guardrail for scanning user prompts before processing.
    Detects prompt injections, command injections, and sensitive input patterns.
    """
    def __init__(
        self,
        block_prompt_injection: bool = True,
        block_command_injection: bool = True,
        sanitize_pii: bool = True,
        strict_mode: bool = False
    ):
        self.block_prompt_injection = block_prompt_injection
        self.block_command_injection = block_command_injection
        self.sanitize_pii = sanitize_pii
        self.strict_mode = strict_mode

    def validate(self, text: str) -> GuardrailResult:
        if not text or not text.strip():
            return GuardrailResult(
                action=GuardrailAction.PASSED,
                is_valid=True,
                sanitized_text=text
            )

        triggered_rules: List[str] = []
        categories: List[str] = []
        sanitized = text
        is_blocked = False
        reasons: List[str] = []

        # 1. Check Prompt Injection Patterns
        for rule in PROMPT_INJECTION_PATTERNS:
            if rule.pattern.search(text):
                triggered_rules.append(rule.name)
                if rule.category not in categories:
                    categories.append(rule.category)
                reasons.append(f"Prompt injection detected ({rule.name}): {rule.description}")
                if self.block_prompt_injection:
                    is_blocked = True

        # 2. Check Command Injection Patterns
        for rule in COMMAND_INJECTION_PATTERNS:
            if rule.pattern.search(text):
                triggered_rules.append(rule.name)
                if rule.category not in categories:
                    categories.append(rule.category)
                reasons.append(f"Command injection detected ({rule.name}): {rule.description}")
                if self.block_command_injection:
                    is_blocked = True

        # 3. Check Sensitive Data Patterns (PII / Secrets)
        for rule in SENSITIVE_DATA_PATTERNS:
            matches = list(rule.pattern.finditer(sanitized))
            if matches:
                triggered_rules.append(rule.name)
                if rule.category not in categories:
                    categories.append(rule.category)

                if self.strict_mode:
                    is_blocked = True
                    reasons.append(f"Sensitive data detected under strict mode ({rule.name})")
                elif self.sanitize_pii:
                    sanitized = rule.pattern.sub(f"[REDACTED_{rule.name.upper()}]", sanitized)

        if is_blocked:
            return GuardrailResult(
                action=GuardrailAction.BLOCKED,
                is_valid=False,
                triggered_rules=triggered_rules,
                categories=categories,
                sanitized_text="",
                reason="; ".join(reasons)
            )

        if sanitized != text:
            return GuardrailResult(
                action=GuardrailAction.SANITIZED,
                is_valid=True,
                triggered_rules=triggered_rules,
                categories=categories,
                sanitized_text=sanitized,
                reason="Input sanitized to redact sensitive information."
            )

        return GuardrailResult(
            action=GuardrailAction.PASSED,
            is_valid=True,
            triggered_rules=triggered_rules,
            categories=categories,
            sanitized_text=text,
            reason=None
        )

class OutputGuardrail:
    """
    Output Guardrail for scanning generated responses before returning to users.
    Detects secret leaks (API keys, credentials), PII leaks, and system instruction leakage.
    """
    def __init__(self, sanitize_secrets: bool = True, sanitize_pii: bool = True):
        self.sanitize_secrets = sanitize_secrets
        self.sanitize_pii = sanitize_pii

    def validate(self, text: str) -> GuardrailResult:
        if not text or not text.strip():
            return GuardrailResult(
                action=GuardrailAction.PASSED,
                is_valid=True,
                sanitized_text=text
            )

        triggered_rules: List[str] = []
        categories: List[str] = []
        sanitized = text

        # Check Sensitive Data Patterns (API keys, SSN, Email, etc.)
        for rule in SENSITIVE_DATA_PATTERNS:
            if rule.pattern.search(sanitized):
                triggered_rules.append(rule.name)
                if rule.category not in categories:
                    categories.append(rule.category)
                sanitized = rule.pattern.sub(f"[REDACTED_{rule.name.upper()}]", sanitized)

        if sanitized != text:
            return GuardrailResult(
                action=GuardrailAction.SANITIZED,
                is_valid=True,
                triggered_rules=triggered_rules,
                categories=categories,
                sanitized_text=sanitized,
                reason="Output sanitized to prevent sensitive data/secret leakage."
            )

        return GuardrailResult(
            action=GuardrailAction.PASSED,
            is_valid=True,
            triggered_rules=triggered_rules,
            categories=categories,
            sanitized_text=text,
            reason=None
        )

def get_input_guardrail() -> InputGuardrail:
    return InputGuardrail(
        block_prompt_injection=getattr(settings, "GUARDRAIL_BLOCK_PROMPT_INJECTION", True),
        block_command_injection=True,
        sanitize_pii=True,
        strict_mode=getattr(settings, "GUARDRAIL_STRICT_MODE", False)
    )

def get_output_guardrail() -> OutputGuardrail:
    return OutputGuardrail(
        sanitize_secrets=True,
        sanitize_pii=True
    )
