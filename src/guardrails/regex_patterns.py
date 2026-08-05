import re
from typing import Dict, List, NamedTuple

class GuardrailPattern(NamedTuple):
    name: str
    category: str
    pattern: re.Pattern
    description: str

# Prompt Injection & System Instruction Overrides
PROMPT_INJECTION_PATTERNS: List[GuardrailPattern] = [
    GuardrailPattern(
        name="ignore_previous_instructions",
        category="prompt_injection",
        pattern=re.compile(
            r"(?i)\bignore\s+(?:all\s+)?(?:previous|above|prior|former|past)\s+(?:instructions|directions|rules|prompts|commands|constraints)\b"
        ),
        description="Attempts to instruct the LLM to ignore prior system prompts."
    ),
    GuardrailPattern(
        name="disregard_system_prompt",
        category="prompt_injection",
        pattern=re.compile(
            r"(?i)\b(?:disregard|forget|override|bypass|cancel|reset)\s+(?:the\s+)?(?:system|initial|original|above|previous)\s+(?:prompt|instructions|rules|context)\b"
        ),
        description="Attempts to disregard or reset system prompt constraints."
    ),
    GuardrailPattern(
        name="role_impersonation_tags",
        category="prompt_injection",
        pattern=re.compile(
            r"(?i)(?:<\|im_start\|>|<\|im_end\|>|\[system\]|\[inst\]|<<sys>>|<s>\[inst\]|<\|system\|>)"
        ),
        description="Delimiters injected to spoof special model control tokens."
    ),
    GuardrailPattern(
        name="jailbreak_personas",
        category="prompt_injection",
        pattern=re.compile(
            r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|mode:)\s+(?:dan|unrestricted|god\s+mode|developer\s+mode|chaos\s+gpt|do\s+anything\s+now|jailbroken)\b"
        ),
        description="Common jailbreak persona switches (e.g. DAN, Developer Mode)."
    ),
    GuardrailPattern(
        name="do_anything_now",
        category="prompt_injection",
        pattern=re.compile(
            r"(?i)\b(?:do\s+anything\s+now|stay\s+in\s+developer\s+mode|ignore\s+safety\s+guidelines|bypass\s+restrictions)\b"
        ),
        description="Explicit requests to bypass safety guidelines or restrictions."
    ),
    GuardrailPattern(
        name="secret_extraction",
        category="prompt_injection",
        pattern=re.compile(
            r"(?i)\b(?:reveal|show|print|display|tell\s+me|output|repeat)\s+(?:your\s+)?(?:system\s+prompt|instructions|api\s+key|hidden\s+rules|developer\s+prompt|secret\s+key|confidential\s+token)\b"
        ),
        description="Attempts to extract internal system prompts or API secrets."
    ),
    GuardrailPattern(
        name="delimiter_markdown_injection",
        category="prompt_injection",
        pattern=re.compile(
            r"(?i)(?:```(?:system|prompt|admin)|#+\s*system\s*prompt|\[OVERRIDE\]|\[SYSTEM\s+MESSAGE\])"
        ),
        description="Markdown structure injection attempting to fake system headers."
    ),
]

# PII and Sensitive Data Leakage Patterns
SENSITIVE_DATA_PATTERNS: List[GuardrailPattern] = [
    GuardrailPattern(
        name="email_address",
        category="pii_leakage",
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        description="Detects email addresses."
    ),
    GuardrailPattern(
        name="social_security_number",
        category="pii_leakage",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        description="Detects US Social Security Numbers (SSN)."
    ),
    GuardrailPattern(
        name="credit_card_number",
        category="pii_leakage",
        pattern=re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        description="Detects credit card numbers."
    ),
    GuardrailPattern(
        name="api_secret_key",
        category="secret_leakage",
        pattern=re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|AIzaSy[a-zA-Z0-9_-]{33}|ghp_[a-zA-Z0-9]{36}|Bearer\s+[a-zA-Z0-9_\-\.=]+)\b"),
        description="Detects API keys and secret bearer tokens."
    ),
    GuardrailPattern(
        name="phone_number",
        category="pii_leakage",
        pattern=re.compile(r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        description="Detects standard phone numbers."
    ),
]

# Malicious Command & SQL Injection Patterns
COMMAND_INJECTION_PATTERNS: List[GuardrailPattern] = [
    GuardrailPattern(
        name="command_injection",
        category="command_injection",
        pattern=re.compile(r"(?i)(?:;\s*(?:rm\s+-rf|drop\s+table|exec\s*\(|system\s*\()|__import__\s*\(|passthru\s*\()"),
        description="Detects code execution or shell command injection attempts."
    )
]

ALL_PATTERNS: List[GuardrailPattern] = (
    PROMPT_INJECTION_PATTERNS + SENSITIVE_DATA_PATTERNS + COMMAND_INJECTION_PATTERNS
)
