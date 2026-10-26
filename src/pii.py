"""PII redaction shared by Langfuse masking (src/tracing.py) and the interaction logger (src/logger.py).

Deliberately conservative: the corpus is synthetic, but the chatbot should behave as if real customers
could paste account details into the chat. Bank-owned contact details (northbridgebank.example, the
1-800-555-01xx support numbers) are left intact because they are legitimate answer content.
"""
import re

_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\b\d{9,17}\b"), "[REDACTED_ACCOUNT]"),
    (re.compile(r"\b[\w.+-]+@(?!northbridgebank\.example)[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
]


def redact(text: str) -> str:
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact_obj(obj):
    """Recursively redact strings inside dicts / lists / tuples."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_obj(v) for v in obj)
    return obj
