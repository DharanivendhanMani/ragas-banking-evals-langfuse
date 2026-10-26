"""Langfuse client setup (masking + release) and shared helpers.

Import order matters: src.config loads .env first, so Langfuse initialises with real credentials.
"""
from typing import Optional

from langfuse import Langfuse, get_client
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch

from src import config
from src.pii import redact

_initialised = False


def _mask_otel_spans(*, params: MaskOtelSpansParams) -> Optional[MaskOtelSpansResult]:
    """Redact PII in every string attribute (inputs, outputs, metadata) before export."""
    patches = {}
    for identifier, span in params.spans.items():
        changed = {k: redact(v) for k, v in span.attributes.items() if isinstance(v, str) and redact(v) != v}
        if changed:
            patches[identifier] = OtelSpanPatch(set_attributes=changed)
    return MaskOtelSpansResult(span_patches=patches) if patches else None


def init_langfuse() -> Langfuse:
    """Create (once) and return the global Langfuse client with PII masking enabled."""
    global _initialised
    if not _initialised:
        Langfuse(mask_otel_spans=_mask_otel_spans, release=config.PROMPT_VERSION)
        _initialised = True
    return get_client()
