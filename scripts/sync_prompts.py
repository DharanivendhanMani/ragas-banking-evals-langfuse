"""Upload the local prompt variants to Langfuse Prompt Management (one label per variant).

Re-running creates a new prompt version only when the text changed.
"""
from src.prompts import PROMPT_NAME, PROMPTS
from src.tracing import init_langfuse

if __name__ == "__main__":
    lf = init_langfuse()
    for label, text in PROMPTS.items():
        try:
            existing = lf.get_prompt(PROMPT_NAME, label=label, type="text", cache_ttl_seconds=0)
            if existing.prompt == text and not existing.is_fallback:
                print(f"{label}: up to date (version {existing.version})")
                continue
        except Exception:
            pass
        created = lf.create_prompt(name=PROMPT_NAME, type="text", prompt=text, labels=[label],
                                   commit_message=f"sync {label} from src/prompts.py")
        print(f"{label}: created version {created.version}")
    lf.flush()
