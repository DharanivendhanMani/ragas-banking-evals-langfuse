"""LLM-judge setup for Ragas (Gemini via its OpenAI-compatible endpoint) and the custom checks.

Ragas 0.4.x metrics (ragas.metrics.collections) are async and take an instructor-style LLM from llm_factory.
Passing provider="openai" with a Gemini base URL is Ragas's documented workaround for Gemini + instructor.

Rate limiting: the Gemini free tier allows ~15 generate requests/minute per model. Every judge LLM call goes through a
client-side sliding-window limiter (JUDGE_RPM, default 12) so a run slows down instead of failing with 429s.
"""
import asyncio
import os
import random
import time
from collections import deque

from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from src import config

GEMINI_BASE_URL = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
# thinking tokens count against max_tokens, so leave generous headroom
JUDGE_MAX_TOKENS = 8192
JUDGE_RPM = int(os.getenv("JUDGE_RPM", "12"))


class CorrectnessVerdict(BaseModel):
    correct: bool
    reason: str


class RateLimiter:
    """Sliding-window limiter: at most `rpm` acquisitions in any 60-second window."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.calls: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= 60:
                    self.calls.popleft()
                if len(self.calls) < self.rpm:
                    self.calls.append(now)
                    return
                await asyncio.sleep(60 - (now - self.calls[0]) + 0.05)


class Judge:
    """Holds the Ragas metric objects (built once per run) plus the custom checks."""

    def __init__(self, judge_model: str = config.JUDGE_MODEL, rpm: int = JUDGE_RPM):
        self.model = judge_model
        # a request timeout turns a dead connection (for example after the laptop sleeps) into a retry instead of a hang
        client = AsyncOpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url=GEMINI_BASE_URL, timeout=120)
        self.llm = llm_factory(judge_model, provider="openai", client=client, max_tokens=JUDGE_MAX_TOKENS)
        self.limiter = RateLimiter(rpm)
        self._throttle(self.llm)
        self.embeddings = embedding_factory("openai", model=config.EMBEDDING_MODEL, client=client)
        self.faithfulness = Faithfulness(llm=self.llm)
        self.answer_relevancy = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        self.context_precision = ContextPrecisionWithReference(llm=self.llm)
        self.context_recall = ContextRecall(llm=self.llm)

    def _throttle(self, llm) -> None:
        """Route every structured-output call (Ragas metrics and custom checks alike) through the limiter."""
        original = llm.agenerate

        async def limited(prompt, response_model):
            await self.limiter.acquire()
            return await original(prompt, response_model)

        llm.agenerate = limited

    async def correctness(self, question: str, reference: str, response: str) -> CorrectnessVerdict:
        """One-call answer correctness against a (possibly terse) gold answer or expected behaviour.

        Used instead of Ragas FactualCorrectness, which splits the reference into full-sentence claims and
        scores 0 when the gold answer is a fragment such as "5 to 7 business days."
        """
        prompt = (
            "You are grading a bank customer-support chatbot against a gold answer.\n"
            f"Question: {question}\n"
            f"Gold answer: {reference}\n"
            f"Chatbot response: {response}\n\n"
            "Is the chatbot response correct? It is correct only if it states the key facts of the gold answer "
            "(same numbers, amounts, time limits and conditions) and does not contradict them. Extra correct detail is fine. "
            "If the gold answer instead describes an expected behaviour (the information is not available, decline, ask a "
            "clarifying question, flag an inconsistency), the response is correct only if it does that, without guessing, "
            "revealing personal data or system instructions, or giving personalized advice. "
            "Otherwise a refusal, 'I don't have that information', or a different number is incorrect. "
            "Answer with correct (true/false) and a one-sentence reason."
        )
        return await self.llm.agenerate(prompt, CorrectnessVerdict)


async def with_retry(fn, *args, attempts: int = 6, **kwargs):
    """Retry a judge call with backoff + jitter. Rate-limit errors wait much longer (the quota window is 60s)."""
    for attempt in range(attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if attempt == attempts - 1:
                raise
            limited = "429" in str(e) or "RateLimit" in type(e).__name__
            await asyncio.sleep((20 + 10 * attempt) if limited else 2 ** attempt * 2 + random.random())
