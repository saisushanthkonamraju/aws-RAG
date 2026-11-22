"""Wire retrieval, the grounded prompt and Gemini together."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import src.config as config
from src.generation.prompts import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    format_context,
    strip_preamble,
)
from src.retrieval.retriever import RetrievedChunk, search

load_dotenv(config.PROJECT_ROOT / ".env")


@dataclass
class Answer:
    """Grounded answer plus the chunks it was built from."""

    text: str
    sources: list[RetrievedChunk]


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini client, failing early with a clear message."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
        seed=config.LLM_SEED,
    )


@lru_cache(maxsize=1)
def get_chain():
    """Prompt -> Gemini -> plain text."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_TEMPLATE)]
    )
    return prompt | get_llm() | StrOutputParser()


_RETRYABLE = (
    "429",
    "rate limit",
    "quota",
    "resource exhausted",
    "unavailable",
    "503",
    "connection",
    "reset",
    "timeout",
    "timed out",
    "10054",
    "temporary failure",
    "deadline",
    "504",
)


def _invoke(chain, payload: dict, attempts: int = 2):
    """Call the chain, backing off on free-tier quota and transient errors."""
    for attempt in range(1, attempts + 1):
        try:
            return chain.invoke(payload)
        except Exception as error:
            retryable = any(marker in str(error).lower() for marker in _RETRYABLE)
            if attempt == attempts or not retryable:
                raise
            time.sleep(2**attempt)


def answer_question(question: str, k: int | None = None) -> Answer:
    """Retrieve context, then ask Gemini to answer strictly from it."""
    chunks = search(question, k=k)
    if not chunks:
        return Answer("No relevant sections were found in the whitepaper.", [])
    text = _invoke(
        get_chain(), {"context": format_context(chunks), "question": question}
    )
    return Answer(text=strip_preamble(text), sources=chunks)
