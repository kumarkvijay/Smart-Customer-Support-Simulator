from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

PRIMARY_MODES = ("raw_llm", "raw_slm", "agent", "rag")
LEGACY_MODES = ("auto", "fast", "full")
SUPPORTED_MODES = PRIMARY_MODES + LEGACY_MODES
MODE_ALIASES = {
    "llm": "raw_llm",
    "slm": "raw_slm",
    "retrieval": "rag",
    "raw_sllm": "raw_slm",
    "raw_llm": "raw_llm",
    "raw_slm": "raw_slm",
    "agent": "agent",
    "rag": "rag",
    "fast_private": "raw_slm",
    "fast_and_private": "raw_slm",
    "fast_private_mode": "raw_slm",
    "fast_and_private_mode": "raw_slm",
    "slm_only": "raw_slm",
    "full_intelligence": "agent",
    "full_intelligence_mode": "agent",
}


def normalize_mode_token(mode: str | None) -> str:
    candidate = (mode or "").strip().lower()
    for source, target in (("&", " and "), ("-", "_"), (" ", "_"), ("/", "_")):
        candidate = candidate.replace(source, target)
    while "__" in candidate:
        candidate = candidate.replace("__", "_")
    return candidate.strip("_")


@dataclass(frozen=True)
class Settings:
    project_root: Path
    knowledge_base_dir: Path
    data_dir: Path
    logs_dir: Path
    default_mode: str
    enable_ollama: bool
    enable_langchain_agent: bool
    fast_model: str
    full_model: str
    ollama_url: str
    top_k: int
    max_history_messages: int


def is_supported_mode(mode: str | None) -> bool:
    candidate = normalize_mode_token(mode)
    if not candidate:
        return False
    return candidate in SUPPORTED_MODES or candidate in MODE_ALIASES


def normalize_mode(mode: str | None, default: str = "agent") -> str:
    candidate = normalize_mode_token(mode or default)
    candidate = MODE_ALIASES.get(candidate, candidate)
    if candidate not in SUPPORTED_MODES:
        return default if default in SUPPORTED_MODES else "agent"
    return candidate


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    default_mode = normalize_mode(os.getenv("SIMULATOR_MODE", "agent"), default="agent")

    return Settings(
        project_root=project_root,
        knowledge_base_dir=project_root / "knowledge_base",
        data_dir=project_root / "data",
        logs_dir=project_root / "logs",
        default_mode=default_mode,
        enable_ollama=os.getenv("ENABLE_OLLAMA", "1").strip() != "0",
        enable_langchain_agent=os.getenv("ENABLE_LANGCHAIN_AGENT", "1").strip() != "0",
        fast_model=os.getenv("FAST_MODEL", "phi3:3.8b").strip(),
        full_model=os.getenv("FULL_MODEL", "llama3.1:8b").strip(),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").strip(),
        top_k=max(1, int(os.getenv("TOP_K", "3"))),
        max_history_messages=max(2, int(os.getenv("MAX_HISTORY_MESSAGES", "8"))),
    )
