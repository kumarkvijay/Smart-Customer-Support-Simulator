from __future__ import annotations

from dataclasses import dataclass

from .config import normalize_mode, normalize_mode_token


@dataclass(frozen=True)
class ChatProfile:
    key: str
    label: str
    simulator_mode: str
    description: str


CHAT_PROFILES = (
    ChatProfile(
        key="fast_private",
        label="Fast & Private Mode",
        simulator_mode="raw_slm",
        description="Small local model only. No retrieval, tools, or order database access.",
    ),
    ChatProfile(
        key="full_intelligence",
        label="Full Intelligence Mode",
        simulator_mode="agent",
        description="Agent mode with retrieval, order lookup, ticketing, and escalation.",
    ),
)

CHAT_PROFILE_ALIASES = {
    "fast_private_mode": "fast_private",
    "fast_and_private_mode": "fast_private",
    "fast_and_private": "fast_private",
    "slm_only": "fast_private",
    "full_intelligence_mode": "full_intelligence",
}

CHAT_PROFILE_BY_KEY = {profile.key: profile for profile in CHAT_PROFILES}
CHAT_PROFILE_BY_MODE = {profile.simulator_mode: profile for profile in CHAT_PROFILES}


def list_chat_profiles() -> tuple[ChatProfile, ...]:
    return CHAT_PROFILES


def normalize_chat_profile_key(
    value: str | None, default: str = "full_intelligence"
) -> str:
    candidate = normalize_mode_token(value)
    candidate = CHAT_PROFILE_ALIASES.get(candidate, candidate)
    if candidate in CHAT_PROFILE_BY_KEY:
        return candidate
    return default


def get_chat_profile(
    value: str | None = None, default: str = "full_intelligence"
) -> ChatProfile:
    return CHAT_PROFILE_BY_KEY[normalize_chat_profile_key(value, default)]


def get_chat_profile_for_mode(mode: str | None) -> ChatProfile | None:
    normalized_mode = normalize_mode(mode)
    return CHAT_PROFILE_BY_MODE.get(normalized_mode)
