from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .knowledge_base import SearchResult

ORDER_ID_RE = re.compile(r"\bORD-\d{4,}\b", re.IGNORECASE)

INTENT_KEYWORDS = {
    "return_policy": ("return", "refund", "exchange"),
    "shipping": ("shipping", "delivery", "ship", "arrive"),
    "password_reset": ("password", "reset", "recovery", "locked out"),
    "warranty": ("warranty", "repair", "replacement coverage"),
    "troubleshooting": ("flicker", "flickering", "not working", "problem", "issue", "broken"),
    "billing": ("charge", "charged", "billing", "invoice", "payment"),
}

SIMPLE_INTENTS = {"return_policy", "shipping", "password_reset", "warranty", "order_status"}
COMPLEX_TERMS = {
    "tried",
    "still",
    "again",
    "multiple",
    "several",
    "frustrated",
    "urgent",
    "manager",
    "complaint",
    "broken",
    "flickering",
    "won't",
    "wont",
}
HUMAN_TERMS = {"human", "agent", "manager", "person", "representative"}
FRUSTRATION_TERMS = {"angry", "upset", "frustrated", "terrible", "again", "complaint"}


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    route: str
    reason: str
    confidence: float
    needs_order_lookup: bool
    should_create_ticket: bool
    should_escalate: bool


def extract_order_id(message: str) -> str | None:
    match = ORDER_ID_RE.search(message)
    if not match:
        return None
    return match.group(0).upper()


def detect_intent(message: str) -> str:
    lower = message.lower()
    if extract_order_id(message):
        return "order_status"

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return intent
    return "general_support"


def confidence_from_results(results: Sequence[SearchResult], simple_intent: bool) -> float:
    top_score = results[0].score if results else 0.0
    confidence = 0.2 + min(top_score / 12.0, 0.6)
    if simple_intent and results:
        confidence += 0.1
    return max(0.1, min(confidence, 0.95))


def decide_route(
    message: str, results: Sequence[SearchResult], requested_mode: str
) -> RouteDecision:
    intent = detect_intent(message)
    lower = message.lower()
    simple_intent = intent in SIMPLE_INTENTS
    complex_query = len(message) > 150 or any(term in lower for term in COMPLEX_TERMS)
    asks_human = any(term in lower for term in HUMAN_TERMS)
    frustration = any(term in lower for term in FRUSTRATION_TERMS)
    low_evidence = not results or results[0].score < 2.5
    confidence = confidence_from_results(results, simple_intent)

    if requested_mode == "fast":
        route = "fast_private"
        reason = "User selected Fast and Private mode."
    elif requested_mode == "full":
        route = "full_power"
        reason = "User selected Full Power mode."
    elif simple_intent and not complex_query and not low_evidence:
        route = "fast_private"
        reason = "Simple intent with relevant knowledge retrieved."
    else:
        route = "full_power"
        reason = "Complex message or weak evidence requires deeper handling."

    should_escalate = asks_human or (route == "full_power" and (low_evidence or frustration))
    should_create_ticket = frustration or should_escalate

    return RouteDecision(
        intent=intent,
        route=route,
        reason=reason,
        confidence=confidence,
        needs_order_lookup=intent == "order_status",
        should_create_ticket=should_create_ticket,
        should_escalate=should_escalate,
    )
