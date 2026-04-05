from __future__ import annotations

import ast
from dataclasses import dataclass
import operator
import re
from typing import Sequence

from .knowledge_base import SearchResult

ORDER_ID_RE = re.compile(r"\bORD-\d{4,}\b", re.IGNORECASE)
CALCULATION_RE = re.compile(
    r"^\s*[-+*/().\d\s]+$|^\s*(what is|what's|calculate|compute|solve)\b",
    re.IGNORECASE,
)

INTENT_KEYWORDS = {
    "return_policy": ("return", "refund", "exchange"),
    "shipping": ("shipping", "delivery", "ship", "arrive"),
    "password_reset": ("password", "reset", "recovery", "locked out"),
    "warranty": ("warranty", "repair", "replacement coverage"),
    "troubleshooting": ("flicker", "flickering", "not working", "problem", "issue", "broken"),
    "billing": ("charge", "charged", "billing", "invoice", "payment"),
}

SIMPLE_INTENTS = {
    "return_policy",
    "shipping",
    "password_reset",
    "warranty",
    "order_status",
    "calculation",
}
SENSITIVE_INTENTS = {"troubleshooting", "billing"}
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
CALCULATION_REPLACEMENTS = (
    (r"\bmultiplied\s+by\b", "*"),
    (r"\bdivided\s+by\b", "/"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\btimes\b", "*"),
    (r"\bover\b", "/"),
)
CALCULATION_FILLER_RE = re.compile(
    r"\b(?:what|is|what's|calculate|compute|solve|equals|equal|to|by)\b",
    re.IGNORECASE,
)
ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


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


def is_simple_calculation(message: str) -> bool:
    lower = message.lower()
    if not any(char.isdigit() for char in message):
        return False
    math_terms = ("plus", "minus", "times", "multiplied", "divided", "equals", "equal")
    operator_chars = {"+", "-", "*", "/", "="}
    return bool(
        CALCULATION_RE.search(message)
        or any(term in lower for term in math_terms)
        or any(char in operator_chars for char in message)
    )


def evaluate_simple_calculation(message: str) -> str | None:
    if not is_simple_calculation(message):
        return None

    expression = message.lower().replace(",", "")
    for pattern, replacement in CALCULATION_REPLACEMENTS:
        expression = re.sub(pattern, f" {replacement} ", expression)

    expression = CALCULATION_FILLER_RE.sub(" ", expression)
    expression = expression.replace("=", " ")
    expression = re.sub(r"[^0-9+\-*/().\s]", " ", expression)
    expression = re.sub(r"\s+", " ", expression).strip()
    if not expression or not any(char.isdigit() for char in expression):
        return None

    try:
        parsed = ast.parse(expression, mode="eval")
        value = _evaluate_calculation_node(parsed.body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None

    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return format(value, ".10f").rstrip("0").rstrip(".")
    return str(value)


def _evaluate_calculation_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp):
        operation = ALLOWED_UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(_evaluate_calculation_node(node.operand))

    if isinstance(node, ast.BinOp):
        operation = ALLOWED_BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported binary operator.")
        left_value = _evaluate_calculation_node(node.left)
        right_value = _evaluate_calculation_node(node.right)
        return operation(left_value, right_value)

    raise ValueError("Unsupported calculation.")


def detect_intent(message: str) -> str:
    lower = message.lower()
    if extract_order_id(message):
        return "order_status"
    if is_simple_calculation(message):
        return "calculation"

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

    sensitive_low_evidence = intent in SENSITIVE_INTENTS and low_evidence
    should_escalate = asks_human or frustration or sensitive_low_evidence
    should_create_ticket = frustration or asks_human or sensitive_low_evidence

    return RouteDecision(
        intent=intent,
        route=route,
        reason=reason,
        confidence=confidence,
        needs_order_lookup=intent == "order_status",
        should_create_ticket=should_create_ticket,
        should_escalate=should_escalate,
    )
