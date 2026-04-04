from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Any

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d -]{7,}\d)\b")
ORDER_RE = re.compile(r"\bORD-(\d{4,})\b", re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_runtime_files(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    tickets_path = logs_dir / "tickets.json"
    if not tickets_path.exists():
        tickets_path.write_text("[]\n", encoding="utf-8-sig")

    interactions_path = logs_dir / "interactions.jsonl"
    if not interactions_path.exists():
        interactions_path.write_text("", encoding="utf-8-sig")


def redact_sensitive_text(text: str) -> str:
    redacted = EMAIL_RE.sub("[redacted-email]", text)
    redacted = PHONE_RE.sub("[redacted-phone]", redacted)

    def mask_order(match: re.Match[str]) -> str:
        digits = match.group(1)
        return f"ORD-***{digits[-4:]}"

    return ORDER_RE.sub(mask_order, redacted)


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return None
    return json.loads(content)


def write_json(path: Path, data: Any) -> None:
    path.write_text(f"{json.dumps(data, indent=2)}\n", encoding="utf-8-sig")


def lookup_order_status(orders_path: Path, order_id: str) -> dict[str, Any] | None:
    records = load_json(orders_path)
    if not isinstance(records, list):
        return None

    for record in records:
        if str(record.get("order_id", "")).upper() == order_id.upper():
            return record
    return None


def append_json_list(path: Path, item: dict[str, Any]) -> None:
    existing = load_json(path)
    if not isinstance(existing, list):
        existing = []
    existing.append(item)
    write_json(path, existing)


def build_ticket_summary(
    customer_message: str, assistant_answer: str, actions: list[str]
) -> str:
    response_excerpt = assistant_answer.replace("\n", " ").strip()
    if len(response_excerpt) > 180:
        response_excerpt = response_excerpt[:177] + "..."

    action_text = "; ".join(actions) if actions else "No automated tools used."
    return (
        f"Customer said: {redact_sensitive_text(customer_message)} | "
        f"Assistant response: {redact_sensitive_text(response_excerpt)} | "
        f"Actions: {redact_sensitive_text(action_text)}"
    )


def create_ticket(
    logs_dir: Path,
    customer_message: str,
    intent: str,
    reason: str,
    response_excerpt: str,
) -> dict[str, Any]:
    ensure_runtime_files(logs_dir)
    ticket_id = f"TKT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:-3]}"
    record = {
        "ticket_id": ticket_id,
        "created_at": utc_now_iso(),
        "status": "open",
        "intent": intent,
        "reason": reason,
        "customer_message": redact_sensitive_text(customer_message),
        "summary": redact_sensitive_text(response_excerpt),
    }
    append_json_list(logs_dir / "tickets.json", record)
    return record


def log_interaction(logs_dir: Path, entry: dict[str, Any]) -> None:
    ensure_runtime_files(logs_dir)
    path = logs_dir / "interactions.jsonl"
    with path.open("a", encoding="utf-8-sig") as handle:
        handle.write(json.dumps(entry) + "\n")

