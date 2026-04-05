from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from typing import Annotated


def _ensure_langgraph_agent_compatibility() -> None:
    runtime = importlib.import_module("langgraph.runtime")

    if not hasattr(runtime, "ServerInfo"):
        class ServerInfo:
            pass

        runtime.ServerInfo = ServerInfo

    if not hasattr(runtime.Runtime, "server_info"):
        runtime.Runtime.server_info = None


_ensure_langgraph_agent_compatibility()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from .config import Settings
from .knowledge_base import format_sources, retrieve
from .router import RouteDecision
from .tools import create_ticket, lookup_order_status

SYSTEM_PROMPT = """
You are TechGear Store's full-support agent in a privacy-first customer support simulator.

Rules:
- For policy, warranty, shipping, return, refund, troubleshooting, or product questions, call retriever_tool before your final answer.
- If the user includes an order number or asks about an order, call order_lookup_tool before answering.
- If the user asks for a human, mentions repeated failures, billing disputes, or the evidence is weak, call escalate_to_human_tool.
- If follow-up is needed, call create_ticket_tool before your final answer.
- Ground every claim in tool output. Do not invent policies, order status, or troubleshooting steps.
- Keep answers concise, empathetic, and privacy-aware.
""".strip()


@dataclass(frozen=True)
class AgentRunResult:
    answer: str
    actions: list[str]
    sources: list[str]
    ticket_id: str | None
    escalated: bool


class SupportLangChainAgent:
    def __init__(self, settings: Settings, knowledge_chunks) -> None:
        self.settings = settings
        self.knowledge_chunks = knowledge_chunks
        self._current_message = ""
        self._run_actions: list[str] = []
        self._run_sources: list[str] = []
        self._ticket_id: str | None = None
        self._escalated = False

        model = ChatOllama(
            model=self.settings.full_model,
            temperature=0,
            base_url=self.settings.ollama_url,
        )
        self.agent = create_agent(
            model=model,
            tools=self._build_tools(),
            system_prompt=SYSTEM_PROMPT,
        )

    def invoke(
        self,
        customer_message: str,
        history: list[dict[str, str]],
        decision: RouteDecision,
    ) -> AgentRunResult:
        self._reset_run_state(customer_message)
        messages = self._build_messages(customer_message, history, decision)
        result = self.agent.invoke({"messages": messages})
        answer = self._extract_final_answer(result.get("messages", []))
        if not answer:
            answer = "I do not have enough grounded information to answer confidently."

        return AgentRunResult(
            answer=answer,
            actions=self._run_actions.copy(),
            sources=self._run_sources.copy(),
            ticket_id=self._ticket_id,
            escalated=self._escalated,
        )

    def _build_messages(
        self,
        customer_message: str,
        history: list[dict[str, str]],
        decision: RouteDecision,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in history[-self.settings.max_history_messages :]:
            role = "assistant" if turn["role"] == "assistant" else "user"
            messages.append({"role": role, "content": turn["content"]})

        guidance = (
            "Routing hints for this turn:\n"
            f"- detected_intent: {decision.intent}\n"
            f"- route_reason: {decision.reason}\n"
            f"- router_confidence: {decision.confidence:.2f}\n"
            f"- router_suggests_escalation: {'yes' if decision.should_escalate else 'no'}\n"
            f"- router_suggests_ticket: {'yes' if decision.should_create_ticket else 'no'}\n\n"
            "Customer message:\n"
            f"{customer_message}"
        )
        messages.append({"role": "user", "content": guidance})
        return messages

    def _build_tools(self):
        @tool
        def retriever_tool(
            query: Annotated[
                str,
                "Full customer question or search query to run against private FAQs, manuals, and past tickets.",
            ]
        ) -> str:
            """Search the private company knowledge base for grounded support information."""
            results = retrieve(query, self.knowledge_chunks, k=self.settings.top_k)
            sources = format_sources(results)
            for source in sources:
                if source not in self._run_sources:
                    self._run_sources.append(source)

            self._append_action_once("Queried the private knowledge base.")
            payload = {
                "matches": [
                    {
                        "source": result.chunk.source,
                        "title": result.chunk.title,
                        "score": result.score,
                        "snippet": result.chunk.text,
                    }
                    for result in results
                ]
            }
            return json.dumps(payload, indent=2)

        @tool
        def order_lookup_tool(
            order_id: Annotated[
                str,
                "Exact order ID from the customer message, for example ORD-78432.",
            ]
        ) -> str:
            """Look up a simulated order by its order ID."""
            normalized_order_id = order_id.strip().upper()
            order_info = lookup_order_status(
                self.settings.data_dir / "orders.json", normalized_order_id
            )
            if order_info:
                self._append_action_once(
                    f"Checked simulated order database for {normalized_order_id}."
                )
                payload = {
                    "found": True,
                    "order_id": order_info["order_id"],
                    "status": order_info.get("status"),
                    "last_update": order_info.get("last_update"),
                    "eta_days": order_info.get("eta_days"),
                    "items": order_info.get("items", []),
                }
                return json.dumps(payload, indent=2)

            self._append_action_once(
                f"Order lookup did not find {normalized_order_id}."
            )
            return json.dumps(
                {
                    "found": False,
                    "order_id": normalized_order_id,
                    "message": "Order not found in the simulator database.",
                },
                indent=2,
            )

        @tool
        def create_ticket_tool(
            reason: Annotated[
                str,
                "Short reason for manual follow-up or case tracking.",
            ]
        ) -> str:
            """Create a support ticket for follow-up."""
            if self._ticket_id:
                return json.dumps(
                    {
                        "status": "already_created",
                        "ticket_id": self._ticket_id,
                        "reason": reason,
                    },
                    indent=2,
                )

            ticket = create_ticket(
                self.settings.logs_dir,
                customer_message=self._current_message,
                intent="agent_follow_up",
                reason=reason,
                response_excerpt=reason,
            )
            self._ticket_id = ticket["ticket_id"]
            self._append_action_once(f"Created support ticket {self._ticket_id}.")
            return json.dumps(ticket, indent=2)

        @tool
        def escalate_to_human_tool(
            reason: Annotated[
                str,
                "Short reason why a human specialist should review this case.",
            ]
        ) -> str:
            """Flag the conversation for human review."""
            self._escalated = True
            self._append_action_once("Flagged conversation for human review.")
            return json.dumps(
                {
                    "status": "escalated",
                    "reason": reason,
                },
                indent=2,
            )

        return [
            retriever_tool,
            order_lookup_tool,
            create_ticket_tool,
            escalate_to_human_tool,
        ]

    def _reset_run_state(self, customer_message: str) -> None:
        self._current_message = customer_message
        self._run_actions = []
        self._run_sources = []
        self._ticket_id = None
        self._escalated = False

    def _append_action_once(self, action: str) -> None:
        if action not in self._run_actions:
            self._run_actions.append(action)

    def _extract_final_answer(self, messages) -> str:
        for message in reversed(messages):
            if getattr(message, "type", "") != "ai":
                continue
            text = self._content_to_text(getattr(message, "content", ""))
            if text:
                return text
        return ""

    def _content_to_text(self, content) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text"):
                        parts.append(str(block["text"]))
                    elif block.get("text"):
                        parts.append(str(block["text"]))
            return "\n".join(part.strip() for part in parts if part).strip()

        return str(content).strip()