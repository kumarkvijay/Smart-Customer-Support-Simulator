from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .knowledge_base import KnowledgeChunk, SearchResult, build_context, format_sources, retrieve
from .providers import chat_with_ollama
from .router import RouteDecision, decide_route, extract_order_id
from .tools import build_ticket_summary, create_ticket, lookup_order_status


@dataclass(frozen=True)
class WorkflowTurnResult:
    answer: str
    route: str
    intent: str
    confidence: float
    sources: list[str]
    actions: list[str]
    ticket_id: str | None
    conversation_history: list[dict[str, str]]


class WorkflowState(TypedDict, total=False):
    message: str
    selected_mode: str
    results: list[SearchResult]
    decision: RouteDecision
    order_info: dict[str, Any] | None
    answer: str
    actions: list[str]
    sources: list[str]
    ticket_id: str | None
    escalated: bool
    agent_succeeded: bool
    conversation_history: list[dict[str, str]]


class SupportConversationWorkflow:
    def __init__(
        self,
        settings: Settings,
        knowledge_chunks: list[KnowledgeChunk],
        agent_runtime: Any | None,
    ) -> None:
        self.settings = settings
        self.knowledge_chunks = knowledge_chunks
        self.agent_runtime = agent_runtime
        self.thread_id = f"support-session-{uuid4().hex}"
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=[KnowledgeChunk, SearchResult, RouteDecision]
        )
        self.checkpointer = InMemorySaver(serde=serializer)
        self.graph = self._build_graph()

    def handle_message(self, message: str, mode: str) -> WorkflowTurnResult:
        state = self.graph.invoke(
            {
                "message": message,
                "selected_mode": mode,
            },
            config={"configurable": {"thread_id": self.thread_id}},
        )
        decision = state["decision"]
        return WorkflowTurnResult(
            answer=state["answer"],
            route=decision.route,
            intent=decision.intent,
            confidence=decision.confidence,
            sources=state.get("sources", []),
            actions=state.get("actions", []),
            ticket_id=state.get("ticket_id"),
            conversation_history=state.get("conversation_history", []),
        )

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("prepare_turn", self._prepare_turn)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("route", self._route_node)
        builder.add_node("full_agent", self._full_agent_node)
        builder.add_node("order_lookup", self._order_lookup_node)
        builder.add_node("generate_answer", self._generate_answer_node)
        builder.add_node("create_ticket", self._create_ticket_node)
        builder.add_node("escalate_case", self._escalate_case_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "prepare_turn")
        builder.add_edge("prepare_turn", "retrieve")
        builder.add_edge("retrieve", "route")
        builder.add_conditional_edges(
            "route",
            self._route_after_decision,
            {
                "full_agent": "full_agent",
                "order_lookup": "order_lookup",
                "generate_answer": "generate_answer",
            },
        )
        builder.add_conditional_edges(
            "full_agent",
            self._route_after_full_agent,
            {
                "order_lookup": "order_lookup",
                "generate_answer": "generate_answer",
                "create_ticket": "create_ticket",
                "escalate_case": "escalate_case",
                "finalize": "finalize",
            },
        )
        builder.add_edge("order_lookup", "generate_answer")
        builder.add_conditional_edges(
            "generate_answer",
            self._route_after_answer,
            {
                "create_ticket": "create_ticket",
                "escalate_case": "escalate_case",
                "finalize": "finalize",
            },
        )
        builder.add_conditional_edges(
            "create_ticket",
            self._route_after_ticket,
            {
                "escalate_case": "escalate_case",
                "finalize": "finalize",
            },
        )
        builder.add_edge("escalate_case", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _prepare_turn(self, state: WorkflowState) -> WorkflowState:
        selected_mode = (state.get("selected_mode") or self.settings.default_mode).strip().lower()
        if selected_mode not in {"auto", "fast", "full"}:
            selected_mode = self.settings.default_mode

        history = state.get("conversation_history", [])
        return {
            "selected_mode": selected_mode,
            "results": [],
            "order_info": None,
            "answer": "",
            "actions": [],
            "sources": [],
            "ticket_id": None,
            "escalated": False,
            "agent_succeeded": False,
            "conversation_history": history[-self.settings.max_history_messages :],
        }

    def _retrieve_node(self, state: WorkflowState) -> WorkflowState:
        results = retrieve(state["message"], self.knowledge_chunks, k=self.settings.top_k)
        return {"results": results}

    def _route_node(self, state: WorkflowState) -> WorkflowState:
        decision = decide_route(
            state["message"],
            state.get("results", []),
            state["selected_mode"],
        )
        return {"decision": decision}

    def _route_after_decision(self, state: WorkflowState) -> str:
        decision = state["decision"]
        if decision.route == "full_power" and self.agent_runtime is not None:
            return "full_agent"
        if decision.needs_order_lookup:
            return "order_lookup"
        return "generate_answer"

    def _route_after_full_agent(self, state: WorkflowState) -> str:
        if not state.get("agent_succeeded"):
            if state["decision"].needs_order_lookup:
                return "order_lookup"
            return "generate_answer"
        return self._post_answer_target(state)

    def _route_after_answer(self, state: WorkflowState) -> str:
        return self._post_answer_target(state)

    def _route_after_ticket(self, state: WorkflowState) -> str:
        if self._needs_escalation(state):
            return "escalate_case"
        return "finalize"

    def _post_answer_target(self, state: WorkflowState) -> str:
        if self._needs_ticket(state):
            return "create_ticket"
        if self._needs_escalation(state):
            return "escalate_case"
        return "finalize"

    def _needs_ticket(self, state: WorkflowState) -> bool:
        decision = state["decision"]
        return decision.should_create_ticket and not state.get("ticket_id")

    def _needs_escalation(self, state: WorkflowState) -> bool:
        decision = state["decision"]
        return decision.should_escalate or state.get("escalated", False)

    def _full_agent_node(self, state: WorkflowState) -> WorkflowState:
        if self.agent_runtime is None:
            return {"agent_succeeded": False}

        try:
            agent_result = self.agent_runtime.invoke(
                state["message"],
                state.get("conversation_history", []),
                state["decision"],
            )
        except Exception:
            return {"agent_succeeded": False}

        answer = agent_result.answer
        if not answer:
            answer = "I do not have enough grounded information to answer confidently."
        actions = list(agent_result.actions)
        sources = list(agent_result.sources)

        if not sources and state["decision"].intent != "order_status":
            fallback_results = retrieve(
                state["message"], self.knowledge_chunks, k=self.settings.top_k
            )
            sources = format_sources(fallback_results)
            if sources and "Queried the private knowledge base." not in actions:
                actions.insert(0, "Queried the private knowledge base.")
            answer = self._generate_template_answer(
                message=state["message"],
                decision=state["decision"],
                results=fallback_results,
                order_info=None,
            )

        return {
            "answer": answer,
            "actions": actions,
            "sources": sources,
            "ticket_id": agent_result.ticket_id,
            "escalated": agent_result.escalated,
            "agent_succeeded": True,
        }

    def _order_lookup_node(self, state: WorkflowState) -> WorkflowState:
        actions = list(state.get("actions", []))
        order_id = extract_order_id(state["message"])
        if not order_id:
            actions.append("Order lookup skipped because no order number was found.")
            return {"order_info": None, "actions": actions}

        order_info = lookup_order_status(self.settings.data_dir / "orders.json", order_id)
        if order_info:
            actions.append(f"Checked simulated order database for {order_id}.")
        else:
            actions.append(f"Order lookup did not find {order_id}.")
        return {"order_info": order_info, "actions": actions}

    def _generate_answer_node(self, state: WorkflowState) -> WorkflowState:
        answer = self._generate_answer(
            message=state["message"],
            decision=state["decision"],
            results=state.get("results", []),
            order_info=state.get("order_info"),
            conversation_history=state.get("conversation_history", []),
        )
        return {
            "answer": answer,
            "sources": format_sources(state.get("results", [])),
        }

    def _create_ticket_node(self, state: WorkflowState) -> WorkflowState:
        actions = list(state.get("actions", []))
        summary = build_ticket_summary(state["message"], state["answer"], actions)
        ticket = create_ticket(
            self.settings.logs_dir,
            customer_message=state["message"],
            intent=state["decision"].intent,
            reason=state["decision"].reason,
            response_excerpt=summary,
        )
        ticket_id = ticket["ticket_id"]
        actions.append(f"Created support ticket {ticket_id}.")
        return {
            "ticket_id": ticket_id,
            "actions": actions,
        }

    def _escalate_case_node(self, state: WorkflowState) -> WorkflowState:
        answer = state["answer"]
        follow_up = "I have marked this case for human follow-up."
        if follow_up not in answer:
            answer = f"{answer}\n\n{follow_up}".strip()
        actions = list(state.get("actions", []))
        actions.append("Flagged conversation for human review.")
        return {
            "answer": answer,
            "actions": actions,
            "escalated": True,
        }

    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        answer = state.get("answer", "").strip()
        if not answer:
            answer = "I do not have enough grounded information to answer confidently."

        history = list(state.get("conversation_history", []))
        history.extend(
            [
                {"role": "user", "content": state["message"]},
                {"role": "assistant", "content": answer},
            ]
        )
        history = history[-self.settings.max_history_messages :]

        return {
            "answer": answer,
            "actions": self._dedupe_list(state.get("actions", [])),
            "sources": self._dedupe_list(state.get("sources", [])),
            "conversation_history": history,
        }

    def _generate_answer(
        self,
        message: str,
        decision: RouteDecision,
        results: list[SearchResult],
        order_info: dict[str, Any] | None,
        conversation_history: list[dict[str, str]],
    ) -> str:
        context = build_context(results)

        if self.settings.enable_ollama and decision.route != "fast_private":
            ollama_answer = self._generate_with_ollama(
                message=message,
                decision=decision,
                context=context,
                order_info=order_info,
                conversation_history=conversation_history,
            )
            if ollama_answer:
                return ollama_answer

        return self._generate_template_answer(message, decision, results, order_info)

    def _generate_with_ollama(
        self,
        message: str,
        decision: RouteDecision,
        context: str,
        order_info: dict[str, Any] | None,
        conversation_history: list[dict[str, str]],
    ) -> str:
        system_prompt = (
            "You are a careful customer support assistant for TechGear Store. "
            "Use only the supplied context and tool output. "
            "Do not invent policies, order details, exceptions, or next steps that are not supported by the context. "
            "If the evidence is weak, say that clearly and suggest escalation. "
            "Do not add greetings, signatures, or channel instructions."
        )
        history = self._history_summary(conversation_history)
        tool_output = json.dumps(order_info, indent=2) if order_info else "No order lookup result."
        style_instruction = (
            "Keep the reply under 120 words and focus on the verified answer only."
            if decision.route == "fast_private"
            else "Keep the reply concise, empathetic, and practical."
        )
        user_prompt = (
            f"Route: {decision.route}\n"
            f"Intent: {decision.intent}\n"
            f"Reason: {decision.reason}\n"
            f"Conversation history: {history}\n\n"
            f"Customer message:\n{message}\n\n"
            f"Retrieved context:\n{context or 'No supporting context found.'}\n\n"
            f"Tool output:\n{tool_output}\n\n"
            f"{style_instruction}"
        )
        model = (
            self.settings.fast_model
            if decision.route == "fast_private"
            else self.settings.full_model
        )
        result = chat_with_ollama(
            base_url=self.settings.ollama_url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if result.ok:
            return result.content
        return ""

    def _generate_template_answer(
        self,
        message: str,
        decision: RouteDecision,
        results: list[SearchResult],
        order_info: dict[str, Any] | None,
    ) -> str:
        lower = message.lower()
        lines: list[str] = []

        if decision.intent != "order_status" and not results:
            lines.append(
                "I do not have enough verified knowledge for a confident answer from the local documents."
            )
            if decision.should_escalate and not order_info:
                lines.append(
                    "Because the evidence is limited or the case looks sensitive, a human agent should review it."
                )
            return "\n".join(lines)

        if decision.intent == "order_status":
            if order_info:
                items = ", ".join(order_info.get("items", [])) or "your item"
                lines.append(
                    f"I checked order {order_info['order_id']}. It is currently {order_info['status']}."
                )
                lines.append(
                    f"Last update: {order_info.get('last_update', 'unknown')}. Estimated arrival: {order_info.get('eta_days', 'unknown')} day(s)."
                )
                lines.append(f"Items in the order: {items}.")
            else:
                lines.append(
                    "I could not find that order number in the simulator database."
                )
        elif any(
            term in lower
            for term in ("flicker", "flickering", "not working", "problem", "issue", "broken")
        ):
            lines.append(
                "I am sorry you are dealing with that. Here is the best grounded guidance from the current support knowledge base."
            )
        else:
            lines.append(
                "Here is the best grounded answer I can provide from the current local support knowledge base."
            )

        if results:
            lines.append("Relevant guidance:")
            for result in results[:2]:
                excerpt = result.chunk.text.replace("\n", " ").strip()
                if len(excerpt) > 220:
                    excerpt = excerpt[:217] + "..."
                lines.append(f"- {result.chunk.title}: {excerpt}")
        elif decision.intent != "order_status":
            lines.append(
                "I do not have enough verified knowledge for a confident answer from the local documents."
            )

        if decision.intent == "return_policy":
            lines.append(
                "If the product is outside the listed return window, the next step is usually a manual policy review or exception request."
            )

        if decision.should_escalate and not order_info:
            lines.append(
                "Because the evidence is limited or the case looks sensitive, a human agent should review it."
            )

        return "\n".join(lines)

    def _history_summary(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "No previous messages."
        recent_turns = history[-self.settings.max_history_messages :]
        return " | ".join(f"{turn['role']}: {turn['content']}" for turn in recent_turns)

    def _dedupe_list(self, items: list[str]) -> list[str]:
        unique_items: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        return unique_items