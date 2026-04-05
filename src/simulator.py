from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any

from .config import Settings, get_settings, normalize_mode
from .knowledge_base import build_context, format_sources, load_knowledge_base, retrieve
from .providers import chat_with_ollama
from .router import (
    RouteDecision,
    decide_route,
    detect_intent,
    evaluate_simple_calculation,
    extract_order_id,
)
from .tools import (
    build_ticket_summary,
    create_ticket,
    ensure_runtime_files,
    log_interaction,
    lookup_order_status,
    redact_sensitive_text,
)

try:
    from .agent import SupportLangChainAgent
    AGENT_IMPORT_ERROR = None
except Exception as exc:
    SupportLangChainAgent = None
    AGENT_IMPORT_ERROR = exc

try:
    from .workflow import SupportConversationWorkflow
    WORKFLOW_IMPORT_ERROR = None
except Exception as exc:
    SupportConversationWorkflow = None
    WORKFLOW_IMPORT_ERROR = exc


@dataclass(frozen=True)
class SupportResponse:
    answer: str
    route: str
    intent: str
    confidence: float
    sources: list[str]
    actions: list[str]
    ticket_id: str | None = None


class SupportSimulator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        ensure_runtime_files(self.settings.logs_dir)
        self.knowledge_chunks = load_knowledge_base(self.settings.knowledge_base_dir)
        self.history: list[dict[str, str]] = []
        self.agent_runtime = None
        self.workflow = None
        self.agent_runtime_error: str | None = (
            f"{type(AGENT_IMPORT_ERROR).__name__}: {AGENT_IMPORT_ERROR}"
            if AGENT_IMPORT_ERROR is not None
            else None
        )
        self.workflow_error: str | None = (
            f"{type(WORKFLOW_IMPORT_ERROR).__name__}: {WORKFLOW_IMPORT_ERROR}"
            if WORKFLOW_IMPORT_ERROR is not None
            else None
        )

        if SupportLangChainAgent and self.settings.enable_langchain_agent:
            try:
                self.agent_runtime = SupportLangChainAgent(
                    self.settings, self.knowledge_chunks
                )
                self.agent_runtime_error = None
            except Exception as exc:
                self.agent_runtime = None
                self.agent_runtime_error = f"{type(exc).__name__}: {exc}"

        if SupportConversationWorkflow:
            try:
                self.workflow = SupportConversationWorkflow(
                    self.settings,
                    self.knowledge_chunks,
                    self.agent_runtime,
                )
                self.workflow_error = None
            except Exception as exc:
                self.workflow = None
                self.workflow_error = f"{type(exc).__name__}: {exc}"

    def handle_message(self, message: str, mode: str | None = None) -> SupportResponse:
        selected_mode = normalize_mode(mode, self.settings.default_mode)

        if selected_mode in {"raw_llm", "raw_slm"}:
            response = self._handle_raw_model_mode(message, selected_mode)
            self._remember(message, response.answer)
            self._log_interaction(message, response, selected_mode)
            return response

        response = self._handle_direct_calculation(message, selected_mode)
        if response is not None:
            self._remember(message, response.answer)
            self._log_interaction(message, response, selected_mode)
            return response

        if selected_mode == "rag":
            response = self._handle_rag_mode(message)
            self._remember(message, response.answer)
            self._log_interaction(message, response, selected_mode)
            return response

        if selected_mode == "agent":
            response = self._handle_agent_mode(message)
            self._remember(message, response.answer)
            self._log_interaction(message, response, selected_mode)
            return response

        if self.workflow is not None:
            try:
                workflow_result = self.workflow.handle_message(message, selected_mode)
            except Exception:
                self.workflow = None
            else:
                self.history = workflow_result.conversation_history
                response = SupportResponse(
                    answer=workflow_result.answer,
                    route=workflow_result.route,
                    intent=workflow_result.intent,
                    confidence=workflow_result.confidence,
                    sources=workflow_result.sources,
                    actions=workflow_result.actions,
                    ticket_id=workflow_result.ticket_id,
                )
                self._log_interaction(message, response, selected_mode)
                return response

        results = retrieve(message, self.knowledge_chunks, k=self.settings.top_k)
        decision = decide_route(message, results, selected_mode)
        response = self._handle_with_agent(message, decision)
        if response is None:
            response = self._handle_with_fallback(message, decision, results)

        self._remember(message, response.answer)
        self._log_interaction(message, response, selected_mode)
        return response

    def _handle_raw_model_mode(self, message: str, selected_mode: str) -> SupportResponse:
        intent = detect_intent(message)
        model = (
            self.settings.full_model
            if selected_mode == "raw_llm"
            else self.settings.fast_model
        )

        if not self.settings.enable_ollama:
            return SupportResponse(
                answer="Ollama is disabled, so raw model mode cannot run.",
                route=selected_mode,
                intent=intent,
                confidence=0.0,
                sources=[],
                actions=["Raw model mode skipped because ENABLE_OLLAMA=0."],
            )

        result = chat_with_ollama(
            base_url=self.settings.ollama_url,
            model=model,
            system_prompt=self._raw_model_system_prompt(),
            user_prompt=self._raw_model_user_prompt(message),
        )
        if not result.ok:
            return SupportResponse(
                answer=f"Raw model mode could not get a response from Ollama. {result.error}",
                route=selected_mode,
                intent=intent,
                confidence=0.0,
                sources=[],
                actions=[f"Attempted direct Ollama call to {model} without retrieval or tools."],
            )

        return SupportResponse(
            answer=result.content,
            route=selected_mode,
            intent=intent,
            confidence=1.0,
            sources=[],
            actions=[f"Queried Ollama model {model} directly without retrieval or tools."],
        )

    def _handle_rag_mode(self, message: str) -> SupportResponse:
        intent = detect_intent(message)
        if intent == "order_status":
            return SupportResponse(
                answer=(
                    "RAG mode only searches the local knowledge base and does not query the "
                    "simulated order database. Use agent mode for order lookups."
                ),
                route="rag",
                intent=intent,
                confidence=0.1,
                sources=[],
                actions=["RAG mode skipped order database access."],
            )

        results = retrieve(message, self.knowledge_chunks, k=self.settings.top_k)
        base_decision = decide_route(message, results, "fast")

        return SupportResponse(
            answer=self._generate_rag_answer(message, results),
            route="rag",
            intent=intent,
            confidence=base_decision.confidence,
            sources=format_sources(results),
            actions=["Queried the private knowledge base."],
        )

    def _handle_agent_mode(self, message: str) -> SupportResponse:
        results = retrieve(message, self.knowledge_chunks, k=self.settings.top_k)
        decision = self._build_agent_mode_decision(message, results)
        response = self._handle_with_agent(message, decision)
        if response is not None:
            return response

        fallback_response = self._handle_with_fallback(message, decision, results)
        diagnostic = "LangChain agent unavailable; used grounded fallback pipeline."
        if self.agent_runtime_error:
            diagnostic = (
                f"LangChain agent unavailable ({self.agent_runtime_error}); "
                "used grounded fallback pipeline."
            )

        return SupportResponse(
            answer=fallback_response.answer,
            route=fallback_response.route,
            intent=fallback_response.intent,
            confidence=fallback_response.confidence,
            sources=fallback_response.sources,
            actions=self._dedupe_list([diagnostic, *fallback_response.actions]),
            ticket_id=fallback_response.ticket_id,
        )

    def _build_agent_mode_decision(self, message: str, results) -> RouteDecision:
        base_decision = decide_route(message, results, "full")
        return RouteDecision(
            intent=base_decision.intent,
            route="agent",
            reason="User selected Agent mode.",
            confidence=base_decision.confidence,
            needs_order_lookup=base_decision.needs_order_lookup,
            should_create_ticket=base_decision.should_create_ticket,
            should_escalate=base_decision.should_escalate,
        )

    def _handle_with_agent(
        self, message: str, decision: RouteDecision
    ) -> SupportResponse | None:
        if decision.route not in {"full_power", "agent"} or self.agent_runtime is None:
            return None

        try:
            agent_result = self.agent_runtime.invoke(message, self.history, decision)
            self.agent_runtime_error = None
        except Exception as exc:
            self.agent_runtime_error = f"{type(exc).__name__}: {exc}"
            return None

        answer = agent_result.answer
        actions = agent_result.actions.copy()
        sources = agent_result.sources.copy()
        ticket_id = agent_result.ticket_id

        if not sources and decision.intent != "order_status":
            fallback_results = retrieve(
                message, self.knowledge_chunks, k=self.settings.top_k
            )
            sources = format_sources(fallback_results)
            if sources and "Queried the private knowledge base." not in actions:
                actions.insert(0, "Queried the private knowledge base.")
            answer = self._generate_template_answer(
                message, decision, fallback_results, None
            )

        if decision.should_create_ticket and not ticket_id:
            summary = build_ticket_summary(message, answer, actions)
            ticket = create_ticket(
                self.settings.logs_dir,
                customer_message=message,
                intent=decision.intent,
                reason=decision.reason,
                response_excerpt=summary,
            )
            ticket_id = ticket["ticket_id"]
            actions.append(f"Created support ticket {ticket_id}.")

        if decision.should_escalate or agent_result.escalated:
            if "Flagged conversation for human review." not in actions:
                actions.append("Flagged conversation for human review.")
            follow_up = "I have marked this case for human follow-up."
            if follow_up not in answer:
                answer = f"{answer}\n\n{follow_up}"

        return SupportResponse(
            answer=answer,
            route=decision.route,
            intent=decision.intent,
            confidence=decision.confidence,
            sources=self._dedupe_list(sources),
            actions=self._dedupe_list(actions),
            ticket_id=ticket_id,
        )

    def _handle_with_fallback(
        self,
        message: str,
        decision: RouteDecision,
        results,
    ) -> SupportResponse:
        actions: list[str] = []
        order_info = self._maybe_lookup_order(message, decision, actions)
        answer = self._generate_answer(message, decision, results, order_info)

        ticket_id = None
        if decision.should_create_ticket:
            summary = build_ticket_summary(message, answer, actions)
            ticket = create_ticket(
                self.settings.logs_dir,
                customer_message=message,
                intent=decision.intent,
                reason=decision.reason,
                response_excerpt=summary,
            )
            ticket_id = ticket["ticket_id"]
            actions.append(f"Created support ticket {ticket_id}.")

        if decision.should_escalate:
            follow_up = "I have marked this case for human follow-up."
            if follow_up not in answer:
                answer = f"{answer}\n\n{follow_up}"
            actions.append("Flagged conversation for human review.")

        return SupportResponse(
            answer=answer,
            route=decision.route,
            intent=decision.intent,
            confidence=decision.confidence,
            sources=format_sources(results),
            actions=self._dedupe_list(actions),
            ticket_id=ticket_id,
        )

    def _handle_direct_calculation(
        self, message: str, selected_mode: str
    ) -> SupportResponse | None:
        if selected_mode not in {"agent", "auto", "fast", "full"}:
            return None

        if detect_intent(message) != "calculation":
            return None

        result = evaluate_simple_calculation(message)
        if result is None:
            return None

        if selected_mode == "agent":
            route = "agent"
        elif selected_mode == "fast":
            route = "fast_private"
        else:
            route = "full_power"

        return SupportResponse(
            answer=result,
            route=route,
            intent="calculation",
            confidence=1.0,
            sources=[],
            actions=[],
        )

    def _maybe_lookup_order(
        self, message: str, decision: RouteDecision, actions: list[str]
    ) -> dict[str, Any] | None:
        if not decision.needs_order_lookup:
            return None

        order_id = extract_order_id(message)
        if not order_id:
            actions.append("Order lookup skipped because no order number was found.")
            return None

        order_info = lookup_order_status(self.settings.data_dir / "orders.json", order_id)
        if order_info:
            actions.append(f"Checked simulated order database for {order_id}.")
        else:
            actions.append(f"Order lookup did not find {order_id}.")
        return order_info

    def _generate_answer(
        self,
        message: str,
        decision: RouteDecision,
        results,
        order_info: dict[str, Any] | None,
    ) -> str:
        context = build_context(results)

        if self.settings.enable_ollama and decision.route != "fast_private":
            ollama_answer = self._generate_with_ollama(
                message=message,
                decision=decision,
                context=context,
                order_info=order_info,
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
    ) -> str:
        system_prompt = (
            "You are a careful customer support assistant for TechGear Store. "
            "Use only the supplied context and tool output. "
            "Do not invent policies, order details, exceptions, or next steps that are not supported by the context. "
            "If the evidence is weak, say that clearly and suggest escalation. "
            "Do not add greetings, signatures, or channel instructions."
        )
        history = self._history_summary()
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
        results,
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
        elif any(term in lower for term in ("flicker", "flickering", "not working", "problem", "issue", "broken")):
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

    def _generate_rag_answer(self, message: str, results) -> str:
        lower = message.lower()
        lines: list[str] = []

        if not results:
            lines.append(
                "I do not have enough verified knowledge for a confident answer from the local documents."
            )
            lines.append(
                "RAG mode only searches the local knowledge base, so unsupported questions should be treated as unknown."
            )
            return "\n".join(lines)

        if any(term in lower for term in ("flicker", "flickering", "not working", "problem", "issue", "broken")):
            lines.append(
                "Here is the grounded guidance retrieved from the local knowledge base."
            )
        else:
            lines.append(
                "Here is the grounded information retrieved from the local knowledge base."
            )

        lines.append("Relevant guidance:")
        for result in results[:2]:
            excerpt = result.chunk.text.replace("\n", " ").strip()
            if len(excerpt) > 220:
                excerpt = excerpt[:217] + "..."
            lines.append(f"- {result.chunk.title}: {excerpt}")

        return "\n".join(lines)

    def _raw_model_system_prompt(self) -> str:
        return (
            "You are being tested as a standalone language model. "
            "You have no access to local files, project code, vector stores, private knowledge bases, databases, or external tools unless the user pasted that content into the prompt. "
            "Answer from general knowledge and reasoning only. "
            "If the user asks about local project files or internal data, state clearly that you cannot access them."
        )

    def _raw_model_user_prompt(self, message: str) -> str:
        return (
            f"Conversation history: {self._history_summary()}\n\n"
            f"User message:\n{message}\n\n"
            "Reply helpfully and concisely."
        )

    def _history_summary(self) -> str:
        if not self.history:
            return "No previous messages."
        recent_turns = self.history[-self.settings.max_history_messages :]
        return " | ".join(f"{turn['role']}: {turn['content']}" for turn in recent_turns)

    def _remember(self, message: str, answer: str) -> None:
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-self.settings.max_history_messages :]

    def _log_interaction(
        self, message: str, response: SupportResponse, selected_mode: str
    ) -> None:
        log_interaction(
            self.settings.logs_dir,
            {
                "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "mode": selected_mode,
                "route": response.route,
                "intent": response.intent,
                "confidence": round(response.confidence, 3),
                "customer_message": redact_sensitive_text(message),
                "assistant_answer": redact_sensitive_text(response.answer),
                "sources": response.sources,
                "actions": [redact_sensitive_text(action) for action in response.actions],
                "ticket_id": response.ticket_id,
            },
        )

    def _dedupe_list(self, items: list[str]) -> list[str]:
        unique_items: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        return unique_items


