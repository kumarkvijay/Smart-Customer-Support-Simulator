from __future__ import annotations

import argparse
from typing import Optional

from src.config import LEGACY_MODES, PRIMARY_MODES, is_supported_mode, normalize_mode
from src.simulator import SupportSimulator, SupportResponse


def format_response(response: SupportResponse) -> str:
    lines = [
        "",
        f"Route: {response.route}",
        f"Intent: {response.intent}",
        f"Confidence: {response.confidence:.0%}",
        "",
        "Assistant:",
        response.answer,
    ]

    if response.sources:
        lines.append("")
        lines.append("Sources:")
        lines.extend(f"- {source}" for source in response.sources)

    if response.actions:
        lines.append("")
        lines.append("Actions:")
        lines.extend(f"- {action}" for action in response.actions)

    if response.ticket_id:
        lines.append("")
        lines.append(f"Ticket: {response.ticket_id}")

    return "\n".join(lines)


def run_interactive(simulator: SupportSimulator, starting_mode: str) -> None:
    mode = normalize_mode(starting_mode, simulator.settings.default_mode)
    primary_modes = "|".join(PRIMARY_MODES)
    legacy_modes = "|".join(LEGACY_MODES)

    print("Smart Customer Support Simulator")
    print(f"Commands: /mode {primary_modes}, /help, /quit")
    print(f"Legacy compatibility modes: {legacy_modes}")
    print(f"Current mode: {mode}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input in {"/quit", "quit", "exit"}:
            print("Session closed.")
            return

        if user_input == "/help":
            print("Ask a support question or use one of the commands.")
            print("Primary modes:")
            print("- raw_llm: direct full-size Ollama model without RAG or tools")
            print("- raw_slm: direct small Ollama model without RAG or tools")
            print("- agent: LangChain agent with retrieval, order lookup, ticketing, and escalation")
            print("- rag: knowledge-base retrieval only, no agent and no order database")
            print("Examples:")
            print('- "/mode raw_llm"')
            print('- "What was Deepak Nitrite Q3 FY26 revenue?"')
            print('- "/mode rag"')
            print('- "My laptop screen is flickering, what should I do?"')
            continue

        if user_input.startswith("/mode "):
            requested_mode = user_input.split(" ", 1)[1].strip()
            if not is_supported_mode(requested_mode):
                print("Invalid mode. Use raw_llm, raw_slm, agent, rag, or legacy auto, fast, full.")
                continue
            mode = normalize_mode(requested_mode, simulator.settings.default_mode)
            print(f"Mode updated to: {mode}")
            continue

        response = simulator.handle_message(user_input, mode=mode)
        print(format_response(response))


def run_single_message(
    simulator: SupportSimulator, mode: str, message: str
) -> SupportResponse:
    response = simulator.handle_message(message, mode=mode)
    print(format_response(response))
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privacy-first Smart Customer Support Simulator"
    )
    parser.add_argument(
        "--mode",
        default="agent",
        help="Execution mode: raw_llm, raw_slm, agent, rag. Legacy: auto, fast, full.",
    )
    parser.add_argument(
        "--message",
        help="Run a single support message and exit.",
    )
    return parser.parse_args()


def main() -> Optional[SupportResponse]:
    args = parse_args()
    simulator = SupportSimulator()

    if not is_supported_mode(args.mode):
        valid_modes = ", ".join((*PRIMARY_MODES, *LEGACY_MODES))
        raise SystemExit(f"Invalid mode '{args.mode}'. Use one of: {valid_modes}.")

    selected_mode = normalize_mode(args.mode, simulator.settings.default_mode)

    if args.message:
        return run_single_message(simulator, selected_mode, args.message)

    run_interactive(simulator, selected_mode)
    return None


if __name__ == "__main__":
    main()
