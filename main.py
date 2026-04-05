from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Optional

from src.chat_profiles import get_chat_profile_for_mode, list_chat_profiles
from src.config import LEGACY_MODES, PRIMARY_MODES, is_supported_mode, normalize_mode
from src.simulator import SupportResponse, SupportSimulator


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


def describe_mode(mode: str) -> str:
    normalized_mode = normalize_mode(mode)
    profile = get_chat_profile_for_mode(normalized_mode)
    if profile is None:
        return normalized_mode
    return f"{normalized_mode} ({profile.label})"


def print_profile_help() -> None:
    print("Phase 5 chat profiles:")
    for profile in list_chat_profiles():
        print(f"- {profile.label}: {profile.description} [maps to {profile.simulator_mode}]")


def print_interactive_help() -> None:
    primary_modes = ", ".join(PRIMARY_MODES)
    legacy_modes = ", ".join(LEGACY_MODES)
    print("Ask a support question or use one of the commands.")
    print("Commands:")
    print("- /mode <mode or profile label>")
    print("- /profiles")
    print("- /help")
    print("- /quit")
    print(f"Primary modes: {primary_modes}")
    print(f"Legacy compatibility modes: {legacy_modes}")
    print_profile_help()
    print("Examples:")
    print('- "/mode Fast & Private Mode"')
    print('- "/mode Full Intelligence Mode"')
    print('- "/mode rag"')
    print('- "What was Deepak Nitrite Q3 FY26 revenue?"')


def run_interactive(simulator: SupportSimulator, starting_mode: str) -> None:
    mode = normalize_mode(starting_mode, simulator.settings.default_mode)

    print("Smart Customer Support Simulator")
    print("Commands: /mode <mode>, /profiles, /help, /quit")
    print_profile_help()
    print(f"Current mode: {describe_mode(mode)}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input in {"/quit", "quit", "exit"}:
            print("Session closed.")
            return

        if user_input == "/help":
            print_interactive_help()
            continue

        if user_input == "/profiles":
            print_profile_help()
            print(f"Current mode: {describe_mode(mode)}")
            continue

        if user_input.startswith("/mode "):
            requested_mode = user_input.split(" ", 1)[1].strip()
            if not is_supported_mode(requested_mode):
                print(
                    "Invalid mode. Use raw_llm, raw_slm, agent, rag, legacy auto/fast/full, or the Phase 5 labels Fast & Private Mode / Full Intelligence Mode."
                )
                continue
            mode = normalize_mode(requested_mode, simulator.settings.default_mode)
            print(f"Mode updated to: {describe_mode(mode)}")
            continue

        response = simulator.handle_message(user_input, mode=mode)
        print(format_response(response))


def run_single_message(
    simulator: SupportSimulator, mode: str, message: str
) -> SupportResponse:
    response = simulator.handle_message(message, mode=mode)
    print(format_response(response))
    return response


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privacy-first Smart Customer Support Simulator"
    )
    parser.add_argument(
        "--mode",
        default="agent",
        help=(
            "Execution mode: raw_llm, raw_slm, agent, rag. Legacy: auto, fast, full. "
            "Friendly aliases: 'Fast & Private Mode' maps to raw_slm and 'Full Intelligence Mode' maps to agent."
        ),
    )
    parser.add_argument(
        "--message",
        help="Run a single support message and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Optional[SupportResponse]:
    args = parse_args(argv)
    simulator = SupportSimulator()

    if not is_supported_mode(args.mode):
        valid_modes = ", ".join((*PRIMARY_MODES, *LEGACY_MODES))
        raise SystemExit(
            f"Invalid mode '{args.mode}'. Use one of: {valid_modes}, Fast & Private Mode, Full Intelligence Mode."
        )

    selected_mode = normalize_mode(args.mode, simulator.settings.default_mode)

    if args.message:
        return run_single_message(simulator, selected_mode, args.message)

    run_interactive(simulator, selected_mode)
    return None


if __name__ == "__main__":
    main()
