import unittest

import main
from src.simulator import SupportResponse


class MainCliTests(unittest.TestCase):
    def test_parse_args_accepts_friendly_mode_label(self) -> None:
        args = main.parse_args(["--mode", "Fast & Private Mode", "--message", "hello"])
        self.assertEqual(args.mode, "Fast & Private Mode")
        self.assertEqual(args.message, "hello")

    def test_describe_mode_includes_phase_five_label(self) -> None:
        self.assertEqual(
            main.describe_mode("agent"),
            "agent (Full Intelligence Mode)",
        )

    def test_format_response_includes_optional_sections(self) -> None:
        response = SupportResponse(
            answer="Grounded answer.",
            route="agent",
            intent="general_support",
            confidence=0.85,
            sources=["knowledge_base/faqs/general_faq.json"],
            actions=["Queried the private knowledge base."],
            ticket_id="TKT-1001",
        )

        formatted = main.format_response(response)

        self.assertIn("Route: agent", formatted)
        self.assertIn("Confidence: 85%", formatted)
        self.assertIn("Sources:", formatted)
        self.assertIn("Actions:", formatted)
        self.assertIn("Ticket: TKT-1001", formatted)


if __name__ == "__main__":
    unittest.main()
