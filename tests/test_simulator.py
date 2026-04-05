import json
import tempfile
import unittest
from pathlib import Path

from src.config import Settings
from src.simulator import SupportSimulator


class SimulatorCalculationTests(unittest.TestCase):
    def _build_settings(self, project_root: Path) -> Settings:
        knowledge_base_dir = project_root / "knowledge_base"
        data_dir = project_root / "data"
        logs_dir = project_root / "logs"
        knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        return Settings(
            project_root=project_root,
            knowledge_base_dir=knowledge_base_dir,
            data_dir=data_dir,
            logs_dir=logs_dir,
            default_mode="agent",
            enable_ollama=False,
            enable_langchain_agent=False,
            fast_model="phi3:3.8b",
            full_model="llama3.1:8b",
            ollama_url="http://localhost:11434",
            top_k=3,
            max_history_messages=8,
        )

    def test_full_intelligence_mode_answers_simple_calculation_without_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = self._build_settings(Path(temp_dir))
            simulator = SupportSimulator(settings)

            response = simulator.handle_message(
                "what is 560 minus 45?",
                mode="Full Intelligence Mode",
            )

            self.assertEqual(response.answer, "515")
            self.assertEqual(response.route, "agent")
            self.assertEqual(response.intent, "calculation")
            self.assertEqual(response.confidence, 1.0)
            self.assertEqual(response.sources, [])
            self.assertEqual(response.actions, [])
            self.assertIsNone(response.ticket_id)

            tickets = json.loads(
                (settings.logs_dir / "tickets.json").read_text(encoding="utf-8-sig")
            )
            self.assertEqual(tickets, [])


if __name__ == "__main__":
    unittest.main()
