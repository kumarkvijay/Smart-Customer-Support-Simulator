import tempfile
import unittest
from pathlib import Path

from src.agent import SupportLangChainAgent
from src.config import Settings


class AgentCompatibilityTests(unittest.TestCase):
    def test_support_langchain_agent_initializes_with_installed_langgraph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            settings = Settings(
                project_root=project_root,
                knowledge_base_dir=project_root / "knowledge_base",
                data_dir=project_root / "data",
                logs_dir=project_root / "logs",
                default_mode="agent",
                enable_ollama=False,
                enable_langchain_agent=True,
                fast_model="phi3:3.8b",
                full_model="llama3.1:8b",
                ollama_url="http://localhost:11434",
                top_k=3,
                max_history_messages=8,
            )

            agent = SupportLangChainAgent(settings, [])

            self.assertIsNotNone(agent.agent)


if __name__ == "__main__":
    unittest.main()
