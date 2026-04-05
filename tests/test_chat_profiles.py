import unittest

from src.chat_profiles import get_chat_profile, get_chat_profile_for_mode, normalize_chat_profile_key
from src.config import is_supported_mode, normalize_mode
from src.knowledge_base import KnowledgeChunk, retrieve
from src.router import decide_route, evaluate_simple_calculation


class ChatProfileTests(unittest.TestCase):
    def test_fast_private_label_maps_to_raw_slm(self) -> None:
        self.assertTrue(is_supported_mode("Fast & Private Mode"))
        self.assertEqual(normalize_mode("Fast & Private Mode"), "raw_slm")

    def test_full_intelligence_label_maps_to_agent(self) -> None:
        self.assertTrue(is_supported_mode("Full Intelligence Mode"))
        self.assertEqual(normalize_mode("Full Intelligence Mode"), "agent")

    def test_profile_aliases_normalize_to_expected_keys(self) -> None:
        self.assertEqual(normalize_chat_profile_key("SLM only"), "fast_private")
        self.assertEqual(normalize_chat_profile_key("full_intelligence_mode"), "full_intelligence")

    def test_get_chat_profile_returns_phase_five_profile(self) -> None:
        profile = get_chat_profile("Fast & Private Mode")
        self.assertEqual(profile.label, "Fast & Private Mode")
        self.assertEqual(profile.simulator_mode, "raw_slm")

    def test_get_chat_profile_for_mode_returns_none_for_non_phase_five_mode(self) -> None:
        self.assertIsNone(get_chat_profile_for_mode("rag"))

    def test_calculation_query_does_not_trigger_escalation_or_ticket(self) -> None:
        decision = decide_route("590 minus 35 equals what?", [], "full")
        self.assertEqual(decision.intent, "calculation")
        self.assertFalse(decision.should_escalate)
        self.assertFalse(decision.should_create_ticket)

    def test_simple_calculation_is_evaluated_deterministically(self) -> None:
        self.assertEqual(evaluate_simple_calculation("590 minus 35 equals what?"), "555")

    def test_numeric_overlap_alone_does_not_count_as_retrieval_match(self) -> None:
        chunk = KnowledgeChunk(
            chunk_id="kb-1",
            source="knowledge_base/manuals/IDFC bank.md",
            category="manuals",
            title="Idfc Bank",
            text="Potential fraud impact was 590 crores and insurance recovery was 35 crores.",
            tokens=("potential", "fraud", "impact", "590", "insurance", "35", "crores"),
        )
        self.assertEqual(retrieve("590 minus 35 equals what?", [chunk]), [])


if __name__ == "__main__":
    unittest.main()
