import unittest

from src.agent import decide
from unittest.mock import patch
from src.agent import build_agent

class TestAgentRouting(unittest.TestCase):

    def test_no_candidates(self):
        state = {"candidates": []}
        self.assertEqual(decide(state), "clarify")

    def test_weak_match(self):
        state = {
            "candidates": [{"score": 1.25}]
        }
        self.assertEqual(decide(state), "weak_match")

    def test_good_match(self):
        state = {
            "candidates": [{"score": 2.25}]
        }
        self.assertEqual(decide(state), "answer")
        
    def test_full_graph_weak_match(self):
        fake_candidates = [
            {
                "id": "water_001",
                "score": 1.25
            }
        ]

        with patch(
            "src.agent.find_faults",
            return_value=fake_candidates
        ):
            agent = build_agent()

            result = agent.invoke({
                "description": "Ekspres gubi wodę",
                "brand": "Melitta",
                "model": ""
            })

        self.assertIn(
            "Znalazłem podobne przypadki",
            result["answer"]
        )

if __name__ == "__main__":
    unittest.main()
