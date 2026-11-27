import json
import unittest
from unittest.mock import patch

from src.agent import build_agent, format_observations


class ComponentDialogueTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_agent()
        self.llm = self.enterContext(patch("src.agent.llm"))

    def turn(self, message, observations=(), thread="case", **device):
        extracted = [{"field": field, "value": value, "quote": quote}
                     for field, value, quote in observations]
        self.llm.invoke.return_value.content = json.dumps({"observations": extracted})
        config = {"configurable": {"thread_id": thread}}
        return self.graph.invoke({"message": message, **device}, config)

    def test_water_hypotheses_without_error_code(self):
        scenarios = [
            ("Pompa pracuje. Woda płynie normalnie, ale cykl się przerywa.",
             [("pump_audible", "tak", "Pompa pracuje"), ("water_flow", "normalnie", "płynie normalnie."),
              ("cycle_error", "tak", "cykl się przerywa")], "water_measurement"),
            ("Pompa milczy i woda nie leci.",
             [("pump_audible", "nie", "Pompa milczy"), ("water_flow", "wcale", "woda nie leci")],
             "water_pump_silent"),
            ("Po wymianie filtra pompa jest głośna.",
             [("air_trigger", "tak", "Po wymianie filtra"), ("pump_loud", "tak", "pompa jest głośna")],
             "water_air")]
        for message, observations, expected in scenarios:
            with self.subTest(expected):
                result = self.turn(message, observations, thread=expected)
                self.assertEqual([r["id"] for r in result["candidates"]], [expected])
                self.assertGreaterEqual(len(format_observations(result)), len(observations))
                self.assertIn("Serwis:", result["answer"])
                self.assertIsNone(result["pending_question"])

    def test_trickle_and_unknown_history(self):
        first = self.turn("Pompa działa. Woda ledwie kapie.",
                          [("pump_audible", "tak", "Pompa działa"), ("water_flow", "kroplami", "ledwie kapie")])
        self.assertEqual(first["pending_question"], "air_trigger")
        second = self.turn("nie wiem")
        self.assertEqual(second["candidates"][0]["id"], "water_restricted")
        self.assertIsNone(second["pending_question"])

    def test_grinder_cases(self):
        scenarios = [
            ("Młynek mieli. Sucha kawa rozsypuje się w środku.",
             [("grinder_audible", "tak", "Młynek mieli"), ("loose_grounds", "tak", "Sucha kawa rozsypuje się w środku")],
             "grinder_chute_damage"),
            ("Młynek pracuje, kawa zalega w kanale.",
             [("grinder_audible", "tak", "Młynek pracuje"), ("chute_blocked", "tak", "kawa zalega w kanale")],
             "grinder_chute_blocked"),
            ("Młynek nie wydaje dźwięku.", [("grinder_audible", "nie", "Młynek nie wydaje dźwięku")],
             "grinder_silent"),
            ("Młynek próbuje ruszyć i staje. Wpadł kamyk.",
             [("grinder_abnormal", "tak", "próbuje ruszyć i staje"), ("foreign_body", "tak", "Wpadł kamyk")],
             "grinder_stalled")]
        for message, observations, expected in scenarios:
            with self.subTest(expected):
                result = self.turn(message, observations, expected, brand="Saeco", model="GranAroma")
                self.assertEqual(result["candidates"][0]["id"], expected)

    def test_negative_answer_and_conflicting_report(self):
        first = self.turn("Młynek nie mieli", brand="Philips")
        self.assertEqual(first["pending_question"], "grinder_audible")
        second = self.turn("nie")
        self.assertEqual(second["component_facts"], {"grinder_audible": "nie"})
        third = self.turn("Słyszę młynek", [("grinder_audible", "tak", "Słyszę młynek")])
        self.assertEqual(third["component_conflicts"], ["grinder_audible"])
        self.assertFalse(third["candidates"])
        final = self.turn("tak")
        self.assertEqual(final["component_facts"]["grinder_audible"], "tak")
        self.assertEqual(final["component_conflicts"], [])

    def test_code_and_group_scope(self):
        saeco = self.turn("Błąd 05", brand="Saeco Gran Aroma")
        self.assertIn("powietrze", saeco["answer"])
        other = self.turn("Pompa milczy, błąd 05", thread="other", brand="Jura")
        self.assertNotIn("granaroma_05", other["component_facts"])
        reply = self.turn("Młynek pracuje, kawa zalega w kanale.",
                          [("grinder_audible", "tak", "Młynek pracuje"), ("chute_blocked", "tak", "zalega w kanale")],
                          thread="group", brand="Philips")
        self.assertEqual(reply["pending_question"], "removable_group")
        reply = self.turn("tak", thread="group")
        self.assertEqual(reply["candidates"][0]["id"], "grinder_chute_blocked")

    def test_flooding_and_missing_evidence(self):
        result = self.turn("Do młynka dostała się woda", [("wet_grinder", "tak", "dostała się woda"),
                           ("grinder_abnormal", "tak", "obcy cytat")], brand="Saeco")
        self.assertIn("odłącz zasilanie", result["answer"])
        self.assertNotIn("grinder_abnormal", result["component_facts"])
        self.assertFalse(result["candidates"])
