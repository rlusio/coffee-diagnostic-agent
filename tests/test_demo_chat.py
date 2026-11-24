import json
import unittest
from unittest.mock import patch

from demo_chat import read_observations, run_observation_demo


class TestObservationDemo(unittest.TestCase):
    def test_unconfirmed_activities_and_hot_water_phase(self):
        for message, phase, expected in (
            ("przy goracej wodzie", "w trakcie nalewania", "brak informacji"),
            ("przy goracej wodzie ale juz po bardziej", "po zakończeniu nalewania", "brak informacji"),
            ("przy gorącej wodzie dopiero na końcu", "po zakończeniu nalewania", "po zakończeniu nalewania"),
        ):
            with self.subTest(message=message):
                reply = {"Płukanie": "nie sprawdzono", "Gorąca woda": "tak",
                         "Parzenie kawy": "nie", "Moment przy gorącej wodzie": phase}
                result = read_observations(json.dumps(reply), message)
                self.assertEqual(result["Płukanie"], "brak informacji")
                self.assertEqual(result["Parzenie kawy"], "brak informacji")
                self.assertEqual(result["Gorąca woda"], "tak")
                self.assertEqual(result["Moment przy gorącej wodzie"], expected)

    @patch("builtins.print")
    @patch("builtins.input", side_effect=["ekspres cieknie przy parzeniu kawy", ""])
    @patch("demo_chat.llm")
    def test_coffee_observation_cannot_set_phase_for_unknown_hot_water(self, model, user_input, output):
        model.invoke.return_value.content = json.dumps({
            "Płukanie": "brak informacji",
            "Gorąca woda": "brak informacji",
            "Parzenie kawy": "tak",
            "Moment przy gorącej wodzie": "po zakończeniu nalewania",
        })
        run_observation_demo()
        messages = model.invoke.call_args.args[0]
        self.assertIn("Kiedy cieknie?", messages[-1].content)
        self.assertIn("ekspres cieknie przy parzeniu kawy", messages[-1].content)
        output.assert_any_call("Moment przy gorącej wodzie: brak informacji")
        output.assert_any_call("Parzenie kawy: tak")
