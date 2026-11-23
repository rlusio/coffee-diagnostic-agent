import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import app


class DiagnosticCliTests(unittest.TestCase):
    def test_complete_case_and_finish_an_independent_case_early(self):
        replies = [
            "Młynek nie mieli", "Melitta", "", "nie", "nie", "Barista TS",
            "tak", "", "nie",
            "Młynek nie mieli", "", "", "", "nie",
            "",
        ]
        output = io.StringIO()
        with (
            patch("builtins.input", side_effect=replies),
            patch("app.log_search") as log_search,
            patch("app.submit_case", return_value=("case-1", True)) as submit,
            patch("src.agent.llm") as llm,
            redirect_stdout(output),
        ):
            app.main()

        submit.assert_called_once()
        case = submit.call_args.kwargs
        self.assertEqual(case["model"], "Barista TS")
        self.assertIn("Słychać młynek: nie", case["symptoms"])
        self.assertIn("Komunikat o braku ziaren: nie", case["symptoms"])
        self.assertEqual(log_search.call_count, 2)
        self.assertEqual(log_search.call_args.kwargs["model"], "")
        transcript = output.getvalue()
        self.assertEqual(transcript.count("Podsumowanie przypadku"), 2)
        self.assertEqual(transcript.count("Czy urządzenie wyświetla komunikat o braku ziaren?"), 2)
        self.assertEqual(transcript.count("Słychać młynek: nie"), 1)
        self.assertIn("Numer zgłoszenia: case-1", transcript)
        llm.invoke.assert_not_called()
