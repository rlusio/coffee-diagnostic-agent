import unittest
from ollama import ResponseError

from src.agent import decide
from unittest.mock import patch
from src.agent import build_agent, finish_reply, update_facts

class TestAgentRouting(unittest.TestCase):

    @patch("src.agent.llm")
    def test_granaroma_asks_what_at_coffee_time_means(self, llm):
        agent = build_agent()
        for reply, timing in (("dopiero jak skończy robić kawę", "po zakończeniu parzenia"),
                              ("już kiedy kawa leci do filiżanki", "podczas parzenia"),
                              ("nie wiem", "przy kawie")):
            with self.subTest(reply=reply):
                config = {"configurable": {"thread_id": reply}}
                agent.invoke({"brand": "saeco", "model": "granaroma", "message": "ekspres cieknie"}, config)
                agent.invoke({"message": "z tylu"}, config)
                state = agent.invoke({"message": "przy kawie"}, config)
                self.assertEqual(state["fault_timing"], "przy kawie")
                self.assertEqual(state["pending_question"], "coffee_leak_phase")
                state = agent.invoke({"message": reply}, config)
                self.assertEqual(state["fault_timing"], timing)
                self.assertEqual(state["pending_question"], "rear_leak_during_hot_water")
                state = agent.invoke({"message": "nie"}, config)
                self.assertEqual("obok przepływomierza" in state["answer"], timing == "po zakończeniu parzenia")
        llm.invoke.assert_not_called()

    @patch("src.agent.llm")
    def test_short_leak_report_uses_model_location_and_timing(self, llm):
        agent = build_agent()
        config = {"configurable": {"thread_id": "short-leak"}}
        state = agent.invoke({"message": "cieknie", "brand": "saeco", "model": "gran aroma"}, config)
        self.assertEqual(state["pending_question"], "leak_location")
        state = agent.invoke({"message": "z tylu"}, config)
        self.assertEqual(state["pending_question"], "fault_timing")
        state = agent.invoke({"message": "po zrobieniu kawy"}, config)
        self.assertEqual(state["pending_question"], "rear_leak_during_hot_water")
        state = agent.invoke({"message": "tak"}, config)
        self.assertEqual(state["pending_question"], "hot_water_leak_phase")
        state = agent.invoke({"message": "nie wiem"}, config)
        self.assertIsNone(state["pending_question"])
        self.assertIn("obok przepływomierza", state["answer"])
        llm.invoke.assert_not_called()

    @patch("src.agent.llm")
    def test_granaroma_leak_with_two_observed_timings(self, llm):
        agent = build_agent()
        config = {"configurable": {"thread_id": "two-timings"}}
        first = agent.invoke({"message": "ekspres przecieka", "brand": "saeco",
                              "model": "gran aroma"}, config)
        self.assertEqual(first["pending_question"], "leak_location")
        timing = "podczas płukania oraz gorącej wody, kawy nie robiłem bo nie chciałem zalać go"
        second = agent.invoke({"message": timing}, config)
        self.assertEqual(second["fault_timing"], "podczas płukania, podczas poboru gorącej wody")
        self.assertEqual(second["answer"], "Gdzie pojawia się woda?")
        self.assertIsNone(second.get("hot_water_flow"))
        final = agent.invoke({"message": "z tyłu"}, config)
        self.assertEqual(final["pending_question"], "hot_water_leak_phase")
        final = agent.invoke({"message": "na końcu"}, config)
        self.assertIsNone(final["pending_question"])
        self.assertIn("obok przepływomierza", final["answer"])
        llm.invoke.assert_not_called()

    @patch("src.agent.llm")
    def test_granaroma_rear_leak_during_rinse(self, llm):
        agent = build_agent()
        for reply, expected in (("tak", True), ("nie", False), ("nie wiem", None)):
            with self.subTest(reply=reply):
                config = {"configurable": {"thread_id": reply}}
                agent.invoke({"message": "Woda leje się z ekspresu", "brand": "saeco",
                              "model": "granaroma"}, config)
                agent.invoke({"message": "z tyłu"}, config)
                state = agent.invoke({"message": "w trakcie płukania od razu"}, config)
                self.assertEqual(state["pending_question"], "rear_leak_during_hot_water")
                final = agent.invoke({"message": reply}, config)
                if expected is True:
                    self.assertEqual(final["pending_question"], "hot_water_leak_phase")
                    final = agent.invoke({"message": "w trakcie nalewania"}, config)
                    self.assertIn("czy zawór się domyka", final["answer"])
                self.assertIs(final.get("rear_leak_during_hot_water"), expected)
                self.assertIsNone(final.get("hot_water_flow"))
                self.assertIsNone(final["pending_question"])
                self.assertIn("podczas płukania", final["answer"])
                self.assertIn("Trop z przypadku serwisowego tej serii", final["answer"])
                self.assertIn("obok przepływomierza", final["answer"])
        llm.invoke.assert_not_called()

    def test_granaroma_discharge_timings(self):
        agent = build_agent()
        for timing, hose_expected in (("tuż po zaparzeniu kawy", True), ("po spienianiu mleka", True),
                                      ("podczas czyszczenia układu mlecznego", True),
                                      ("podczas parzenia", False), ("w trakcie nalewania gorącej wody", False)):
            with self.subTest(timing=timing):
                config = {"configurable": {"thread_id": timing}}
                agent.invoke({"message": "ekspres przecieka", "brand": "Saeco", "model": "GranAroma"}, config)
                agent.invoke({"message": "z tyłu"}, config)
                result = agent.invoke({"message": timing}, config)
                if result["pending_question"] == "rear_leak_during_hot_water":
                    result = agent.invoke({"message": "nie"}, config)
                self.assertIsNone(result["pending_question"])
                self.assertEqual("obok przepływomierza" in result["answer"], hose_expected)

    @patch("src.agent.llm")
    def test_external_leak_with_brand_and_series_in_one_field(self, llm):
        agent = build_agent()
        config = {"configurable": {"thread_id": "rear-leak"}}
        first = agent.invoke({"brand": "saeco gran aroma", "model": "",
                              "message": "Woda się leje z ekspresu"}, config)
        self.assertEqual(first["brand"], "Saeco")
        self.assertEqual(first["model"], "GranAroma")
        self.assertEqual(first["pending_question"], "leak_location")
        second = agent.invoke({"message": "Woda z tyłu ekspresu"}, config)
        self.assertEqual(second["leak_location"], "z tyłu ekspresu")
        self.assertEqual(second["pending_question"], "fault_timing")
        llm.invoke.assert_not_called()

    def test_short_sound_correction_while_waiting_for_model(self):
        agent = build_agent()
        config = {"configurable": {"thread_id": "short-correction"}}
        for message in ("Młynek nie mieli", "Nie", "Nie", "Jednak słychać"):
            result = agent.invoke({"message": message}, config)
        self.assertIs(result["grinder_audible"], True)
        self.assertIs(result["no_beans_warning"], False)
        self.assertFalse(result.get("model"))
        self.assertEqual(result["pending_question"], "model")
        final = agent.invoke({"message": "Barista TS"}, config)
        self.assertEqual(final["model"], "Barista TS")

    def test_short_sound_reply_without_grinder_context(self):
        state = {"message": "Jednak słychać", "pending_question": "model"}
        self.assertEqual(update_facts(state), {})

    def test_correction_keeps_model_question_open(self):
        agent = build_agent()
        for model_reply in ("Barista TS", "Nie wiem"):
            with self.subTest(model_reply=model_reply):
                config = {"configurable": {"thread_id": model_reply}}
                agent.invoke({"message": "Młynka nie słychać"}, config)
                agent.invoke({"message": "Nie"}, config)
                corrected = agent.invoke({"message": "Jednak słychać młynek"}, config)
                self.assertIs(corrected["grinder_audible"], True)
                self.assertIs(corrected["no_beans_warning"], False)
                self.assertEqual(corrected["pending_question"], "model")
                self.assertEqual(corrected["asked_questions"].count("model"), 1)
                self.assertFalse(corrected.get("model"))
                final = agent.invoke({"message": model_reply}, config)
                self.assertIsNone(final["pending_question"])
                self.assertEqual(final["question_answers"]["model"], model_reply)
                if model_reply == "Nie wiem":
                    self.assertFalse(final.get("model"))
                else:
                    self.assertEqual(final["model"], model_reply)

    def test_grinder_advice_is_shown_once_per_conversation(self):
        agent = build_agent()
        config = {"configurable": {"thread_id": "grinder-advice"}}
        first = agent.invoke({"message": "Młynek nie mieli"}, config)
        self.assertIn("Zweryfikuj poziom ziaren", first["answer"])
        second = agent.invoke({"message": "Nie"}, config)
        self.assertEqual(second["answer"], "Czy przy próbie przygotowania kawy słychać młynek?")
        third = agent.invoke({"message": "Nie"}, config)
        self.assertEqual(third["answer"], "Jaki to model ekspresu?")
        self.assertEqual(third["shown_record_ids"], ["grinder_no_grind_001"])
        fourth = agent.invoke({"message": "Barista TS"}, config)
        self.assertEqual(fourth["model"], "Barista TS")
        self.assertIsNone(fourth["pending_question"])
        self.assertNotIn("?", fourth["answer"])
        self.assertIs(fourth["grinder_audible"], False)
        self.assertIs(fourth["no_beans_warning"], False)
        fresh = agent.invoke({"message": "Młynek nie mieli"}, {"configurable": {"thread_id": "fresh"}})
        self.assertEqual(fresh["answer"], first["answer"])

    @patch("src.agent.llm")
    def test_grinder_checks_come_from_selected_record(self, llm):
        agent = build_agent()
        for output in ("grinder_no_grind_001", "Zweryfikuj połączenie młynka z pojemnikiem."):
            with self.subTest(output=output):
                llm.invoke.return_value.content = output
                config = {"configurable": {"thread_id": output}}
                state = agent.invoke({"message": "Młynek nie mieli"}, config)
                answer = state["answer"]
                self.assertNotIn("połączenie młynka", answer)
                if output == "grinder_no_grind_001":
                    checks = state["candidates"][0]["safe_checks"]
                    self.assertTrue(checks)
                    for check in checks:
                        self.assertIn(check, answer.splitlines())
                self.assertEqual(state["pending_question"], "no_beans_warning")
                llm.invoke.assert_not_called()

    @patch("src.agent.llm")
    def test_leak_prompt_excludes_weak_candidates(self, llm):
        llm.invoke.return_value.content = "Przyczyna wymaga doprecyzowania."
        agent = build_agent()
        config = {"configurable": {"thread_id": "leak-context"}}
        agent.invoke({"message": "Woda pod ekspresem podczas parzenia"}, config)
        agent.invoke({"message": "tak"}, config)
        agent.invoke({"message": "Woda jest czysta. Pompa pracuje."}, config)
        messages = llm.invoke.call_args.args[0]
        prompt = messages[-1].content
        self.assertIn("Gorąca woda działa prawidłowo: tak", prompt)
        self.assertIn("leak_external_generic", prompt)
        self.assertNotIn("hot_water_flow", prompt)
        self.assertNotIn("coffee_not_hot_general", prompt)
        self.assertNotIn("brewing_no_coffee_001", prompt)

    @patch("src.agent.llm")
    def test_cuda_failure_continues_conversation_on_cpu(self, llm):
        llm.invoke.side_effect = ResponseError("CUDA error: the provided PTX was compiled with an unsupported toolchain", 500)
        cpu_model = llm.model_copy.return_value
        cpu_model.invoke.return_value.content = "Opis testowy."
        agent = build_agent()
        config = {"configurable": {"thread_id": "cpu-fallback"}}
        agent.invoke({"message": "Woda pod ekspresem"}, config)
        second = agent.invoke({"message": "Tylko podczas parzenia kawy"}, config)
        self.assertEqual(second["pending_question"], "hot_water_flow")
        third = agent.invoke({"message": "Tak"}, config)
        self.assertTrue(third["hot_water_flow"])
        fourth = agent.invoke({"message": "Woda jest czysta. Pompa pracuje."}, config)
        self.assertTrue(fourth["use_cpu"])
        agent.invoke({"message": "Nie wiem."}, config)
        llm.invoke.assert_called_once()
        llm.model_copy.assert_called_with(update={"num_gpu": 0})
        self.assertEqual(cpu_model.invoke.call_count, 2)

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
            }, config={"configurable": {"thread_id": "weak-match"}})

        self.assertIn(
            "Znalazłem podobne przypadki",
            result["answer"]
        )

    @patch("src.agent.llm")
    def test_location_alone_prompts_for_timing(self, llm):
        agent = build_agent()
        config = {"configurable": {"thread_id": "location-only"}}
        result = agent.invoke({"message": "Woda pod ekspresem"}, config)
        self.assertEqual(result["answer"], "Woda pojawia się pod ekspresem.\nKiedy występuje problem?")
        self.assertEqual(result["pending_question"], "fault_timing")
        self.assertIsNone(result.get("fault_timing"))
        llm.invoke.assert_not_called()

    @patch("src.agent.llm")
    def test_demo_yes_and_echoed_record(self, llm):
        llm.invoke.return_value.content = 'Hipotezy. ID rekordu: [{"id": "leak_external_generic"}]'
        agent = build_agent()
        for reply in ("Tak", "Tak\\"):
            with self.subTest(reply=reply):
                config = {"configurable": {"thread_id": reply}}
                agent.invoke({"message": "woda pod ekspresem"}, config)
                second = agent.invoke({"message": "tylko podczas parzenia kawy"}, config)
                self.assertEqual(second["pending_question"], "hot_water_flow")
                self.assertIn("Wodę widać pod ekspresem podczas parzenia.", second["answer"])
                third = agent.invoke({"message": reply}, config)
                self.assertIs(third["hot_water_flow"], True)
                self.assertNotIn("{", third["answer"])
                self.assertIn("Czy woda jest czysta", third["answer"])
                self.assertIn("Wodę widać pod ekspresem podczas parzenia.", third["answer"])
                llm.invoke.assert_not_called()
                fourth = agent.invoke({"message": "Czysta. Pompa pracuje."}, config)
                llm.invoke.assert_called_once()
                self.assertNotIn("{", fourth["answer"])
                llm.invoke.reset_mock()

    @patch("src.agent.llm")
    def test_conversation_uses_facts_across_turns(self, llm):
        llm.invoke.return_value.content = "Przyczyna pozostaje hipotezą. Gdzie pojawia się woda?"
        agent = build_agent()
        config = {"configurable": {"thread_id": "leak"}}
        first = agent.invoke({"message": "Woda pod ekspresem tylko podczas parzenia."}, config)
        self.assertEqual(first["leak_location"], "pod ekspresem")
        self.assertEqual(first["fault_timing"], "podczas parzenia")
        self.assertEqual(first["pending_question"], "hot_water_flow")
        self.assertEqual(first["answer"].count("?"), 1)
        second = agent.invoke({"message": "Nie."}, config)
        self.assertIs(second["hot_water_flow"], False)
        self.assertEqual(second["leak_location"], first["leak_location"])
        self.assertNotIn(second["pending_question"], ("hot_water_flow", "leak_location", "fault_timing"))
        third = agent.invoke({"message": "Jednak podczas płukania."}, config)
        self.assertEqual(third["fault_timing"], "podczas płukania")
        fresh = agent.invoke({"message": "Młynek nie mieli."}, {"configurable": {"thread_id": "grinder"}})
        self.assertIsNone(fresh.get("hot_water_flow"))
        self.assertNotIn(fresh["pending_question"], ("hot_water_flow", "leak_location", "fault_timing"))

    @patch("src.agent.llm")
    def test_grinder_conversation(self, llm):
        llm.invoke.return_value.content = "Potrzebne są dodatkowe obserwacje."
        agent = build_agent()
        config = {"configurable": {"thread_id": "quiet-grinder"}}
        first = agent.invoke({"message": "Młynka nie słychać."}, config)
        self.assertTrue(first["candidates"], "Brak kandydatów po opisie objawu.")
        self.assertEqual(first["candidates"][0]["id"], "grinder_no_grind_001")
        self.assertIs(first["grinder_audible"], False)
        self.assertEqual(first["pending_question"], "no_beans_warning")
        self.assertIn("komunikat o braku ziaren?", first["answer"])
        second = agent.invoke({"message": "Tak."}, config)
        self.assertTrue(second["candidates"], "Brak kandydatów po odpowiedzi na pytanie.")
        self.assertEqual(second["candidates"][0]["id"], "grinder_no_grind_001")
        self.assertIs(second["no_beans_warning"], True)
        self.assertIs(second["grinder_audible"], False)
        self.assertNotIn(second["pending_question"], ("grinder_audible", "no_beans_warning"))

    def test_explicit_grinder_observations(self):
        state = {"message": "Słychać młynek, nie ma komunikatu o braku ziaren."}
        self.assertEqual(update_facts(state), {"grinder_audible": True, "no_beans_warning": False})
        self.assertEqual(update_facts({"message": "Młynek nie mieli."}), {})
        self.assertEqual(update_facts({"message": "W pojemniku brak ziaren."}), {})
        unknown = update_facts({"message": "Nie wiem", "pending_question": "grinder_audible"})
        self.assertNotIn("grinder_audible", unknown)
        self.assertEqual(unknown["question_answers"]["grinder_audible"], "Nie wiem")

    @patch("src.agent.llm")
    def test_milk_delivery_conversation(self, llm):
        llm.invoke.return_value.content = "Przyczyna wymaga doprecyzowania."
        agent = build_agent()
        config = {"configurable": {"thread_id": "milk"}}
        first = agent.invoke({"message": "Brak pianki."}, config)
        self.assertIsNone(first.get("milk_delivery"))
        self.assertEqual(first["pending_question"], "milk_delivery")
        second = agent.invoke({"message": "Leci słabo."}, config)
        self.assertEqual(second["milk_delivery"], "słaby przepływ mleka")
        self.assertIn("słychać parę", second["answer"])

        config = {"configurable": {"thread_id": "no-milk"}}
        first = agent.invoke({"message": "Mleko nie leci."}, config)
        self.assertEqual(first["milk_delivery"], "brak poboru mleka")
        self.assertIn("słychać parę", first["answer"])

    def test_milk_delivery_requires_context(self):
        self.assertEqual(update_facts({"message": "Leci słabo."}), {})
        facts = update_facts({"message": "Mleko płynie, ale nie tworzy piany."})
        self.assertEqual(facts["milk_delivery"], "mleko płynie bez piany")

    @patch("src.agent.llm")
    def test_power_conversation(self, llm):
        llm.invoke.return_value.content = "Przyczyna wymaga doprecyzowania."
        agent = build_agent()
        config = {"configurable": {"thread_id": "power"}}
        first = agent.invoke({"message": "Ekspres się nie włącza. Gniazdko działa."}, config)
        self.assertIs(first["outlet_works"], True)
        self.assertIsNone(first.get("machine_responds"))
        self.assertEqual(first["pending_question"], "machine_responds")
        second = agent.invoke({"message": "Nie."}, config)
        self.assertIs(second["machine_responds"], False)
        self.assertIs(second["outlet_works"], True)
        self.assertIn("przepięciu lub zalaniu", second["answer"])

    def test_power_observations(self):
        facts = update_facts({"message": "Ciemny wyświetlacz."})
        self.assertNotIn("machine_responds", facts)
        self.assertNotIn("outlet_works", facts)
        facts = update_facts({"message": "Gniazdko nie działa."})
        self.assertIs(facts["outlet_works"], False)
        facts = update_facts({"message": "Dioda miga."})
        self.assertIs(facts["machine_responds"], True)

    def test_unknown_flow_and_hot_water_timing(self):
        state = {"pending_question": "hot_water_flow", "asked_questions": ["hot_water_flow"],
                 "message": "Nie wiem", "leak_location": "w tacce", "fault_timing": "w spoczynku"}
        self.assertNotIn("hot_water_flow", update_facts(state))
        state.update(update_facts(state))
        self.assertNotEqual(finish_reply(state, "Opis.")["pending_question"], "hot_water_flow")
        self.assertEqual(update_facts({"message": "Cieknie podczas poboru gorącej wody."}),
                         {"fault_timing": "podczas poboru gorącej wody"})

if __name__ == "__main__":
    unittest.main()
