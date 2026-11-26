import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.granaroma_demo import build_interview


def extraction(location="bez zmiany", noticed="bez zmiany", timing="bez zmiany", breaker="bez zmiany", **extra):
    facts = {"leak_location": location, "noticed_at": noticed, "fault_timing": timing, "breaker_tripped": breaker}
    facts.update(makes_coffee="bez zmiany", tray_excess="bez zmiany", tank_side="bez zmiany")
    facts.update(extra)
    return AIMessage(content=json.dumps(facts))


class TestGranAromaDemo(unittest.TestCase):
    @patch("src.granaroma_demo.llm")
    def test_vague_tray_location_is_clarified(self, model):
        for reply, location, next_question in (("pod spodem", "pod tacką", None),
                                                ("w środku", "w tacce", "tray_excess"),
                                                ("nie wiem", "bez zmiany", None)):
            with self.subTest(reply=reply):
                model.invoke.side_effect = [extraction("w tacce"), extraction(location)]
                graph = build_interview()
                config = {"configurable": {"thread_id": reply}}
                first = graph.invoke({"message": "Jest mokro przy tacce."}, config)
                self.assertEqual(first["pending_question"], "tray_position")
                final = graph.invoke({"message": reply}, config)
                self.assertEqual(final["pending_question"], next_question)

    @patch("src.granaroma_demo.llm")
    def test_under_tray_cleaning_advice(self, model):
        model.invoke.return_value = extraction("pod tacką")
        graph = build_interview()
        config = {"configurable": {"thread_id": "under-tray"}}
        final = graph.invoke({"message": "Kałuża pod tacką"}, config)
        self.assertIsNone(final["pending_question"])
        self.assertIn("Oczyść miejsce", final["answer"])

    @patch("src.granaroma_demo.llm")
    def test_tray_volume_and_tank(self, model):
        for quantity in ("tak", "nie", "nie wiadomo"):
            model.invoke.side_effect = [extraction("w tacce"), extraction(tray_excess=quantity)]
            graph = build_interview()
            config = {"configurable": {"thread_id": quantity}}
            first = graph.invoke({"message": "Woda w tacce"}, config)
            self.assertEqual(first["pending_question"], "tray_excess")
            final = graph.invoke({"message": quantity}, config)
            if quantity == "tak":
                self.assertEqual(final["pending_question"], "makes_coffee")
                final = graph.invoke({"message": "nie"}, config)
            self.assertEqual("zaworu gorącej wody" in final["answer"], quantity == "tak")
            self.assertEqual("zaparzacza" in final["answer"], quantity == "tak")
        model.invoke.side_effect = [extraction("przy zbiorniku na wodę")]
        final = build_interview().invoke({"message": "Z prawego boku przy zbiorniku wody"},
                                         {"configurable": {"thread_id": "tank"}})
        self.assertIn("Pęknięcie pojemnika", final["answer"])

    @patch("src.granaroma_demo.llm")
    def test_full_tray_with_coffee_status_already_known(self, model):
        for coffee, reply in (("nie", "Nie robi kawy"), ("tak", "Robi kawę")):
            model.invoke.return_value = extraction(tray_excess="tak", makes_coffee=coffee)
            state = build_interview().invoke({"message": "Tacka się przelewa. " + reply},
                                             {"configurable": {"thread_id": coffee}})
            self.assertIsNone(state["pending_question"])
            self.assertEqual("zaparzacza" in state["answer"], coffee == "nie")

    @patch("src.granaroma_demo.llm")
    def test_explicit_side_survives_incomplete_extraction(self, model):
        model.invoke.side_effect = [
            extraction(timing="użytkownik nie widział początku lub nie pamięta"),
            extraction(location="z boku", tank_side="nie wiadomo"),
        ]
        graph = build_interview()
        config = {"configurable": {"thread_id": "side-with-coffee"}}
        state = graph.invoke({"message": "Zauważyłem wyciek z boku ekspresu, zrobiłem kilka kaw wcześniej"}, config)
        self.assertEqual(state["leak_location"], "z boku")
        self.assertEqual(state["pending_question"], "tank_side")
        self.assertIsNone(state.get("fault_timing"))
        state = graph.invoke({"message": "bliżej przodu"}, config)
        self.assertEqual(state["pending_question"], "tank_side")
        state = graph.invoke({"message": "po przeciwnej stronie"}, config)
        self.assertEqual(state["pending_question"], "breaker_tripped")
        final = graph.invoke({"message": "nie"}, config)
        self.assertIn("elektrozaworu", final["answer"])

    @patch("src.granaroma_demo.llm")
    def test_side_location_selects_service_record(self, model):
        for reply, cause in (("tak", "złączki nad grzałką"), ("nie", "uszczelka przy połączeniu wężyka"),
                             ("nie wiem", "Brakuje informacji")):
            model.invoke.side_effect = [extraction("z tyłu ekspresu", "po kawie"), extraction("z boku")]
            graph = build_interview()
            config = {"configurable": {"thread_id": reply}}
            first = graph.invoke({"message": "Kałuża była po kawie."}, config)
            self.assertEqual(first["pending_question"], "leak_location")
            second = graph.invoke({"message": "Z boku."}, config)
            self.assertEqual(second["pending_question"], "tank_side")
            side = graph.invoke({"message": "z drugiej"}, config)
            self.assertEqual(side["pending_question"], "breaker_tripped")
            final = graph.invoke({"message": reply}, config)
            self.assertIn(cause, final["answer"])

    @patch("src.granaroma_demo.llm")
    def test_side_near_front_does_not_identify_component(self, model):
        for reply, expect_tank in (("od strony zbiornika", True), ("nie wiem", False)):
            model.invoke.return_value = extraction("z boku", tank_side="po przeciwnej stronie")
            graph = build_interview()
            config = {"configurable": {"thread_id": reply}}
            first = graph.invoke({"message": "Cieknie z boku bliżej przodu"}, config)
            self.assertEqual(first["pending_question"], "tank_side")
            final = graph.invoke({"message": reply}, config)
            self.assertNotIn("elektrozaworu", final["answer"])
            self.assertEqual("Pęknięcie pojemnika" in final["answer"], expect_tank)

    @patch("src.granaroma_demo.llm")
    def test_breaker_stops_interview_even_without_location(self, model):
        model.invoke.return_value = extraction(breaker="tak")
        graph = build_interview()
        final = graph.invoke({"message": "Cieknie i wybiło bezpieczniki."},
                             {"configurable": {"thread_id": "breaker"}})
        self.assertIsNone(final["pending_question"])
        self.assertIn("Nie uruchamiaj", final["answer"])

    @patch("src.granaroma_demo.llm")
    def test_onset_decides_hose_hypothesis(self, model):
        graph = build_interview()
        for timing, expected_hose in (("użytkownik nie widział początku lub nie pamięta", False), ("podczas parzenia", False),
                                      ("pod koniec parzenia", True)):
            with self.subTest(timing=timing):
                model.invoke.side_effect = [extraction("z tyłu ekspresu", "po kawie"), extraction(timing=timing)]
                config = {"configurable": {"thread_id": timing}}
                first = graph.invoke({"message": "Po kawie zauważyłem kałużę za ekspresem."}, config)
                self.assertEqual(first["pending_question"], "fault_timing")
                final = graph.invoke({"message": timing}, config)
                self.assertIsNone(final["pending_question"])
                self.assertEqual(final["leak_location"], "z tyłu ekspresu")
                self.assertEqual(final["noticed_at"], "po kawie")
                self.assertEqual("obok przepływomierza" in final["answer"], expected_hose)
                prompt = json.loads(model.invoke.call_args.args[0][-1].content)
                self.assertEqual(prompt["ostatnie pytanie"], first["answer"])

    @patch("src.granaroma_demo.llm")
    def test_ambiguous_after_coffee_clarified_once(self, model):
        for timing, gives_hint in (("tuż po parzeniu", True), ("bez zmiany", False)):
            model.invoke.side_effect = [extraction("z tyłu ekspresu", "po kawie"),
                                        extraction(timing="nie wiadomo"), extraction(timing=timing)]
            graph = build_interview()
            config = {"configurable": {"thread_id": timing}}
            graph.invoke({"message": "Po kawie zauważyłem kałużę z tyłu."}, config)
            state = graph.invoke({"message": "Po zrobieniu kawy dopiero."}, config)
            self.assertIn("kałuża już tam była", state["answer"])
            reply = "Widziałem pierwsze krople tuż po parzeniu." if gives_hint else "Po kawie."
            final = graph.invoke({"message": reply}, config)
            self.assertIsNone(final["pending_question"])
            self.assertEqual("obok przepływomierza" in final["answer"], gives_hint)

    @patch("src.granaroma_demo.llm")
    def test_short_choice_uses_timing_from_question(self, model):
        for reply, gives_hint in (("pierwsze krople", True), ("kałuża już tam była", False)):
            with self.subTest(reply=reply):
                model.invoke.side_effect = [extraction("z tyłu ekspresu", "po kawie"),
                                            extraction(timing="użytkownik nie widział początku lub nie pamięta")]
                graph = build_interview()
                config = {"configurable": {"thread_id": reply}}
                graph.invoke({"message": "Po kawie zauważyłem kałużę z tyłu."}, config)
                clarification = graph.invoke({"message": "po zrobieniu kawy dopiero"}, config)
                self.assertIn("tuż po zakończeniu parzenia", clarification["answer"])
                final = graph.invoke({"message": reply}, config)
                self.assertIsNone(final["pending_question"])
                self.assertEqual("obok przepływomierza" in final["answer"], gives_hint)

    @patch("src.granaroma_demo.llm")
    def test_correction_replaces_location(self, model):
        model.invoke.side_effect = [extraction("z tyłu ekspresu", "po kawie"),
                                    extraction("inne miejsce", timing="pod koniec parzenia")]
        graph = build_interview()
        config = {"configurable": {"thread_id": "correction"}}
        graph.invoke({"message": "Kałuża z tyłu po kawie."}, config)
        final = graph.invoke({"message": "Jednak z przodu, zaczęło pod koniec parzenia."}, config)
        self.assertEqual(final["leak_location"], "inne miejsce")
        self.assertNotIn("obok przepływomierza", final["answer"])
