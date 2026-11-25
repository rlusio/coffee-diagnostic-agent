"""Eksperyment: Bielik odczytuje obserwacje, Python prowadzi rozmowę."""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from ollama import ResponseError

from src.agent import DiagnosticState, llm
from src.knowledge_base import load_faults, normalize


class Interview(DiagnosticState, total=False):
    noticed_at: str
    onset_clarified: bool
    previous_reply: str
    breaker_tripped: str
    tray_cleaned: str
    brew_area_cleaned: str
    tray_excess: str


UNKNOWN = "nie wiadomo"
AFTER_COFFEE_QUESTION = "Czy tuż po zakończeniu parzenia widziałeś pierwsze krople, czy kałuża już tam była, kiedy spojrzałeś?"
AFTER_COFFEE_CHOICES = {
    "pierwsze krople": "tuż po parzeniu",
    "kałuża już tam była": "użytkownik nie widział początku lub nie pamięta",
}
CLEANING = ["ponad miesiąc temu", "w ostatnim miesiącu", UNKNOWN]
FIELDS = {
    "leak_location": ["z tyłu ekspresu", "z boku", "z boku bliżej tyłu", "z boku bliżej przodu",
                      "pod tacką", "w tacce", "przy zbiorniku na wodę", "inne miejsce", UNKNOWN],
    "tray_cleaned": CLEANING,
    "brew_area_cleaned": CLEANING,
    "tray_excess": ["tak", "nie", UNKNOWN],
    "breaker_tripped": ["tak", "nie"],
    "noticed_at": ["po kawie", "przy innej czynności", UNKNOWN],
    "fault_timing": ["podczas parzenia", "pod koniec parzenia", "tuż po parzeniu", "podczas płukania",
                     "inny moment", "użytkownik nie widział początku lub nie pamięta", UNKNOWN],
}
QUESTIONS = {
    "leak_location": "Gdzie zauważyłeś wodę?",
    "fault_timing": "Czy widziałeś, kiedy woda zaczęła wypływać? Jeśli zastałeś już kałużę, wystarczy „nie wiem”.",
}
LABELS = {"leak_location": "Miejsce", "noticed_at": "Zauważenie wody", "fault_timing": "Początek wycieku",
          "breaker_tripped": "Zadziałało zabezpieczenie elektryczne",
          "tray_cleaned": "Ręczne mycie tacki", "brew_area_cleaned": "Mycie zaparzacza i jego komory",
          "tray_excess": "Nadmierna ilość wody w tacce"}


def supported_location(proposed, message, state):
    text = normalize(message)
    if proposed == "przy zbiorniku na wodę" and re.search(r"zbiornik|pojemnik", text):
        return proposed
    if proposed in ("pod tacką", "w tacce") and re.search(r"tac[cky]", text):
        return proposed
    if proposed in ("pod tacką", "w tacce", "przy zbiorniku na wodę"):
        return "bez zmiany"
    side = re.search(r"\b(bok\w*|bocz\w*)", text)
    rear = re.search(r"\b(tyl|tylu|tylem|tyln\w*)\b|\bza (ekspresem|urzadzeniem)", text)
    front = re.search(r"\bprz[oó]d\w*|\bprzedni\w*|\bprzodu", text)
    if state.get("pending_question") == "side_position" and state.get("leak_location") == "z boku":
        side = True
    if side:
        if rear and not front:
            return "z boku bliżej tyłu"
        if front and not rear:
            return "z boku bliżej przodu"
        return "z boku"
    if proposed.startswith("z boku") or (proposed == "z tyłu ekspresu" and not rear):
        return "bez zmiany"
    return proposed


def extract_facts(state):
    message = state["message"]
    if state.get("answer") == AFTER_COFFEE_QUESTION:
        selected = message.strip().lower().strip(".!?")
        if selected in AFTER_COFFEE_CHOICES:
            return {"fault_timing": AFTER_COFFEE_CHOICES[selected], "previous_reply": message}
    properties = {}
    known = {}
    for field, values in FIELDS.items():
        properties[field] = {"type": "string", "enum": ["bez zmiany", *values]}
        if state.get(field):
            known[field] = state[field]
    schema = {"type": "object", "properties": properties, "required": list(FIELDS), "additionalProperties": False}
    instruction = (
        "Odczytaj wyłącznie nowe obserwacje lub sprostowania z ostatniej wypowiedzi. "
        "leak_location = gdzie widać wodę, noticed_at = kiedy użytkownik dostrzegł wodę, "
        "fault_timing = kiedy użytkownik widział POCZĄTEK wypływania. "
        "Kałuża po kawie bez podania miejsca: leak_location bez zmiany. Rozróżnij bok bliżej tyłu "
        "od samego tyłu. breaker_tripped oznacza zadziałanie bezpiecznika lub różnicówki; "
        "ręczne wyłączenie ekspresu to inna obserwacja. "
        "Rozróżnij wodę POD tacką i wodę W tacce. tray_excess = czy do tacki trafia nadmiernie dużo wody. "
        "tray_cleaned i brew_area_cleaned opisują ostatnie ręczne mycie tacki oraz zaparzacza z jego komorą. "
        "Automatyczne płukanie to osobna czynność. Uwzględnij, którego elementu dotyczy ostatnie pytanie. "
        "Dla pominiętych informacji wpisuj bez zmiany. Jeśli użytkownik wyraźnie mówi, że nie wie "
        "lub nie widział początku, dla fault_timing wybierz 'użytkownik nie widział początku lub nie pamięta'. "
        "Sama odpowiedź 'dopiero po kawie' nie mówi, czy widział pierwsze krople: wpisz bez zmiany. "
        "Jeśli widział pierwsze krople tuż po zakończeniu parzenia, wpisz tuż po parzeniu. Nie diagnozuj."
    )
    context = {"znane fakty": known, "poprzednia wypowiedź użytkownika": state.get("previous_reply", ""),
               "ostatnie pytanie": state.get("answer", ""), "wypowiedź": message}
    messages = [SystemMessage(content=instruction), HumanMessage(content=json.dumps(context, ensure_ascii=False))]
    cpu = state.get("use_cpu", False)
    model = llm
    if cpu:
        model = llm.model_copy(update={"num_gpu": 0})
    try:
        result = model.invoke(messages, format=schema)
    except ResponseError as error:
        if cpu or "PTX was compiled with an unsupported toolchain" not in str(error):
            raise
        print("Błąd zgodności CUDA. Ponawiam próbę na CPU.", flush=True)
        cpu = True
        result = llm.model_copy(update={"num_gpu": 0}).invoke(messages, format=schema)

    facts = json.loads(result.content)
    if not isinstance(facts, dict) or facts.keys() != FIELDS.keys():
        raise ValueError("Niepoprawna struktura obserwacji")
    updates = {"use_cpu": cpu, "previous_reply": message}
    for field, values in FIELDS.items():
        value = facts[field]
        if value != "bez zmiany" and value not in values:
            raise ValueError("Niepoprawna wartość pola " + field)
        if field == "leak_location":
            text = normalize(message)
            explicit_side = re.search(r"\bz boku\b", text)
            uncertain_side = re.search(r"\b(nie|chyba|moze|albo|lub)\b", text)
            if value not in (UNKNOWN, "bez zmiany") or (explicit_side and not uncertain_side):
                value = supported_location(value, message, state)
        if field == "breaker_tripped" and not re.search(r"bezpiecznik|roznicowk|korki|zabezpieczen|wylacznik nadpradow", normalize(message)):
            continue
        if field in ("tray_cleaned", "brew_area_cleaned") and state.get("pending_question") != field:
            text = normalize(message)
            item_pattern = r"tac[cky]" if field == "tray_cleaned" else r"zaparzacz|komor"
            if not re.search(r"my[lctj]|myci|czysz|czyst", text) or not re.search(item_pattern, text):
                continue
        if field == "tray_excess" and state.get("pending_question") != field:
            if not re.search(r"duz|spor|nadmiern|szybk|przelewa|wiecej|pelna|zwykle|normaln", normalize(message)):
                continue
        if value == "bez zmiany":
            continue
        if field == "fault_timing" and value == AFTER_COFFEE_CHOICES["kałuża już tam była"]:
            explicit_unknown = re.search(r"\bnie (wiem|pamietam|widzial\w*|zauwazyl\w*)\b", normalize(message))
            if not explicit_unknown and not state.get("onset_clarified"):
                continue
        # An uncertain extraction alone must not end the interview before the clarifying question.
        if value == UNKNOWN and state.get("pending_question") != field and not state.get(field):
            continue
        updates[field] = value
    return updates


def tray_or_tank_reply(state, location):
    record_ids = []
    advice = []
    if location == "pod tacką":
        for field, item in (("tray_cleaned", "tackę"), ("brew_area_cleaned", "zaparzacz i jego komorę")):
            if not state.get(field):
                return {"pending_question": field, "answer": f"Kiedy ostatnio ręcznie myłeś {item}?"}
            if state[field] == "ponad miesiąc temu":
                advice.append(f"Sprawdź i umyj {item} zgodnie z instrukcją użytkownika.")
        record_ids = ["saeco_granaroma_under_tray"]
    elif location == "w tacce":
        amount = state.get("tray_excess")
        if not amount:
            return {"pending_question": "tray_excess", "answer": "Czy do tacki trafia wyraźnie więcej wody niż zwykle?"}
        if amount != "tak":
            return {"pending_question": None, "answer": "Część wody z płukania trafia do tacki. Sama jej obecność nie wskazuje na usterkę."}
        record_ids = ["saeco_granaroma_hot_water_drain_leak", "saeco_granaroma_brew_unit_drain"]
    else:
        record_ids = ["saeco_granaroma_tank_crack"]
    causes = []
    for record in load_faults():
        if record["id"] in record_ids:
            causes.extend(record["possible_causes"])
            if location != "w tacce":
                advice.extend(record["safe_checks"])
    answer = "Możliwe przyczyny: " + "; ".join(causes) + "."
    if advice:
        answer += " " + " ".join(advice)
    return {"pending_question": None, "answer": answer}


def choose_step(state):
    location = state.get("leak_location")
    timing = state.get("fault_timing")
    if state.get("breaker_tripped") == "tak":
        return {"pending_question": None, "answer": "Przy wycieku i zadziałaniu zabezpieczenia pozostaw ekspres wyłączony. Nie uruchamiaj go ponownie. Przekaż urządzenie do serwisu i zgłoś oba objawy."}
    if not location:
        return {"pending_question": "leak_location", "answer": QUESTIONS["leak_location"]}
    if location in ("pod tacką", "w tacce", "przy zbiorniku na wodę"):
        return tray_or_tank_reply(state, location)
    if location == "z boku" and state.get("pending_question") != "side_position":
        return {"pending_question": "side_position", "answer": "Gdzie na boku było widać wodę: bliżej tyłu czy przodu ekspresu?"}
    side_records = {"z boku bliżej tyłu": "saeco_granaroma_heater_fitting_leak",
                    "z boku bliżej przodu": "saeco_granaroma_solenoid_leak"}
    if location in side_records:
        for record in load_faults():
            if record["id"] == side_records[location]:
                causes = "; ".join(record["possible_causes"])
                return {"pending_question": None, "answer": f"Wodę widać {location}. Tropy serwisowe: {causes}. Źródło wymaga potwierdzenia w serwisie."}
    if location == "z tyłu ekspresu" and timing in (None, UNKNOWN) and not state.get("onset_clarified"):
        question = QUESTIONS["fault_timing"]
        if state.get("pending_question") == "fault_timing":
            question = "Czy widziałeś pierwsze krople, czy kałuża już tam była, kiedy spojrzałeś?"
            if state.get("noticed_at") == "po kawie":
                question = AFTER_COFFEE_QUESTION
            return {"pending_question": "fault_timing", "onset_clarified": True,
                    "answer": question}
        if state.get("noticed_at") == "po kawie":
            question = "Po kawie zobaczyłeś wodę. Czy pamiętasz, kiedy pojawiły się pierwsze krople?"
        return {"pending_question": "fault_timing", "answer": question}
    if location == "z tyłu ekspresu" and timing in ("pod koniec parzenia", "tuż po parzeniu", "podczas płukania"):
        for record in load_faults():
            if record["id"] == "saeco_granaroma_hose_solenoid_leak":
                cause = ". ".join(record["possible_causes"]).rstrip(".")
                return {"pending_question": None, "answer": f"Wyciek z tyłu {timing}. Trop serwisowy: {cause}. Źródło wymaga potwierdzenia w serwisie."}
    return {"pending_question": None, "answer": "Te obserwacje nie wystarczają do zawężenia źródła wycieku."}


def build_interview():
    graph = StateGraph(Interview)
    graph.add_node("extract_facts", extract_facts)
    graph.add_node("choose_step", choose_step)
    graph.add_edge(START, "extract_facts")
    graph.add_edge("extract_facts", "choose_step")
    graph.add_edge("choose_step", END)
    return graph.compile(checkpointer=InMemorySaver())


def run_interview():
    agent = build_interview()
    config = {"configurable": {"thread_id": "granaroma-demo"}}
    print("Saeco GranAroma — opisz wyciek własnymi słowami. Enter kończy rozmowę.")
    while True:
        message = input("\nTy: ").strip()
        if not message:
            return
        try:
            state = agent.invoke({"message": message}, config)
        except (ResponseError, ConnectionError) as error:
            print("Błąd lokalnej Ollamy:", error)
            return
        except ValueError as error:
            print("Nie udało się zapisać obserwacji:", error)
            continue
        for field, label in LABELS.items():
            print(f"{label}: {state.get(field, 'brak informacji')}")
        print("Agent:", state["answer"])
        if not state["pending_question"]:
            return
