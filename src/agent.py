"""Coffee Machine Diagnostic Agent.

LangGraph controls the diagnostic workflow.
Ollama generates responses using retrieved fault records.
"""
from __future__ import annotations
from typing import TypedDict
import re

from langgraph.graph import END, START, StateGraph
from src.knowledge_base import find_faults, load_faults, normalize, normalize_identifier, split_catalog_device
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from ollama import ResponseError
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
    model="SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0",
    temperature=0.2
)

class DiagnosticState(TypedDict, total=False): #total false - nie wymagamy wszystkich pol poki co
    description: str #opis usterki podany przez uzytkownika
    brand: str 
    model: str 
    candidates: list[dict] #lista mozliwych usterek znaleziona w bazie
    route: str #Pole przewidziane na informację o wybranej ścieżce, obecnie nieużywane
    answer: str
    message: str
    use_cpu: bool
    leak_location: str | None
    fault_timing: str | None
    hot_water_flow: bool | None
    rear_leak_during_hot_water: bool | None
    hot_water_leak_phase: str
    grinder_audible: bool | None
    no_beans_warning: bool | None
    milk_delivery: str | None
    outlet_works: bool | None
    machine_responds: bool | None
    pending_question: str | None
    asked_questions: list[str]
    question_answers: dict[str, str]
    shown_record_ids: list[str]


LOCATIONS = {
    "z tyłu ekspresu": r"z tylu(?: ekspresu| urzadzenia)?\b",
    "pod ekspresem": r"pod (?:ekspresem|urzadzeniem)",
    "w tacce": r"w (?:tacce|tacy)",
    "w pojemniku na fusy": r"w pojemniku na fusy",
    "przy dyszy": r"przy dyszy",
}
TIMINGS = {
    "po zakończeniu parzenia": r"po zakonczeniu parzenia|po (?:zaparzeniu|parzeniu|zrobieniu) kawy|po kawie|(?:na|pod) koniec parzenia|na koncu parzenia",
    "po spienianiu mleka": r"po (?:zakonczeniu )?(?:spieniania|spienianiu|spienieniu)(?: mleka)?|po mleku",
    "podczas czyszczenia układu mlecznego": r"(?:podczas |przy |w trakcie )(?:automatycznego )?czyszczeni\w* ukladu mlecznego",
    "po odcięciu strumienia gorącej wody": r"(?:po|przy) odcieci\w* (?:strumienia )?goracej wody|na (?:koniec|koncu) nalewania goracej wody",
    "w trakcie nalewania gorącej wody": r"(?:podczas |przy |w trakcie )nalewania goracej wody",
    "podczas parzenia": r"(?:podczas |przy |w trakcie )(?:parzeni\w*|kaw\w*|robieni\w* kaw\w*)",
    "podczas płukania": r"(?:podczas |przy |w trakcie )plukani\w*",
    "podczas spieniania mleka": r"(?:podczas |przy |w trakcie )spieniani\w*(?: mleka)?",
    "podczas poboru gorącej wody": r"(?:podczas |przy |w trakcie )(?:poboru |pobierania )?goracej wody",
    "w spoczynku": r"w spoczynku|na postoju",
}
QUESTIONS = {
    "leak_location": "Gdzie pojawia się woda?",
    "fault_timing": "Kiedy występuje problem?",
    "hot_water_flow": "Czy gorąca woda leci prawidłowo?",
    "rear_leak_during_hot_water": "Czy przy pobieraniu gorącej wody też widzisz wyciek z tyłu ekspresu?",
    "hot_water_leak_phase": "Czy cieknie już w trakcie nalewania gorącej wody, czy dopiero na końcu, gdy strumień ustaje?",
    "grinder_audible": "Czy przy próbie przygotowania kawy słychać młynek?",
    "no_beans_warning": "Czy urządzenie wyświetla komunikat o braku ziaren?",
    "milk_delivery": "Czy mleko nie jest pobierane wcale, leci słabo, czy tylko nie tworzy piany?",
    "outlet_works": "Czy gniazdko zasila inne urządzenie?",
    "machine_responds": "Czy ekspres reaguje w jakikolwiek sposób (dioda, dźwięk, ekran)?",
    "model": "Jaki to model ekspresu?",
    "symptoms": "Jaki dokładnie objaw obserwujesz?",
}


def read_grinder_facts(message: str, pending: str | None) -> dict:
    facts = {}
    if pending in ("grinder_audible", "no_beans_warning"):
        if message == "tak":
            facts[pending] = True
        elif message == "nie":
            facts[pending] = False

    if "nie wiem" in message or "nie jestem pew" in message:
        return facts

    quiet_phrases = ("mlynka nie slychac", "nie slychac mlynka", "nie slysze mlynka",
                     "mlynek milczy", "mlynek nie wydaje zadnego dzwieku")
    audible_phrases = ("slychac mlynek", "slysze mlynek", "mlynek slychac",
                      "mlynek piszczy", "mlynek buczy")
    for phrase in quiet_phrases:
        if phrase in message:
            facts["grinder_audible"] = False
            break
    if "grinder_audible" not in facts:
        for phrase in audible_phrases:
            if phrase in message:
                facts["grinder_audible"] = True
                break

    absent_warnings = ("nie ma komunikatu o braku ziaren", "brak komunikatu o braku ziaren",
                       "nie wyswietla komunikatu o braku ziaren", "nie pokazuje komunikatu o braku ziaren",
                       "komunikat o braku ziaren sie nie pojawia", "nie pojawia sie komunikat o braku ziaren")
    for phrase in absent_warnings:
        if phrase in message:
            facts["no_beans_warning"] = False
            return facts

    visible_warnings = ("komunikat o braku ziaren", "wyswietla brak ziaren", "pokazuje brak ziaren",
                        "na ekranie brak ziaren")
    for phrase in visible_warnings:
        if phrase in message:
            facts["no_beans_warning"] = True
            break
    return facts


def read_milk_delivery(message: str, pending: str | None) -> str | None:
    if "nie wiem" in message or "nie jestem pew" in message:
        return None

    if pending == "milk_delivery":
        if message in ("wcale", "nie leci", "wcale nie leci"):
            return "brak poboru mleka"
        if message in ("slabo", "leci slabo"):
            return "słaby przepływ mleka"
        if message in ("tylko brak piany", "tylko nie tworzy piany"):
            return "mleko płynie bez piany"

    if "mleko nie leci" in message or "nie pobiera mleka" in message:
        return "brak poboru mleka"
    if "mleko leci slabo" in message or "mleko slabo leci" in message:
        return "słaby przepływ mleka"
    if "mleko leci" in message or "mleko plynie" in message:
        for phrase in ("bez pian", "nie tworzy piany", "brak piany", "nie ma piany"):
            if phrase in message:
                return "mleko płynie bez piany"
    return None


def read_power_facts(message: str, pending: str | None) -> dict:
    facts = {}
    if "nie wiem" in message or "nie jestem pew" in message:
        return facts

    if pending in ("outlet_works", "machine_responds"):
        if message == "tak":
            facts[pending] = True
        elif message == "nie":
            facts[pending] = False

    if "gniazdko nie dziala" in message or "gniazdko nie zasila" in message:
        facts["outlet_works"] = False
    elif "gniazdko dziala" in message or "gniazdko zasila inne urzadzenie" in message:
        facts["outlet_works"] = True

    # A dark screen alone does not tell us whether the machine makes a sound.
    if "ekspres nie reaguje" in message or "brak reakcji po wlaczeniu" in message:
        facts["machine_responds"] = False
    elif "ekspres reaguje" in message or "dioda miga" in message or "ekran sie zapala" in message:
        facts["machine_responds"] = True
    return facts


def update_facts(state: DiagnosticState) -> dict:
    # description is also supported as the initial input of the original CLI.
    text = state.get("message")
    if not text:
        text = state.get("description", "")

    text = normalize(text)
    message = text.strip(" \t\r\n.!?,;:/\\")
    pending = state.get("pending_question")
    updates = read_grinder_facts(message, pending)
    sound_replies = {
        "slychac": True,
        "jednak slychac": True,
        "nie slychac": False,
        "jednak nie slychac": False,
    }
    short_sound_reply = message in sound_replies
    if short_sound_reply:
        if pending == "grinder_audible" or state.get("grinder_audible") is not None:
            updates["grinder_audible"] = sound_replies[message]
    power_facts = read_power_facts(message, pending)
    updates.update(power_facts)
    milk_delivery = read_milk_delivery(message, pending)
    if milk_delivery is not None:
        updates["milk_delivery"] = milk_delivery
    for field, patterns in (("leak_location", LOCATIONS), ("fault_timing", TIMINGS)):
        values = []
        for value, pattern in patterns.items():
            for match in re.finditer(pattern, message):
                preceding_text = message[:match.start()]
                negation = re.search(r"\bnie\s*$", preceding_text)
                if not negation:
                    values.append(value)
                    break
        if values:
            if field == "fault_timing" and re.search(r"\b(?:plukania|parzenia|kawy) (?:oraz|i) goracej wody\b", message):
                if "podczas poboru gorącej wody" not in values:
                    values.append("podczas poboru gorącej wody")
            updates[field] = ", ".join(values)

    flow = re.search(r"\bgoraca woda (nie dziala|nie leci|nie plynie|leci slabo|dziala|leci prawidlowo|plynie normalnie)\b", message)
    if flow:
        flow_description = flow.group(1)
        updates["hot_water_flow"] = flow_description in {"dziala", "leci prawidlowo", "plynie normalnie"}
    elif pending == "hot_water_flow":
        if message in {"tak", "dziala", "prawidlowo"}:
            updates["hot_water_flow"] = True
        elif message in {"nie", "nie dziala", "nie leci", "slabo"}:
            updates["hot_water_flow"] = False
    if pending == "rear_leak_during_hot_water":
        if message in ("tak", "tez cieknie", "tak, tez cieknie"):
            updates[pending] = True
        elif message in ("nie", "nie cieknie", "nie, jest sucho"):
            updates[pending] = False
    if pending in ("rear_leak_during_hot_water", "hot_water_leak_phase"):
        reply_timing = updates.get("fault_timing")
        if reply_timing == "w trakcie nalewania gorącej wody":
            updates["hot_water_leak_phase"] = "w trakcie nalewania"
        elif reply_timing == "po odcięciu strumienia gorącej wody":
            updates["hot_water_leak_phase"] = "po odcięciu strumienia"
        if "hot_water_leak_phase" in updates and state.get("fault_timing"):
            del updates["fault_timing"]
        if message in ("na koncu", "dopiero na koncu", "na sam koniec", "na samym koncu", "po odcieciu", "po odcieciu strumienia"):
            updates["hot_water_leak_phase"] = "po odcięciu strumienia"
        elif message in ("w trakcie", "podczas nalewania", "w trakcie nalewania", "caly czas"):
            updates["hot_water_leak_phase"] = "w trakcie nalewania"
        if "hot_water_leak_phase" in updates:
            updates["rear_leak_during_hot_water"] = True
    if not pending or not message:
        return updates

    unknown = message in ("nie wiem", "nie pamietam", "nie znam modelu", "trudno powiedziec")
    if short_sound_reply and pending == "model":
        return updates
    if pending == "model" and not unknown and not updates and message not in ("tak", "nie"):
        model = state.get("message", "")
        updates["model"] = model.strip()

    # Questions from the knowledge base may have a free-text answer.
    free_text = pending == "symptoms" or pending not in QUESTIONS
    if unknown or pending in updates or (free_text and not updates):
        answers = state.get("question_answers", {}).copy()
        answers[pending] = state.get("message", "").strip()
        updates["question_answers"] = answers
    return updates


def retrieve(state: DiagnosticState) -> dict:
    new_message = state.get("message", "")
    previous_description = state.get("description", "")

    description = previous_description + " " + new_message
    description = description.strip()

    brand, model = split_catalog_device(state.get("brand", ""), state.get("model", ""))
    candidates = find_faults(description, brand, model)

    return {
        "brand": brand,
        "model": model,
        "description": description,
        "candidates": candidates
    }


MIN_ANSWER_SCORE = 2


def decide(state: DiagnosticState) -> str:
    candidates = state.get("candidates", [])

    if state.get("leak_location") == "z tyłu ekspresu" and state.get("fault_timing"):
        brand = normalize_identifier(state.get("brand", ""))
        model = normalize_identifier(state.get("model", ""))
        if brand == "saeco" and model == "granaroma":
            if not candidates or candidates[0]["score"] < MIN_ANSWER_SCORE:
                return "granaroma_rear_leak"
            if candidates[0]["id"] in ("leak_external_generic", "saeco_granaroma_hose_solenoid_leak"):
                return "granaroma_rear_leak"

    if not candidates:
        return "clarify"

    best_score = candidates[0]["score"]

    if best_score < MIN_ANSWER_SCORE:
        return "weak_match"

    return "answer"



def question_topic(question: str) -> str:
    text = normalize(question)
    if text.startswith("czy gniazdko zasila"):
        return "outlet_works"
    if text.startswith("czy ekspres reaguje"):
        return "machine_responds"
    if text.startswith("czy mleko nie jest pobierane"):
        return "milk_delivery"
    if "slychac mlynek" in text:
        return "grinder_audible"
    if "komunikat o braku ziaren" in text:
        return "no_beans_warning"
    if text.startswith(("gdzie pojawia", "w ktorym miejscu")):
        return "leak_location"
    if text.startswith(("kiedy ", "czy woda pojawia sie przede wszystkim podczas",
                        "czy problem wystepuje podczas przygotowywania")):
        return "fault_timing"
    if "gorac" in text and "para" not in text:
        for phrase in ("podaje", "podawanie", "dziala prawidlowo"):
            if phrase in text:
                return "hot_water_flow"
    return question


def finish_reply(state: DiagnosticState, body: str) -> dict:
    asked = state.get("asked_questions", [])
    candidates = state.get("candidates", [])
    options = []
    pending = state.get("pending_question")
    if pending:
        options.append(pending)
    answers = state.get("question_answers", {})
    best_candidate = {}
    if candidates:
        best_candidate = candidates[0]

    if state.get("grinder_audible") is not None or state.get("no_beans_warning") is not None:
        options.extend(("no_beans_warning", "grinder_audible"))
    if state.get("leak_location") or best_candidate.get("category") == "water_leak":
        options.extend(("leak_location", "fault_timing", "hot_water_flow"))
    for question in best_candidate.get("diagnostic_questions", []):
        topic = question_topic(question)
        options.append(topic)
    options.append("model")
    if best_candidate.get("score", 0) < MIN_ANSWER_SCORE:
        options.append("symptoms")
    for topic in options:
        if topic in answers:
            continue
        known_value = state.get(topic)
        if known_value is not None and known_value != "":
            continue

        question = QUESTIONS.get(topic, topic)
        updated_questions = asked.copy()
        if topic not in asked:
            updated_questions.append(topic)
        answer_text = question
        if body:
            answer_text = f"{body}\n{question}"
        return {
            "answer": answer_text,
            "pending_question": topic,
            "asked_questions": updated_questions,
        }
    if not body:
        body = "Na podstawie dotychczasowych informacji nie mogę dokładniej ustalić przyczyny."
    return {"answer": body, "pending_question": None}


def clarify(state: DiagnosticState) -> dict:
    return finish_reply(state, "Nie znalazłem pasującego wpisu.")

def weak_match(state: DiagnosticState) -> dict:
    return finish_reply(state, "Znalazłem podobne przypadki, ale opis nie pozwala jeszcze zawęzić diagnozy.")

def format_observations(state: DiagnosticState) -> list[str]:
    labels = {
        "leak_location": "Gdzie widać wodę",
        "fault_timing": "Moment usterki",
        "hot_water_flow": "Gorąca woda działa prawidłowo",
        "rear_leak_during_hot_water": "Wyciek z tyłu przy pobieraniu gorącej wody",
        "hot_water_leak_phase": "Moment wycieku przy gorącej wodzie",
        "grinder_audible": "Słychać młynek",
        "no_beans_warning": "Komunikat o braku ziaren",
        "milk_delivery": "Podawanie mleka",
        "outlet_works": "Gniazdko działa",
        "machine_responds": "Ekspres reaguje",
    }
    observations = []
    for field, label in labels.items():
        value = state.get(field)
        if value is None:
            continue
        if value is True:
            value = "tak"
        elif value is False:
            value = "nie"
        observations.append(f"{label}: {value}")
    return observations


def reply_with_record(state: DiagnosticState, record: dict) -> dict:
    shown = state.get("shown_record_ids", [])
    record_id = record["id"]
    if record_id in shown:
        return finish_reply(state, "")

    body = "Dostępne informacje nie pozwalają potwierdzić przyczyny."
    causes = "; ".join(record["possible_causes"])
    if causes:
        body = f"Możliwe przyczyny ({record['id']}): {causes}."
    for check in record.get("safe_checks", []):
        body += f"\n{check}"
    reply = finish_reply(state, body)
    reply["shown_record_ids"] = shown + [record_id]
    return reply


def answer(state: DiagnosticState) -> dict:
    candidates = state.get("candidates", [])
    candidates = candidates[:3]
    location = state.get("leak_location")
    asked = state.get("asked_questions", [])
    if state.get("pending_question") == "leak_location" and not location:
        if "leak_location" not in state.get("question_answers", {}):
            return finish_reply(state, "")
    if candidates[0]["id"] == "leak_external_generic" and not location and "leak_location" not in asked:
        return finish_reply(state, "")
    if location and not state.get("fault_timing") and "fault_timing" not in asked:
        return finish_reply(state, f"Woda pojawia się {location}.")
    timing = state.get("fault_timing")
    if location and timing:
        observation = f"Wodę widać {location} {timing}."
        if "hot_water_flow" not in asked and state.get("hot_water_flow") is None:
            return finish_reply(state, observation)
        if state.get("pending_question") == "hot_water_flow":
            return finish_reply(state, observation + " Samo miejsce pojawienia się wody nie wskazuje, skąd ona wypływa.")

    eligible_records = {}
    evidence = []
    for candidate in candidates:
        if candidate["score"] < MIN_ANSWER_SCORE:
            continue
        eligible_records[candidate["id"]] = candidate
        evidence.append(f"Rekord: {candidate['id']}")
        if not candidate.get("verified"):
            evidence.append("Rekord niezweryfikowany.")
        causes = "; ".join(candidate.get("possible_causes", []))
        evidence.append(f"Możliwe przyczyny: {causes}")
        evidence.append(candidate.get("notes", ""))
    if len(eligible_records) == 1:
        best_record = candidates[0]
        return reply_with_record(state, best_record)

    context = "\n".join(evidence)
    facts = "\n".join(format_observations(state))

    messages = [
        SystemMessage(content="""
Jesteś asystentem diagnostycznym serwisu
automatycznych ekspresów do kawy.

Wybierz rekord, którego hipotezy najlepiej pasują do obserwacji użytkownika.
Zwróć wyłącznie jego identyfikator. Jeśli żaden nie pasuje, odpowiedz BRAK.
"""),
        HumanMessage(content=f"""
Urządzenie:
Marka: {state.get("brand", "nieznana")}
Model: {state.get("model", "nieznany")}

Informacje uzyskane dotychczas:
{state.get("description", "")}

Aktualne obserwacje (uwzględnij sprostowania użytkownika):
{facts}

Pasujące rekordy:
{context}

""")
    ]

    use_cpu = state.get("use_cpu", False)
    model = llm
    if use_cpu:
        model = llm.model_copy(update={"num_gpu": 0})
    try:
        response = model.invoke(messages)
    except ResponseError as error:
        if use_cpu or "PTX was compiled with an unsupported toolchain" not in str(error):
            raise
        print("Ollama zgłosiła błąd zgodności CUDA. Ponawiam tę turę na CPU.")
        model = llm.model_copy(update={"num_gpu": 0})
        response = model.invoke(messages)
        use_cpu = True

    record_id = response.content.strip()
    record = eligible_records.get(record_id)
    if record is not None:
        reply = reply_with_record(state, record)
    else:
        reply = finish_reply(state, "Dostępne informacje nie pozwalają potwierdzić przyczyny.")
    reply["use_cpu"] = use_cpu
    return reply



def granaroma_rear_leak_reply(state: DiagnosticState) -> dict:
    timing = state["fault_timing"]
    topic = "rear_leak_during_hot_water"
    observed = state.get(topic)
    phase = state.get("hot_water_leak_phase")
    if "w trakcie nalewania gorącej wody" in timing:
        phase = "w trakcie nalewania"
    elif "po odcięciu strumienia gorącej wody" in timing:
        phase = "po odcięciu strumienia"
    if phase or "podczas poboru gorącej wody" in timing:
        observed = True
    summary = f"Wodę widać z tyłu ekspresu {timing}."
    if observed is True and not phase:
        topic = "hot_water_leak_phase"
    if (observed is None or (observed and not phase)) and topic not in state.get("question_answers", {}):
        asked = state.get("asked_questions", []).copy()
        if topic not in asked:
            asked.append(topic)
        return {"answer": summary + "\n" + QUESTIONS[topic],
                "pending_question": topic, "asked_questions": asked}

    if observed is False:
        summary += " Przy pobieraniu gorącej wody z tyłu jest sucho."
    if phase == "po odcięciu strumienia" and "po odcięciu strumienia" not in timing:
        summary += " Przy gorącej wodzie wyciek widać dopiero po odcięciu strumienia."
    if phase == "w trakcie nalewania":
        summary += " Wyciek przy ciągłym nalewaniu nie odpowiada zwykłemu zrzutowi po odcięciu strumienia. Serwis może sprawdzić, czy zawór się domyka i czy osad utrudnia jego pracę."
    discharge_timings = ("podczas płukania", "po zakończeniu parzenia", "po spienianiu mleka",
                         "podczas czyszczenia układu mlecznego")
    matches_discharge = any(moment in timing for moment in discharge_timings)
    if phase != "po odcięciu strumienia" and not matches_discharge:
        return {"answer": summary + " Źródło wycieku wymaga ustalenia w serwisie.", "pending_question": None}
    for record in load_faults():
        if record["id"] == "saeco_granaroma_hose_solenoid_leak" and record.get("evidence_status") == "observed":
            for cause in record["possible_causes"]:
                summary += "\nTrop z przypadku serwisowego tej serii: " + cause.rstrip(".") + "."
            break
    summary += "\nŻeby potwierdzić źródło w tym egzemplarzu, serwis musi ustalić, z którego połączenia wypływa woda."
    return {"answer": summary, "pending_question": None}


def build_agent():
    builder = StateGraph(DiagnosticState)
    builder.add_node("update_facts", update_facts)
    builder.add_node("retrieve", retrieve)
    builder.add_node("clarify", clarify)
    builder.add_node("weak_match", weak_match)
    builder.add_node("answer", answer)
    builder.add_node("granaroma_rear_leak", granaroma_rear_leak_reply)
    builder.add_edge(START, "update_facts")
    builder.add_edge("update_facts", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        decide,
        {
            "clarify": "clarify",
            "weak_match": "weak_match",
            "granaroma_rear_leak": "granaroma_rear_leak",
            "answer": "answer"
        }
    )
    builder.add_edge("clarify", END)
    builder.add_edge("weak_match", END)
    builder.add_edge("answer", END)
    builder.add_edge("granaroma_rear_leak", END)
    return builder.compile(checkpointer=InMemorySaver())
