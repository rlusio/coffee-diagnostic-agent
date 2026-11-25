"""Interaktywne demo jednej rozmowy diagnostycznej z lokalną Ollamą."""

import argparse
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from ollama import ResponseError

from src.agent import build_agent, format_observations, llm
from src.knowledge_base import normalize


LEAK_VALUES = ["brak informacji", "tak", "nie", "nie sprawdzono"]
FIELDS = {
    "Płukanie": LEAK_VALUES,
    "Gorąca woda": LEAK_VALUES,
    "Parzenie kawy": LEAK_VALUES,
    "Moment przy gorącej wodzie": ["brak informacji", "w trakcie nalewania", "po zakończeniu nalewania"],
}


ACTIVITY_WORDS = {
    "Płukanie": r"\bpluk\w*",
    "Gorąca woda": r"\bgorac\w*\s+wod\w*",
    "Parzenie kawy": r"\b(kaw\w*|parzeni\w*|espresso|cappuccino|latte)\b",
}
PHASE_WORDS = {
    "w trakcie nalewania": r"\b(w trakcie|podczas|przy) nalewania\b|\bgdy (?:jeszcze )?(?:plynie|leci) strumien",
    "po zakończeniu nalewania": r"\b(po nalaniu|po zakonczeniu nalewania|po odcieciu|na (?:samym )?koncu|gdy strumien ustaje)\b",
}


def read_observations(content, message):
    observations = json.loads(content)
    if not isinstance(observations, dict) or observations.keys() != FIELDS.keys():
        raise ValueError("Niekompletna odpowiedź modelu")
    for field, allowed in FIELDS.items():
        if observations[field] not in allowed:
            raise ValueError("Nieznana wartość obserwacji")
    text = normalize(message)
    for field, pattern in ACTIVITY_WORDS.items():
        if not re.search(pattern, text):
            observations[field] = "brak informacji"

    phase = observations["Moment przy gorącej wodzie"]
    observations["Moment przy gorącej wodzie"] = "brak informacji"
    if observations["Gorąca woda"] == "tak" and phase in PHASE_WORDS:
        # A timing mentioned for coffee or rinsing does not establish a hot-water phase.
        for part in re.split(r"[.!?;]|\ba przy\b", text):
            if re.search(ACTIVITY_WORDS["Parzenie kawy"], part) or re.search(ACTIVITY_WORDS["Płukanie"], part):
                continue
            if re.search(ACTIVITY_WORDS["Gorąca woda"], part) and re.search(PHASE_WORDS[phase], part):
                observations["Moment przy gorącej wodzie"] = phase
                break
    return observations


def run_observation_demo():
    properties = {}
    for field, allowed in FIELDS.items():
        properties[field] = {"type": "string", "enum": allowed}
    schema = {"type": "object", "properties": properties, "required": list(FIELDS)}
    instruction = SystemMessage(content=(
        "Zapisz fakty z odpowiedzi na pytanie o wyciek jako JSON. "
        "Dla każdej czynności: tak = cieknie, nie = użytkownik zaobserwował brak wycieku, "
        "nie sprawdzono = użytkownik mówi, że nie próbował. Pominięta czynność = brak informacji. "
        "Moment przy gorącej wodzie wpisz tylko gdy użytkownik wyraźnie określił fazę nalewania. "
        "Parzenie kawy jest osobną czynnością. Nie zgaduj."
    ))
    print("Eksperyment z Bielikiem. Każda odpowiedź jest osobną próbą. Enter kończy.")
    model = llm
    while True:
        question = "Kiedy cieknie?"
        print("\nAgent:", question)
        message = input("Ty: ").strip()
        if not message:
            return
        print("\nBielik odczytuje obserwacje...", flush=True)
        messages = [instruction, HumanMessage(content=f"Pytanie: {question}\nOdpowiedź: {message}")]
        try:
            try:
                response = model.invoke(messages, format=schema)
            except ResponseError as error:
                if model is not llm or "PTX was compiled with an unsupported toolchain" not in str(error):
                    raise
                print("Błąd zgodności CUDA. Ponawiam próbę na CPU.", flush=True)
                model = llm.model_copy(update={"num_gpu": 0})
                response = model.invoke(messages, format=schema)
        except (ConnectionError, ResponseError) as error:
            print(f"Nie udało się uzyskać odpowiedzi Bielika. Sprawdź lokalną Ollamę.\n{error}")
            return
        try:
            observations = read_observations(response.content, message)
        except ValueError:
            print("Bielik zwrócił niepoprawny format:", response.content)
            continue
        for field in FIELDS:
            print(f"{field}: {observations[field]}")


def show_diagnostics(state):
    model = state.get("model")
    if model:
        print(f"\nModel ekspresu: {model}")
    print("\nZapamiętane obserwacje:")
    observations = format_observations(state)
    for observation in observations:
        print(f"  {observation}")
    if not observations:
        print("  Brak rozpoznanych obserwacji.")

    candidates = state.get("candidates", [])
    if candidates:
        first = candidates[0]
        print(f"Pierwszy rekord: {first['id']} (wynik dopasowania: {first['score']})")
        for candidate in candidates[1:]:
            print(f"Kolejny rekord: {candidate['id']} (wynik dopasowania: {candidate['score']})")
    else:
        print("Brak pasujących rekordów.")


def main():
    agent = build_agent()
    config = {"configurable": {"thread_id": "local-demo"}}
    print("Demo diagnostyczne. Pusty Enter w rozmowie kończy program.")
    brand = input("Marka (opcjonalnie): ").strip()
    model = input("Model (opcjonalnie): ").strip()
    turn = {"brand": brand, "model": model}

    while True:
        message = input("\nTy: ").strip()
        if not message:
            break
        turn["message"] = message
        state = agent.invoke(turn, config=config)
        print("\nAgent:", state["answer"])
        show_diagnostics(state)
        turn = {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--obserwacje", action="store_true", help="pokaż interpretację opisu wycieku przez Bielika")
    modes.add_argument("--granaroma", action="store_true", help="wyciek w GranAromie: Bielik odczytuje fakty, graf wybiera pytania")
    args = parser.parse_args()
    if args.obserwacje:
        run_observation_demo()
    elif args.granaroma:
        from src.granaroma_demo import run_interview
        run_interview()
    else:
        main()
