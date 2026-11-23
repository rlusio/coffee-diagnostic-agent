"""Local command-line demonstration: fault retrieval + explicit user review submissions."""
from src.agent import build_agent, format_observations
from uuid import uuid4
from src.review_store import get_public_status, log_search, submit_case


def yes(value: str) -> bool:
    return value.strip().casefold() in {"t", "tak", "y", "yes"}


def offer_submission(brand: str, model: str, symptoms: str) -> None:
    choice = input("Zgłosić ten przypadek do późniejszej weryfikacji? [t/N]: ")
    if not yes(choice):
        return
    if len(symptoms) > 1200:
        symptoms = input("Opis rozmowy przekracza limit zgłoszenia. Podsumuj objawy (do 1200 znaków): ").strip()
    if not brand or not model:
        print("Dla nowego zgłoszenia potrzebna jest marka i model/seria.")
        brand = brand or input("Marka: ").strip()
        model = model or input("Model/seria: ").strip()
    reported_cause = input("Jaka jest Twoja hipoteza lub ustalona przyczyna? (opcjonalnie): ").strip()
    certainty = "unknown"
    if reported_cause:
        certainty = "confirmed_by_reporter" if yes(input("Czy ustaliłeś tę przyczynę podczas naprawy? [t/N]: ")) else "suspected"
    email = ""
    consent = False
    if yes(input("Chcesz podać email do jednorazowych informacji o zgłoszeniu? [t/N]: ")):
        email = input("Email (zapis lokalny w SQLite): ").strip()
        consent = yes(input("Zgadzasz się na przechowywanie adresu w celu odpowiedzi o zgłoszeniu? [t/N]: "))
        if not consent:
            email = ""
    try:
        case_id, created = submit_case(brand=brand, model=model, symptoms=symptoms,
                                       reported_cause=reported_cause, cause_certainty=certainty,
                                       email=email, consent_to_email=consent)
        print(f"Numer zgłoszenia: {case_id}. Status: pending. "
              + ("Zapisano do weryfikacji." if created else "Taki przypadek jest już w kolejce."))
        print("Uwaga: wysyłka e-mail jest jeszcze NIEAKTYWNA; status sprawdzisz tym numerem.")
    except ValueError as exc:
        print(f"Nie zapisano zgłoszenia: {exc}")


def run_case(agent, description, brand, model):
    config = {"configurable": {"thread_id": str(uuid4())}}
    turn = {"message": description, "brand": brand, "model": model}

    while True:
        state = agent.invoke(turn, config=config)
        print("\nAgent:", state["answer"])
        if not state.get("pending_question"):
            break
        message = input("Ty (Enter kończy ten przypadek): ").strip()
        if not message:
            break
        turn = {"message": message}

    brand = state.get("brand", "")
    model = state.get("model", "")
    observations = format_observations(state)
    print("\nPodsumowanie przypadku")
    print("Zgłoszony objaw:", description)
    if brand or model:
        print("Ekspres:", brand, model)
    for observation in observations:
        print(observation)
    log_search(found_count=len(state.get("candidates", [])), brand=brand, model=model)
    summary_lines = [state["description"]]
    summary_lines.extend(observations)
    symptoms = "\n".join(summary_lines)
    offer_submission(brand, model, symptoms)


def main():
    agent = build_agent()
    print("Coffee Diagnostic Agent — lokalna demonstracja, nie zastępuje dokumentacji producenta")
    print("Aby sprawdzić status zgłoszenia, wpisz: status. Pusty Enter = koniec.")
    while True:
        description = input("\nObjawy: ").strip()
        if not description:
            break
        if description.lower() == "status":
            case_id = input("Numer zgłoszenia: ").strip()
            case = get_public_status(case_id)
            if case:
                print(f"Status: {case['status']}. Komentarz: {case['reviewer_note'] or 'brak'}")
            else:
                print("Nie znaleziono zgłoszenia.")
            continue
        brand = input("Marka (opcjonalnie): ").strip()
        model = input("Model lub seria (opcjonalnie): ").strip()
        run_case(agent, description, brand, model)


if __name__ == "__main__":
    main()
