"""Interaktywne demo jednej rozmowy diagnostycznej z lokalną Ollamą."""

from src.agent import build_agent, format_observations


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
    main()
