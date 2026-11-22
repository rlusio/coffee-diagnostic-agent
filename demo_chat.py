from src.agent import build_agent

agent = build_agent()

config = {
    "configurable": {
        "thread_id": "melitta_test_001"
    }
}

result1 = agent.invoke(
    {
        "message": "Woda wycieka pod ekspresem.",
        "brand": "Melitta",
        "model": "Barista TS"
    },
    config=config
)

print("ODPOWIEDŹ 1:")
print(result1["answer"])

result2 = agent.invoke(
    {
        "message": "Dzieje się to tylko podczas parzenia kawy."
    },
    config=config
)

print("\nODPOWIEDŹ 2:")
print(result2["answer"])

print("\nZAPAMIĘTANY OPIS:")
print(result2["description"])
