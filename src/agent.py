"""Coffee Machine Diagnostic Agent.

LangGraph controls the diagnostic workflow.
Ollama generates responses using retrieved fault records.
"""
from __future__ import annotations
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from src.knowledge_base import find_faults
from langgraph.checkpoint.memory import InMemorySaver
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
    model="llama3.2",
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


def retrieve(state: DiagnosticState) -> dict:
    new_message = state.get("message", "")
    previous_description = state.get("description", "")

    description = (
        previous_description + " " + new_message
    ).strip()

    candidates = find_faults(
        description,
        state.get("brand", ""),
        state.get("model", "")
    )

    return {
        "description": description,
        "candidates": candidates
    }


def decide(state: DiagnosticState) -> str:
    candidates = state.get("candidates", [])

    if not candidates:
        return "clarify"

    best_score = candidates[0]["score"]

    if best_score < 2:
        return "weak_match"

    return "answer"



def clarify(state: DiagnosticState) -> dict:
    return {"answer": "Nie znalazłem pasującego wpisu. Podaj markę, model, komunikat błędu i dokładne objawy."}

def weak_match(state: DiagnosticState) -> dict:
    return {
        "answer": (
            "Znalazłem podobne przypadki, ale opis "
            "nie pozwala jeszcze zawęzić diagnozy. "
            "Podaj dokładny model ekspresu, moment "
            "występowania problemu oraz dodatkowe objawy."
        )
    }

def answer(state: DiagnosticState) -> dict:
    candidates = state.get("candidates", [])[:3]
    print("\n--- DEBUG: WYNIKI WYSZUKIWANIA ---")
    for candidate in candidates:
        print(
            f"ID: {candidate.get('id')} | "
            f"Score: {candidate.get('score')}"
        )
    print("---------------------------------\n")
    context = json.dumps(
        candidates,
        ensure_ascii=False,
        default=str
    )

    previous_answer = state.get("answer", "")

    messages = [
        SystemMessage(content="""
Jesteś asystentem diagnostycznym serwisu
automatycznych ekspresów do kawy.

Zasady:
1. Korzystaj wyłącznie z dostarczonych rekordów
   przy opisywaniu konstrukcji i typowych usterek.
2. Nie przedstawiaj hipotez jako pewnej diagnozy.
3. Rekordy niezweryfikowane oznaczaj jako hipotezy.
4. Uwzględniaj wcześniejsze informacje użytkownika.
5. Nie powtarzaj pytań, na które użytkownik
   już odpowiedział.
6. Zadawaj najwyżej jedno konkretne pytanie
   diagnostyczne na odpowiedź.
7. Nie podawaj instrukcji pracy przy elementach
   pod napięciem ani pod ciśnieniem.
8. Jeśli brakuje danych, przyznaj to.
9. Podaj ID rekordu, na którym opierasz sugestię.
10. Odpowiadaj po polsku, krótko i rzeczowo.
"""),
        HumanMessage(content=f"""
Urządzenie:
Marka: {state.get("brand", "nieznana")}
Model: {state.get("model", "nieznany")}

Informacje uzyskane dotychczas:
{state.get("description", "")}

Poprzednia odpowiedź:
{state.get("answer", "")}

Pasujące rekordy:
{context}

Zadanie:
1. Ustal, co użytkownik już powiedział.
2. Nie pytaj ponownie o znane informacje.
3. Wybierz jedno pytanie, które najlepiej
   pozwoli odróżnić możliwe przyczyny.
4. Jeśli nie masz wystarczających danych,
   nie sugeruj konkretnej awarii.
""")
    ]

    response = llm.invoke(messages)

    return {"answer": response.content}



def build_agent():
    builder = StateGraph(DiagnosticState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("clarify", clarify)
    builder.add_node("weak_match", weak_match)
    builder.add_node("answer", answer)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        decide,
        {
            "clarify": "clarify",
            "weak_match": "weak_match",
            "answer": "answer"
        }
    )
    builder.add_edge("clarify", END)
    builder.add_edge("weak_match", END)
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=InMemorySaver())
