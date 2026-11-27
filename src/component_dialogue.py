"""Observation-based diagnosis of water delivery and grinding problems."""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from ollama import ResponseError

from src.knowledge_base import load_faults, normalize, normalize_identifier, split_catalog_device


# The same question describes a field to Bielik and to the user.
QUESTIONS = {
    "pump_audible": "Czy podczas próby poboru wody słychać pompę?",
    "water_flow": "Jak wypływa woda z wylewki: normalnie, słabo, kroplami, nieregularnie czy wcale?",
    "cycle_error": "Czy ekspres przerywa cykl lub zgłasza problem z napełnieniem?",
    "pump_loud": "Czy pompa pracuje wyraźnie głośniej niż zwykle?",
    "air_trigger": "Czy problem zaczął się po wymianie filtra, opróżnieniu zbiornika lub długim postoju?",
    "priming_helped": "Czy po odpowietrzeniu zgodnie z instrukcją przepływ wrócił do normy?",
    "hot_water_affected": "Czy problem występuje również przy pobieraniu gorącej wody?",
    "water_to_tray": "Czy podczas próby woda trafia do tacki?",
    "tank_unchanged": "Czy poziom wody w zbiorniku pozostaje bez zmian?",
    "recent_descaling": "Czy problem pojawił się po odkamienianiu?",
    "water_elsewhere": "Czy podczas poboru wody widać wyciek poza tacką?",
    "grinder_audible": "Czy podczas przygotowania kawy słychać młynek?",
    "grinder_abnormal": "Czy młynek wydaje nietypowy dźwięk lub próbuje ruszyć i się zatrzymuje?",
    "loose_grounds": "Czy dużo suchej, zmielonej kawy rozsypuje się wewnątrz ekspresu?",
    "chute_blocked": "Czy zmielona kawa zalega w kanale prowadzącym do zaparzacza?",
    "grounds_delivered": "Czy zmielona kawa dociera do zaparzacza?",
    "removable_group": "Czy w tym ekspresie można wyjąć zaparzacz zgodnie z instrukcją?",
    "foreign_body": "Czy do młynka dostał się twardy przedmiot?",
    "wet_grinder": "Czy do młynka dostała się woda?",
    "recent_cleaning": "Czy problem z młynkiem pojawił się po czyszczeniu?",
    "wet_electrics": "Czy woda mogła zalać elementy elektryczne?",
}
WATER = ["pump_audible", "water_flow", "cycle_error", "pump_loud", "air_trigger",
         "hot_water_affected", "water_to_tray", "tank_unchanged", "priming_helped",
         "recent_descaling", "water_elsewhere"]
GRINDER = ["grinder_audible", "grinder_abnormal", "loose_grounds", "chute_blocked",
           "grounds_delivered", "foreign_body", "wet_grinder", "removable_group", "recent_cleaning"]
FLOW = ["normalnie", "słabo", "kroplami", "nieregularnie", "wcale"]
UNKNOWN = "nie wiadomo"


def device(state):
    brand, model = split_catalog_device(state.get("brand", ""), state.get("model", ""))
    return normalize_identifier(brand), normalize_identifier(model)


def domain_for(state):
    if state.get("component_domain"):
        return state["component_domain"]
    if state.get("pending_question"):
        return None
    text = normalize(state.get("message") or state.get("description", ""))
    brand, model = device(state)
    if re.search(r"wyciek|cieknie|przecieka|woda z tylu|woda pod ekspresem", text) and not re.search(r"pomp|nie pobiera wody", text):
        return None
    if brand in ("saeco", "philips") and re.search(r"mlyn|miel|ziarn|wysyp|kanal.*kaw|zmielon|such.*kaw", text):
        return "grinder"
    if brand == "saeco" and model == "granaroma" and has_code(text):
        return "water"
    if re.search(r"pomp|przeplyw|napelnieni|zapowietrz|wymianie filtra|nie (?:pobiera|podaje) wody|(?:woda|wody).*(?:nie leci|nie plynie|kapie)|nie leci (?:goraca )?woda|pojedyncze krople|brak wody|woda (?:ledw|slabo)|(?:goraca woda|z dyszy|z wylewki).*krop", text):
        return "water"
    return None


def has_code(text):
    if text.strip(" .") == "05":
        return True
    return re.search(r"\b(?:kod|blad|error)\s*[:#]?\s*0?5\b", text) is not None


def read_observations(model, message, fields, pending, cpu):
    reply = normalize(message).strip(" .!?")
    if pending in fields:
        if reply in ("nie wiem", "nie pamietam", "trudno powiedziec"):
            return [(pending, UNKNOWN)], cpu
        choices = FLOW if pending == "water_flow" else ["tak", "nie"]
        for choice in choices:
            if reply == normalize(choice):
                return [(pending, choice)], cpu
    if reply in ("tak", "nie"):
        return [], cpu

    properties = {"field": {"type": "string", "enum": fields},
                  "value": {"type": "string", "enum": ["tak", "nie", UNKNOWN, *FLOW]},
                  "quote": {"type": "string"}}
    item = {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}
    schema = {"type": "object", "properties": {"observations": {"type": "array", "items": item,
                                                                 "maxItems": len(fields)}},
              "required": ["observations"]}
    context = {"pola": {field: QUESTIONS[field] for field in fields},
               "pytanie": QUESTIONS.get(pending), "wypowiedź": message}
    prompt = "Zwróć JSON z listą observations. Każdy element ma field (angielski klucz z pola), " \
             "value (odpowiedź po polsku) oraz quote (dosłowny krótki fragment wypowiedzi). " \
             'Przykład dla zdania „Pompa milczy”: {"observations":[{"field":"pump_audible","value":"nie","quote":"Pompa milczy"}]}. ' \
             "Wypisz tylko obserwacje wyrażone w wypowiedzi, każdą z dosłownym cytatem. " \
             "Nie diagnozuj. Pomiń brakujące informacje. Odróżniaj pompę od młynka. " \
             "Dla water_flow używaj poziomu wypływu. Dla pozostałych pól użyj tak, nie lub nie wiadomo. " \
             "Głośna pompa oznacza też słyszalną pompę. Mielenie oznacza słyszalny młynek. " \
             "Sprzeczne obserwacje tego samego pola zapisz osobno."
    messages = [SystemMessage(content=prompt), HumanMessage(content=json.dumps(context, ensure_ascii=False))]
    options = {"temperature": 0, "num_predict": 640}
    if cpu:
        options["num_gpu"] = 0
    try:
        result = model.invoke(messages, format=schema, options=options)
    except ResponseError as error:
        if cpu or "PTX was compiled with an unsupported toolchain" not in str(error):
            raise
        cpu = True
        options["num_gpu"] = 0
        result = model.invoke(messages, format=schema, options=options)
    observations = []
    for entry in json.loads(result.content)["observations"]:
        field = entry["field"]
        value = entry["value"]
        quote = entry["quote"]
        if not isinstance(quote, str):
            continue
        quote = quote.strip(" \t\r\n.,;:!?„”\"")
        choices = FLOW if field == "water_flow" else ["tak", "nie"]
        if field in fields and value in [*choices, UNKNOWN] and quote and normalize(quote) in normalize(message):
            observations.append((field, value))
    return observations, cpu


def component_reply(state, model):
    domain = domain_for(state)
    fields = WATER if domain == "water" else GRINDER
    pending = state.get("pending_question")
    message = state.get("message") or state.get("description", "")
    facts = state.get("component_facts", {}).copy()
    conflicts = list(state.get("component_conflicts", []))
    cpu = state.get("use_cpu", False)
    parse_warning = ""
    try:
        observations, cpu = read_observations(model, message, [*fields, "wet_electrics"], pending, cpu)
    except (ConnectionError, ResponseError, ValueError, KeyError, TypeError):
        observations = []
        parse_warning = "Nie udało się odczytać opisu przez Bielika. "
    resolved = pending in conflicts
    if ("pump_loud", "tak") in observations and ("pump_audible", "tak") not in observations:
        observations.append(("pump_audible", "tak"))
    for field, value in observations:
        if field in conflicts:
            if field == pending and resolved:
                facts[field] = value
                conflicts.remove(field)
                resolved = False
            continue
        previous = facts.get(field)
        if previous not in (None, UNKNOWN, value):
            if value != UNKNOWN:
                conflicts.append(field)
        else:
            facts[field] = value

    brand, series = device(state)
    granaroma = brand == "saeco" and series == "granaroma"
    if granaroma and has_code(normalize(message)):
        facts["granaroma_05"] = "tak"
    description = state.get("description", "")
    if state.get("message"):
        description = (description + " " + message).strip()
    updates = {"component_domain": domain, "component_facts": facts, "component_conflicts": conflicts,
               "use_cpu": cpu, "description": description, "pending_question": None, "candidates": []}
    if (facts.get("wet_electrics") == "tak" or facts.get("wet_grinder") == "tak"
            or "wet_electrics" in conflicts or "wet_grinder" in conflicts):
        updates["answer"] = "Przerwij używanie ekspresu. Jeśli można to zrobić bezpiecznie, odłącz zasilanie. Przed ponownym użyciem zleć kontrolę w serwisie."
        return updates

    def ask(field, prefix=""):
        updates["pending_question"] = field
        updates["answer"] = parse_warning + prefix + QUESTIONS[field]
        return updates

    if conflicts:
        return ask(conflicts[0], "Mam dwie różne odpowiedzi. Doprecyzuj proszę. ")
    if parse_warning and pending in fields:
        return ask(pending)

    records = []
    evidence = []
    for record in load_faults():
        for pattern in record.get("observation_patterns", []):
            if not all(facts.get(field) == value for field, value in pattern.items()):
                continue
            if record["id"].startswith("grinder_chute") and not granaroma and facts.get("removable_group") != "tak":
                if "removable_group" not in facts:
                    return ask("removable_group")
                continue
            records.append(record)
            for field in pattern:
                if field not in evidence:
                    evidence.append(field)
            break

    # A recent filter/tank event helps separate air from an obstruction.
    record_ids = {record["id"] for record in records}
    if "water_air" in record_ids:
        records = [r for r in records if r["id"] != "water_restricted"]
    elif "water_restricted" in record_ids and "air_trigger" not in facts:
        return ask("air_trigger")
    elif "water_restricted" in record_ids and facts.get("air_trigger") == "tak" and "pump_loud" not in facts:
        return ask("pump_loud")

    if records:
        updates["candidates"] = records
        lines = [observation_line(field, facts[field]) for field in evidence]
        for record in records:
            lines.append("Możliwe przyczyny: " + "; ".join(record["possible_causes"]) + ".")
            lines.extend(record["safe_checks"])
            lines.append("Serwis: " + record["service_check"])
        updates["answer"] = "\n".join(lines)
        return updates

    if domain == "water":
        next_fields = ["pump_audible", "water_flow"]
        if facts.get("water_flow") == "normalnie":
            next_fields.append("cycle_error")
        else:
            next_fields.extend(["air_trigger", "pump_loud", "hot_water_affected", "water_to_tray"])
    else:
        next_fields = ["grinder_audible", "grinder_abnormal", "loose_grounds", "chute_blocked",
                       "grounds_delivered", "foreign_body", "wet_grinder"]
    for field in next_fields:
        if field not in facts:
            return ask(field)
    updates["answer"] = "Zebrane obserwacje nie wskazują jednej przyczyny. Przekaż opis i komunikat ekspresu do serwisu."
    return updates


def observation_line(field, value):
    if field == "granaroma_05":
        return "Kod 05 w Saeco GranAroma oznacza powietrze w układzie wodnym."
    labels = {"pump_audible": "Słychać pompę", "water_flow": "Wypływ wody",
              "cycle_error": "Przerwany cykl lub błąd napełnienia", "pump_loud": "Głośna pompa",
              "air_trigger": "Początek po zmianie filtra, opróżnieniu zbiornika lub postoju",
              "priming_helped": "Poprawa po odpowietrzeniu", "grinder_audible": "Słychać młynek",
              "loose_grounds": "Sucha kawa rozsypana w obudowie", "chute_blocked": "Kawa zalega w kanale",
              "grinder_abnormal": "Nietypowa praca młynka", "foreign_body": "Ciało obce w młynku"}
    return f"{labels.get(field, QUESTIONS[field])} — {value}"
