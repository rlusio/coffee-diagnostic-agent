"""Wyszukiwanie hipotez serwisowych: nie potwierdza diagnozy ani zgodności modelu."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'faults.json'
STOPWORDS = {'a', 'i', 'w', 'z', 'na', 'nie', 'ale', 'do', 'jest', 'sie', 'po', 'ze', 'oraz'}


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKD', str(text).lower().replace('ł', 'l'))
    return ''.join(c for c in text if not unicodedata.combining(c))


def normalize_identifier(value: str) -> str:
    """Np. De'Longhi -> delonghi, 'Gran Aroma' -> granaroma."""
    return re.sub(r'[^a-z0-9]', '', normalize(value))


def split_catalog_device(brand: str, model: str) -> tuple[str, str]:
    """Recognize a complete catalog name entered in the brand field."""
    if model or not brand.strip():
        return brand, model
    with DATA_PATH.with_name("machines.json").open(encoding="utf-8") as stream:
        machines = json.load(stream)
    entered = normalize_identifier(brand)
    for machine in machines:
        for name in (machine.get("model"), machine.get("series")):
            if name and entered == normalize_identifier(machine["brand"] + name):
                return machine["brand"], name
    return brand, model


def tokenize(text: str) -> set[str]:
    return {word for word in re.findall(r'[a-z0-9]+', normalize(text))
            if len(word) > 2 and word not in STOPWORDS}


def load_faults(path: Path = DATA_PATH) -> list[dict]:
    with path.open(encoding='utf-8') as stream:
        faults = json.load(stream)
    if not isinstance(faults, list):
        raise ValueError('Knowledge base must contain a list of faults')
    required = {'id', 'symptoms', 'possible_causes', 'diagnostic_questions', 'verified', 'source'}
    seen = set()
    for record in faults:
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError('Invalid fault record')
        if record['id'] in seen:
            raise ValueError(f'Duplicate fault ID: {record["id"]}')
        seen.add(record['id'])
        if not isinstance(record['symptoms'], list) or not isinstance(record['possible_causes'], list):
            raise ValueError(f'Invalid list fields in {record["id"]}')
    return faults


def _is_applicable(record: dict, brand: str, model: str) -> bool:
    """Nie dopasowuj wpisów serii do nieznanego modelu."""
    rule = record.get('applicability', {})
    brands = [normalize_identifier(b) for b in rule.get('brands', [])]
    models = [normalize_identifier(m) for m in rule.get('models', [])]
    series = [normalize_identifier(s) for s in rule.get('series', [])]
    brand_id = normalize_identifier(brand)
    model_id = normalize_identifier(model)
    if brands and brand_id not in brands:
        return False
    if models and model_id not in models:
        return False
    if series and not (model_id and any(s in model_id for s in series)):
        return False
    return True


def find_faults(description: str, brand: str = '', model: str = '', top_k: int = 3,
                faults: list[dict] | None = None) -> list[dict]:
    """Zwróć kandydatów, nie diagnozę. Wpisy z danej serii wymagają nazwy serii w modelu."""
    faults = faults if faults is not None else load_faults()
    query = tokenize(description)
    results = []
    for record in faults:
        if not _is_applicable(record, brand, model):
            continue
        required_words = record.get("required_symptom_words", [])
        if required_words and query.isdisjoint(tokenize(" ".join(required_words))):
            continue
        phrases = record.get('symptoms', []) + record.get('keywords', [])
        overlap = max((len(query & tokenize(phrase)) for phrase in phrases), default=0)
        if overlap == 0:
            continue
        applicable = record.get('applicability', {})
        # Przy równym dopasowaniu tekstowym preferujemy wpis dla podanej marki,
        # jednak NIGDY nie ogłaszamy na tej podstawie pewnej diagnozy.
        brand_specific = bool(applicable.get('brands', []))
        score = overlap + (0.25 if brand_specific else 0)
        exact_models = [normalize_identifier(m) for m in applicable.get('models', [])]
        confirmed = bool(record.get('verified')) and bool(model) and normalize_identifier(model) in exact_models
        results.append({
            'id': record['id'], 'score': score, 'category': record.get('category'),
            'component': record.get('component'), 'possible_causes': record['possible_causes'],
            'diagnostic_questions': record['diagnostic_questions'],
            'safe_checks': record.get('safe_checks', []),
            'notes': record.get('notes', ''), 'evidence_status': record.get('evidence_status', 'unknown'),
            'verified': record['verified'], 'source': record['source'],
            'model_compatibility_confirmed': confirmed,
        })
    return sorted(results, key=lambda r: (-r['score'], r['id']))[:top_k]
