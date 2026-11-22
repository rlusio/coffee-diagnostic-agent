# Coffee Machine Diagnostic Agent — MVP v3

Edukacyjny prototyp agenta wspierającego diagnostykę ekspresów automatycznych. Łączy uporządkowane dane o objawach i modelach z prostym grafem LangGraph. **Nie jest to zweryfikowany system diagnozujący awarie.** Na tym etapie bez LLM i bez Streamlit; to świadomie rozdzielony fundament pod późniejszą integrację Ollamy.

## Co zawiera v3?

- `data/faults.json`: **35 rekordów**. Usterki układu wodnego, wycieki, objawy mleka, brak zasilania i grzania, a także szczególne obserwacje serwisowe. Dodatkowe pytania o wyciek, gorącą wodę i przepełnienie tacki oraz odnotowany pojedynczy przypadek naprawy modułu De'Longhi (nie jest to diagnoza ogólna).
- `data/machines.json`: **23 wpisy** z modelami i seriami. Wpis w katalogu **nie oznacza**, że wewnętrzna konstrukcja i wszystkie usterki zostały potwierdzone. Informacje o istnieniu serii i zgodności z konkretnymi usterkami są rozdzielone.
- `src/knowledge_base.py`: wyszukiwanie kandydatów na podstawie symptomów z filtrem marki i serii; proste tokeny, a nie semantyczne ML.
- `src/agent.py`: prosty graf LangGraph (`retrieve` → `clarify` lub `answer`).
- `src/review_store.py`: SQLite — **oddzielna kolejka zgłoszeń do ręcznej weryfikacji**, zdarzenia, statusy i szkice e-maili w outboxie.
- `admin.py`: **tylko lokalny CLI administratora**, do przeglądania zgłoszeń, zmiany statusu i przeglądania logów.
- `app.py`: lokalna konsola wyszukiwania i możliwość **dobrowolnego** zgłoszenia nowego przypadku; pole `status` umożliwia sprawdzenie stanu sprawy po numerze.

## Instalacja (Windows CMD, Python 3.13)

W nowym katalogu, z aktywnym środowiskiem `.venv`:

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python app.py
```

Uwaga: jeśli masz już `.venv`, nie musisz tworzyć go ponownie. Nie nadpisuj własnych zmian bez kopii zapasowej.

## Jak działa weryfikacja?

1. Użytkownik sam potwierdza, że chce **przesłać przypadek**. Zwykłe pytania diagnostyczne nie trafiają automatycznie do kolejki.
2. Zgłoszenie zapisuje się **lokalnie** w `data/review_queue.sqlite3` ze statusem `pending` (oczekuje).
3. Administrator przegląda i ustawia status: `in_review`, `needs_info`, `approved_for_kb` albo `closed`. `closed` może oznaczać m.in. duplikat, spam lub brak możliwości potwierdzenia; nie oznacza, że każde nowe zgłoszenie odrzucamy.
4. `approved_for_kb` **nie dopisuje automatycznie żadnej usterki do `faults.json`**. Zmiana bazy wiedzy wymaga ręcznego sprawdzenia i aktualizacji repozytorium.
5. Jeśli użytkownik dobrowolnie poda adres e-mail i zgodę, zmiana statusu na `needs_info`, `approved_for_kb` lub `closed` przygotuje szkic wiadomości w tabeli `outbox`. **NIC NIE JEST JESZCZE WYSYŁANE.** Podłączenie dostawcy e-maili (SMTP/API) i systemu ochrony przed spamem to osobny, przyszły etap.

### Panel administratora — na razie CLI

```cmd
python admin.py list
python admin.py list --status pending
python admin.py show NUMER_ZGLOSZENIA
python admin.py review NUMER_ZGLOSZENIA needs_info --note "Podaj dokładny model"
python admin.py review NUMER_ZGLOSZENIA approved_for_kb --note "Potwierdzono po weryfikacji"
python admin.py outbox
python admin.py logs
```

Nie publikuj CLI `admin.py` jako dostępnego publicznie endpointu ani bez uwierzytelniania. To lokalny prototyp; prawdziwy panel administracyjny wymaga autoryzacji.

## Jakie logi zapisujemy?

- `search` — liczba znalezionych kandydatów i informacja, czy marka/model były podane, **bez surowego tekstu zgłoszenia**.
- `case_submitted` i `duplicate_submission` — informacje o dobrowolnym zgłoszeniu/duplikacie.
- `status_updated` — historia decyzji administratora.
- `email_queued_not_sent` — utworzony szkic powiadomienia, **nie wysłana wiadomość**.

Zgłoszone przez użytkownika objawy, marka, model, ewentualny adres e-mail i notatki trafiają do prywatnej bazy SQLite; plik wyklucza `.gitignore`. **Nie publikuj go na GitHubie.** Do publicznego wdrożenia trzeba dodać politykę prywatności, retencję/usuwanie danych, kontrolę dostępu, ograniczenia zgłoszeń, ochronę przed botami, obsługę zgód oraz bezpieczny serwis e-mail.

## Wiarygodność i bezpieczeństwo

- `evidence_status: observed` — relacja technika, a nie automatycznie potwierdzony fakt dla każdego modelu.
- `evidence_status: tentative` — hipoteza wymagająca sprawdzenia.
- `verified: false` — brak niezależnej weryfikacji w instrukcji dla wskazanego modelu.
- Puste `applicability.models` oznacza brak potwierdzonej listy modeli, nie zgodność uniwersalną.
- Nie instruować nieprzeszkolonych użytkowników do pracy pod napięciem, na gorących elementach lub przy układach ciśnieniowych. Nie zalecać automatycznie nieautoryzowanych przeróbek ani lutowania modułu.
- Nie kopiować danych klientów, materiałów poufnych pracodawcy ani danych identyfikujących klienta do zgłoszeń.

## Następny etap

Dopracować dopasowywanie objawów i logikę pytań, a następnie podłączyć Ollamę do ekstrakcji z tekstu i LangGraph do rozmów wieloetapowych. Powiadomienia e-mail to rozszerzenie — nie wymóg do nauki Agentic AI w MVP.
