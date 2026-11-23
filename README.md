# Coffee Machine Diagnostic Agent

Polskojęzyczny asystent do wstępnej diagnostyki automatycznych ekspresów do kawy. Działa lokalnie w terminalu. Użytkownik opisuje objaw własnymi słowami, a agent dobiera pytania, zapamiętuje odpowiedzi i podpowiada, co można bezpiecznie sprawdzić. Rozmową steruje **LangGraph**, a lokalny model **Bielik 4.5B Instruct** działa przez **Ollamę**.

Projekt łączy moje doświadczenie w serwisowaniu ekspresów z zainteresowaniem programowaniem i modelami językowymi. Rozwijam go na podstawie sytuacji, które znam z pracy. Na przykład przy zgłoszeniu „woda pod ekspresem” ważne jest dla mnie rozróżnienie, gdzie użytkownik widzi wodę i co pozwoliłoby ustalić źródło wycieku.

**Etap projektu — lokalne MVP przed testami zewnętrznymi.** Na przypadkach serwisowych dopracowuję opisy objawów, dopasowanie do marek i serii oraz kolejność pytań. Wagi Bielika pozostają takie jak w opublikowanym modelu. Rekordy obejmują obserwacje serwisowe i przykłady ilustracyjne. Sugerowane przyczyny wymagają potwierdzenia.

## Przykładowa rozmowa — wyciek w Saeco GranAroma

Fragment rozmowy z lokalnego demo. Użytkownik wpisał `saeco granaroma` w polu marki i pozostawił pole modelu puste.

> **Użytkownik:** Woda się leje z ekspresu.
>
> **Agent:** Gdzie pojawia się woda?
>
> **Użytkownik:** Woda z tyłu ekspresu.
>
> **Agent:** Woda pojawia się z tyłu ekspresu. Kiedy występuje problem?
>
> **Użytkownik:** W trakcie płukania od razu.
>
> **Agent:** Wodę widać z tyłu ekspresu podczas płukania. Czy przy pobieraniu gorącej wody też widzisz wyciek z tyłu ekspresu?
>
> **Użytkownik:** Tak.
>
> **Agent:** Wodę widać z tyłu ekspresu podczas płukania. Czy cieknie już w trakcie nalewania gorącej wody, czy dopiero na końcu, gdy strumień ustaje?
>
> **Użytkownik:** Dopiero na końcu.
>
> **Agent:** Wodę widać z tyłu ekspresu podczas płukania. Przy gorącej wodzie wyciek widać dopiero po odcięciu strumienia. Trop z przypadku serwisowego tej serii: Przewód zrzutowy zespołu elektrozaworów mógł wypiąć się z małego elementu obok przepływomierza. Wtedy woda przeznaczona do tacki spływa do wnętrza i może wypłynąć z tyłu ekspresu. Po jednej czarnej kawie wyciek może być niewielki, a po kawie mlecznej z płukaniem od razu może być widać więcej wody. Żeby potwierdzić źródło w tym egzemplarzu, serwis musi ustalić, z którego połączenia wypływa woda.

W tym przykładzie użytkownik zaobserwował wyciek podczas dwóch czynności. W bazie dla tej serii opisano m.in. przypadek wypięcia węża z elementu obok przepływomierza. Żeby powiązać tę konkretną przyczynę ze zgłoszeniem, potrzebna jest dodatkowa obserwacja wskazująca na to połączenie.

## Jak powstaje odpowiedź

1. **Stan rozmowy.** Graf zapisuje rozpoznane fakty, m.in. miejsce pojawienia się wody, moment usterki i dźwięk młynka. Pamięć jednego przypadku działa w obrębie uruchomionego programu.
2. **Wyszukiwanie w JSON.** Agent porównuje słowa z opisu z objawami i frazami w bazie oraz uwzględnia ograniczenia marki, modelu i serii. Wynik dopasowania opisuje podobieństwo tekstu. Na jego podstawie nie da się wyliczyć prawdopodobieństwa awarii.
3. **Wybór kolejnego kroku.** Przy słabym dopasowaniu graf prosi o szczegóły. Kiedy potrzebny jest wybór rekordu i tylko jeden spełnia próg, graf korzysta z niego bezpośrednio. Przy kilku kandydatach przekazuje ich opisy Bielikowi, który może wskazać rekord lub zrezygnować z wyboru.
4. **Hipoteza i sprawdzenie.** Program pobiera możliwe przyczyny i zalecane czynności z wybranego rekordu. Pilnuje, o co jeszcze warto zapytać i które porady użytkownik już otrzymał.

## Uruchomienie przez Docker Compose

Uruchom Docker Desktop w trybie kontenerów Linux. W katalogu projektu przygotuj obraz aplikacji i pobierz Bielika do magazynu Ollamy w kontenerze (około 5,1 GB):

```powershell
docker compose build app
docker compose up -d --wait ollama
docker compose exec ollama ollama pull SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0
```

Rozpocznij rozmowę w terminalu:

```powershell
docker compose run --rm app
```

Przy kolejnym uruchomieniu wystarczy ostatnie polecenie. Model działa na CPU. Wolumen `models` przechowuje pobranego Bielika, a `submissions` przechowuje lokalną bazę zgłoszeń. Polecenie `docker compose down` zatrzymuje zestaw i zachowuje oba wolumeny. Opcja `-v` usuwa ich zawartość. Zgłoszenia zapisane w wersji dockerowej można wyświetlić poleceniem `docker compose run --rm app python admin.py list`.

## Uruchomienie na Windows bez Dockera

Potrzebujesz Pythona 3.13, PowerShella i zainstalowanej Ollamy. W katalogu projektu wykonaj:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0
python app.py
```

Ollama powinna działać podczas rozmów wymagających wyboru przez model. Aby zobaczyć ranking rekordów i zapamiętane obserwacje po każdej odpowiedzi, uruchom `python demo_chat.py`.

## Zgłoszenia i prywatność

Po zakończeniu rozmowy można dobrowolnie zapisać przypadek do ręcznej weryfikacji. Aplikacja zwraca numer zgłoszenia. W tej samej instalacji wpisz `status` przy pytaniu `Objawy:`, a następnie podaj numer. Lokalna baza SQLite przechowuje zgłoszenia oddzielnie od bazy wiedzy JSON. Podanie e-maila jest opcjonalne i wymaga zgody na zapis; wysyłka wiadomości jest obecnie nieaktywna.

```powershell
python admin.py list --status pending
```

## Kod i testy

- [src/agent.py](src/agent.py) — stan rozmowy, graf LangGraph i integracja z Bielikiem.
- [src/knowledge_base.py](src/knowledge_base.py) — wyszukiwanie usterek w [bazie przypadków](data/faults.json). [Katalog urządzeń](data/machines.json) zawiera informacje o markach i modelach.
- [src/review_store.py](src/review_store.py) — zapis zgłoszeń, statusów i zdarzeń w SQLite.
- [tests/](tests/) — testy wyszukiwania, rozmów wieloturowych, CLI i obsługi zgłoszeń. Można je uruchomić bez aktywnej Ollamy.

```powershell
python -m unittest discover -s tests -v
```

## Dalsze prace

Przed testami z użytkownikami chcę sprawdzić trafność pytań i hipotez na kolejnych przypadkach, także na opisach sformułowanych inaczej niż w bazie. Do rozpatrzenia później pozostają interfejs graficzny, wyszukiwanie semantyczne i opcjonalne powiadomienia.

| Objaw do zbadania | Hipoteza do sprawdzenia |
|---|---|
| Błąd napełniania. Istotne są dokładny komunikat i model ekspresu. | Zabrudzony lub uszkodzony przepływomierz. |
| Kawa kapie, płynie nierówno lub tylko z jednego z dwóch otworów. | Zabrudzona wylewka kawy. |
| Podczas parzenia kawa nie wypływa albo leci wyłącznie kroplami. | Zatkane sito zaparzacza. W praktyce serwisowej obserwowałem przy zatkanym sicie również silne kropelkowanie. |
| W ekspresie z kapucynatorem mleko płynie niestabilnie lub tylko z jednego z kilku otworów. | Zabrudzony kapucynator w modelach wyposażonych w taki układ. |

Przy opisie „leci z jednej dziurki” pierwsze pytanie powinno ustalić, czy chodzi o kawę, czy o mleko.
