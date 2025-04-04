# 🤖 KI Workflow Generator & Runner: Ein Multi-Agenten-Framework mit dynamischer Generierung

Dieses Projekt präsentiert ein fortschrittliches Framework zur Erstellung, Verwaltung und Ausführung von KI-gesteuerten Workflows. Es kombiniert die Leistungsfähigkeit von Large Language Models (LLMs) – speziell Google Gemini – mit einem flexiblen Multi-Agenten-System und einer einzigartigen **dynamischen Workflow-Generierungsfunktion**. Über eine intuitive Streamlit-Weboberfläche können Benutzer sowohl vordefinierte, spezialisierte Workflows nutzen als auch die KI beauftragen, maßgeschneiderte, mehrstufige Agenten-Abläufe für komplexe Aufgaben *on-the-fly* zu entwerfen und auszuführen.

Der Kern des Projekts liegt in der Orchestrierung von "Agenten" – spezialisierten LLM-Instanzen –, die zusammenarbeiten, Informationen austauschen und externe Werkzeuge (Tools) nutzen können. Der Code ist mit Fokus auf **Klarheit, Wartbarkeit und moderne Python-Praktiken** (PEP 634 Pattern Matching, PEP 604 Union Types, Typ-Annotationen) entwickelt worden.

**Zielgruppe:** Dieses Framework richtet sich an KI-Forscher, Entwickler, Datenwissenschaftler, Studenten und alle, die an der Spitze der LLM-Anwendungen arbeiten möchten. Es eignet sich hervorragend für die Exploration von Multi-Agenten-Kollaboration, automatisierter Problemlösung, dynamischer Prozessautomatisierung und der Entwicklung komplexer KI-gesteuerter Anwendungen.

---

## ✨ Herausragende Merkmale

*   **🚀 Dynamische Workflow-Generierung:** Das Kernstück! Eine Meta-KI entwirft auf Basis einer einfachen Zielbeschreibung eine komplette Multi-Agenten-Workflow-Konfiguration (JSON), die dann sofort ausgeführt werden kann. Dies ermöglicht eine beispiellose Flexibilität und Anpassungsfähigkeit an neue Aufgaben.
*   **🧩 Modulare Multi-Agenten-Architektur:** Definieren Sie Workflows als Abfolgen spezialisierter Agenten, die jeweils eigene Anweisungen, Fähigkeiten (Tools, Websuche) und Datenzugriffe (Dateien, Ergebnisse anderer Agenten) besitzen.
*   **🛠️ Erweiterbare Tool-Integration (Function Calling):** Agenten können vordefinierte Python-Funktionen (z.B. Rechner, Datumsabfrage, potenziell beliebige APIs oder benutzerdefinierte Logik) aufrufen, um über reines Textverständnis hinauszugehen und aktiv mit externen Systemen zu interagieren.
*   **📂 Kontext-Anreicherung durch Dateien:** Benutzer können diverse Dateitypen (Text, Code, Bilder) hochladen, die ausgewählten Agenten als wichtiger Kontext für ihre Aufgaben dienen.
*   **🌐 Optionale Websuche:** Ermöglichen Sie Agenten den Zugriff auf aktuelle Informationen aus dem Internet über die Google Search API.
*   **🖥️ Interaktive Streamlit-Weboberfläche:** Eine benutzerfreundliche UI für Workflow-Auswahl, Aufgabenstellung, Datei-Upload, Prozessverfolgung und detaillierte Ergebnisanzeige.
*   **📄 Strukturierte Konfiguration & Ausgabe:** Workflows werden über klares JSON definiert. Ergebnisse werden pro Agent visualisiert, inklusive Status, Output, genutzten Quellen und potenziell extrahierten Code-Dateien.
*   **📦 Download von Artefakten:** Generierte Code-Dateien oder andere textbasierte Artefakte können bequem als ZIP-Archiv heruntergeladen werden.
*   **🐍 Moderne Python-Implementierung:** Einsatz von Python 3.10+ Features für robusten und lesbaren Code (Pattern Matching für Konfigurationsvalidierung, Typ-Annotationen).

---

## 🏛️ Architektur & Konzepte

Dieses Framework basiert auf einer modularen Architektur, die Flexibilität und Erweiterbarkeit ermöglicht.

```
+--------------------------+      +------------------------+      +---------------------+
|   Streamlit Web UI       | <--> |   Workflow Manager     | <--> |   Google GenAI API  |
| (Nutzerinteraktion,      |      | (Konfig laden/valid., |      | (Gemini LLM)        |
|  Input, Output Anzeige)  |      |  Agenten steuern)      |      +----------^----------+
+-------------^------------+      +-----------^------------+                 |
              |                            |                               | Tools via
              | Uploaded Files             | Agent Config (JSON)           | Function Calling
              |                            |                               |
+-------------v------------+      +--------v---------+      +--------------v--------------+
|   Agent Runner           | ---> |   Agent Instance   | ---> |   Tool Executor           |
| (Führt Agenten aus,      |      | (LLM Call, Prompt  |      | (Führt Python Funktionen |
|  verwaltet Nachrichten)  |      |  Aufbau, Tool-Wahl)|      |  aus, gibt Ergebnis zurück)|
+--------------------------+      +--------------------+      +---------------------------+
              |                            ^
              | Ergebnisse/Nachrichten     | Tool Ergebnis
              +----------------------------+
```
*(Diagramm: Vereinfachte Darstellung der Hauptkomponenten und des Datenflusses)*

**Kernkomponenten:**

1.  **Streamlit Web UI:** Die Schnittstelle zum Benutzer. Ermöglicht Auswahl des Workflows, Eingabe der Aufgabe, Datei-Uploads und Anzeige der Ergebnisse. Nutzt Streamlits `session_state` zur Verwaltung des Zustands zwischen Interaktionen.
2.  **Workflow Manager:**
    *   Lädt und validiert die JSON-Konfigurationsdateien für die Workflows. Nutzt **Pattern Matching (match/case)** zur robusten Überprüfung der Struktur und Typen der Agenten-Definitionen.
    *   Orchestriert den Ablauf, insbesondere im **Dynamischen Modus**: Ruft zuerst den `WorkflowGenerator`-Agenten auf.
    *   Interpretiert die `round`- und `receives_messages_from`-Felder, um die Ausführungsreihenfolge und Abhängigkeiten zu bestimmen.
3.  **Agent Runner:**
    *   Iteriert durch die (vordefinierten oder generierten) Agenten gemäß ihrer Reihenfolge und Abhängigkeiten.
    *   Bereitet den spezifischen Input (Systemanweisung, Nutzeranfrage, vorherige Ergebnisse, Dateien) für jeden Agenten vor.
    *   Verwaltet den `message_store`, in dem die Ausgaben der Agenten für nachfolgende Agenten gespeichert werden.
4.  **Agent Instance:** Repräsentiert einen einzelnen Agenten im Workflow.
    *   Formuliert den finalen Prompt für das LLM.
    *   Konfiguriert den API-Aufruf an Google GenAI (Modell, Temperatur, Tools, Websuche).
    *   Interpretiert die Antwort des LLM:
        *   Extrahiert den Text-Output.
        *   Erkennt und initiiert **Function Calls** (Tool-Nutzung).
        *   Verarbeitet die Ergebnisse von Tool-Aufrufen und sendet sie zurück an das LLM für die finale Antwort.
5.  **Tool Executor:**
    *   Empfängt Anfragen vom Agenten, ein bestimmtes Tool (Python-Funktion aus `AVAILABLE_TOOLS`) auszuführen.
    *   Ruft die entsprechende Funktion mit den vom LLM übergebenen Argumenten auf.
    *   Gibt das Ergebnis der Funktion standardisiert an die Agent Instance zurück. **Sicherheitshinweis:** Aktuell nutzt der `simple_calculator` `eval()`, was unsicher ist. Siehe FAQ.
6.  **Google GenAI API:** Die externe Schnittstelle zu den Gemini-Modellen, die die eigentliche "Intelligenz" der Agenten liefern.

**Ablauf-Modi:**

*   **Statischer Workflow:**
    1.  Nutzer wählt einen vordefinierten Workflow (z.B. "Python Aufgabe").
    2.  Workflow Manager lädt die entsprechende JSON-Datei (z.B. `agents_config_python.json`).
    3.  Agent Runner führt die Agenten gemäß `round` und `receives_messages_from` aus. Agenten erhalten die initiale Nutzeranfrage und ggf. Dateien oder die Ergebnisse vorheriger Agenten.
*   **Dynamischer Workflow:**
    1.  Nutzer wählt "Dynamischer Workflow Generator".
    2.  Workflow Manager lädt `generator_agent_config.json`.
    3.  Agent Runner führt *nur* den Generator-Agenten aus. Dieser erhält die Nutzer-Zielbeschreibung und ggf. Dateien.
    4.  Der Generator-Agent gibt eine **neue JSON-Konfiguration** (eine Liste von Agenten-Definitionen) als seinen Output zurück.
    5.  Workflow Manager validiert diese *generierte* Konfiguration.
    6.  Agent Runner führt nun die Agenten aus der *generierten* Konfiguration aus, wobei die ursprüngliche Nutzeranfrage (und ggf. Dateien) als initialer Input dient.

**Multi-Agenten-Interaktion:**

*   Die Kommunikation erfolgt primär **sequentiell** basierend auf den Abhängigkeiten (`receives_messages_from`).
*   Der Output eines Agenten wird im `session_state.message_store` gespeichert und als Input für abhängige Agenten bereitgestellt.
*   Dies ermöglicht einfache Pipeline-Strukturen, aber auch komplexere Informationsflüsse, wenn Agenten Ergebnisse von mehreren Vorgängern erhalten.

**Function Calling & Tool-Erweiterbarkeit:**

*   Das LLM wird über die API informiert, welche Tools (Funktionen) verfügbar sind, inklusive ihrer Namen, Beschreibungen und Parameter-Schemata.
*   Wenn das LLM entscheidet, dass zur Beantwortung einer Anfrage ein Tool nützlich ist, gibt es statt einer Textantwort eine spezielle `FunctionCall`-Nachricht zurück.
*   Der `Agent Runner` fängt dies ab, ruft den `Tool Executor` auf, führt die Python-Funktion aus und sendet das Ergebnis als `FunctionResponse` zurück an das LLM.
*   Das LLM nutzt dieses Ergebnis dann, um seine finale Antwort zu formulieren.
*   **Neue Tools hinzufügen:**
    1.  Definiere eine neue Python-Funktion (achte auf klare Docstrings und Typ-Annotationen für Parameter).
    2.  Füge die Funktion zum `AVAILABLE_TOOLS`-Dictionary im Skript hinzu (`"tool_name": function_reference`).
    3.  Definiere das Parameter-Schema für die Funktion im `Agent Runner` (im Abschnitt `Konfiguration der Tools`), damit das LLM weiß, welche Argumente es übergeben muss.
    4.  Referenziere `"tool_name"` in der `callable_tools`-Liste eines Agenten in der JSON-Konfiguration.

---

## 🚀 Erste Schritte

### Voraussetzungen

1.  **Python:** Version 3.10 oder höher (wegen Pattern Matching `match/case` und Union Types `|`).
2.  **Google AI API Key:** Erforderlich für die Interaktion mit den Gemini-Modellen. Holen Sie sich Ihren Schlüssel über [Google AI Studio](https://aistudio.google.com/app/apikey).
3.  **Git (Optional):** Zum Klonen des Repositories.

### Installation

1.  **Repository klonen (falls zutreffend):**
    ```bash
    git clone <repository-url>
    cd <repository-ordner>
    ```
    Alternativ: Code herunterladen und entpacken.

2.  **Abhängigkeiten installieren:**
    Erstellen Sie eine `requirements.txt`-Datei:
    ```txt
    streamlit
    google-generativeai
    python-dotenv
    Pillow
    ```
    Installieren Sie die Pakete (idealerweise in einer virtuellen Umgebung):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    pip install -r requirements.txt
    ```

3.  **API-Key konfigurieren:**
    *   Erstellen Sie eine Datei namens `.env` im Hauptverzeichnis.
    *   Fügen Sie Ihren API-Schlüssel hinzu:
        ```
        API_KEY=DEIN_GOOGLE_AI_API_KEY
        ```
    *   Ersetzen Sie `DEIN_GOOGLE_AI_API_KEY` durch Ihren tatsächlichen Schlüssel. Das Skript lädt diesen Schlüssel automatisch beim Start.

### Ausführung

Starten Sie die Streamlit-Anwendung aus dem Projektverzeichnis:

```bash
streamlit run dein_script_name.py
```

*(Ersetzen Sie `dein_script_name.py` durch den tatsächlichen Namen Ihrer Haupt-Python-Datei)*

Die Anwendung sollte sich in Ihrem Standard-Webbrowser öffnen.

---

## 🔧 Konfiguration der Workflows

Die Flexibilität des Systems basiert auf JSON-Konfigurationsdateien, die die Agenten und ihre Interaktionen definieren.

*   **Standard-Workflows:** Vordefinierte Abläufe (z.B. `agents_config_python.json`). Enthalten eine **Liste** von Agenten-Objekten.
*   **Generator-Workflow:** Die spezielle Konfiguration (`generator_agent_config.json`) für den Agenten, der *neue* Workflow-JSONs generiert. Diese Datei enthält typischerweise nur **einen** Agenten.

### Detaillierte Agenten-Konfigurationsstruktur

Hier ist ein Beispiel für ein einzelnes Agenten-Objekt innerhalb der JSON-Liste eines Workflows, mit Erklärungen zu allen Feldern:

```json
{
  "name": "Code_Reviewer", // String: Eindeutiger Bezeichner des Agenten. Muss innerhalb des Workflows einzigartig sein. Wird in `receives_messages_from` anderer Agenten referenziert.
  "round": 3, // Integer: Definiert die grobe Ausführungsphase im *statischen* Workflow. Niedrigere Zahlen zuerst. Dient der Sortierung vor der Abhängigkeitsprüfung.
  "description": "Überprüft den generierten Code auf Qualität, Fehler und Einhaltung von Best Practices.", // String (Optional): Eine kurze Beschreibung, was der Agent tut. Wird z.B. in der Sidebar angezeigt.
  "system_instruction": "Du bist ein erfahrener Code-Rezensent mit Fokus auf Python. Analysiere den bereitgestellten Code sorgfältig auf logische Fehler, potenzielle Bugs, Lesbarkeit, Performance-Engpässe und die Einhaltung von PEP 8. Gib konstruktives Feedback und schlage konkrete Verbesserungen vor. Liste die wichtigsten Punkte klar auf. Wenn keine Probleme gefunden werden, bestätige dies explizit.", // String: Die zentrale Anweisung, die das Verhalten, die Rolle und das Ziel des Agenten definiert. Kritisch für die Leistung!
  "temperature": 0.4, // Float (Optional): Steuert die Kreativität/Zufälligkeit der LLM-Antworten (typ. Bereich 0.0 - 1.0). Niedrigere Werte = deterministischer, höhere = kreativer. Wenn nicht angegeben, wird ein Standardwert des Modells oder der Implementierung verwendet.
  "receives_messages_from": ["Python_Coder", "Requirement_Analyst"], // List[String] (Optional): Eine Liste der `name`-Attribute von Agenten, deren *gesamter Output* als Input für diesen Agenten verwendet wird. Wenn leer oder nicht vorhanden, erhält der Agent die ursprüngliche Benutzeranfrage (+ ggf. Dateien). Der Agent startet erst, wenn alle genannten Vorgänger abgeschlossen sind.
  "callable_tools": ["get_current_datetime"], // List[String] (Optional): Liste der Namen von Tools (Python-Funktionen aus `AVAILABLE_TOOLS`), die dieser Agent über Function Calling verwenden darf.
  "enable_web_search": true, // Boolean (Optional): Wenn `true`, darf der Agent die Google Search API nutzen, um auf aktuelle Webinformationen zuzugreifen (falls vom Modell unterstützt und konfiguriert). Standard ist `false`.
  "accepts_files": false // Boolean (Optional): Wenn `true`, erhält dieser Agent zusätzlich zu seinem regulären Input (Nutzeranfrage oder Output der Vorgänger) auch den Inhalt der vom Benutzer hochgeladenen Dateien. Nützlich für Agenten, die direkt mit Dateiinhalten arbeiten sollen (z.B. Analyse, Zusammenfassung). Standard ist `false`.
}
```

**Wichtige Hinweise zur Konfiguration:**

*   Die **Qualität der `system_instruction`** ist entscheidend für das Verhalten des Agenten. Seien Sie präzise und klar in der Rollen- und Aufgabenbeschreibung.
*   Das Zusammenspiel von `round` und `receives_messages_from` definiert den tatsächlichen Ausführungsfluss. Abhängigkeiten (`receives_messages_from`) haben Vorrang vor der `round`-Nummer.
*   Im **dynamisch generierten** Workflow werden diese Felder von der Generator-KI selbst erstellt. Die Qualität des generierten Workflows hängt stark von der Fähigkeit des Generators ab, sinnvolle Agentenrollen, Anweisungen und Abhängigkeiten zu definieren.

---

## 💻 Benutzung der Streamlit Weboberfläche

Die Weboberfläche ist Ihr Kontrollzentrum für das Framework.



**Schritt-für-Schritt Anleitung:**

1.  **Workflow auswählen:** Nutzen Sie das Dropdown-Menü (`Wähle den Workflow:`), um entweder einen der vordefinierten Workflows (z.B. "Python Aufgabe") zu starten oder wählen Sie "**Dynamischer Workflow Generator**", um einen neuen Workflow basierend auf Ihrem Ziel zu erstellen.
2.  **Aufgabe definieren:**
    *   **Für vordefinierte Workflows:** Geben Sie eine spezifische Aufgabe in das Textfeld ein (z.B. "Schreibe eine Python-Klasse für einen einfachen Web-Scraper").
    *   **Für den Generator:** Beschreiben Sie das *Gesamtziel*, das der zu generierende Workflow erreichen soll (z.B. "Entwirf einen Workflow, der einen wissenschaftlichen Artikel zusammenfasst, die Kernaussagen extrahiert und mögliche nächste Forschungsschritte vorschlägt").
3.  **Dateien hochladen (Optional):** Klicken Sie auf `Browse files` unter `📎 Dateien hochladen (Kontext):`, um relevante Dokumente (Code, Text, Bilder etc.) hinzuzufügen. Diese werden Agenten zur Verfügung gestellt, die `accepts_files: true` haben und sie als Input benötigen. Sie können hochgeladene Dateien über das `❌`-Symbol neben ihrem Namen wieder entfernen.
4.  **Starten:** Klicken Sie auf den Haupt-Button (`🚀 'Workflow starten'` oder `🧬 Workflow generieren & ausführen`).
5.  **Verarbeitung verfolgen:**
    *   Die App zeigt den Fortschritt an, indem sie den aktuell arbeitenden Agenten hervorhebt (`🧠 Agent: ...`).
    *   Im dynamischen Modus sehen Sie zuerst die Ausgabe des Generators (die generierte JSON-Konfiguration), bevor die eigentliche Ausführung beginnt.
    *   Tool-Aufrufe (`-> Tool ...`) und deren Ergebnisse werden ebenfalls kurz signalisiert.
6.  **Ergebnisse untersuchen:**
    *   **Gesamtstatus:** Oben im Ergebnisbereich sehen Sie eine Meldung (✅ Erfolg / ❌ Fehler).
    *   **Agenten-Details:** Jeder Agent des Laufs erhält einen eigenen ausklappbaren Bereich (`Expander`). Klicken Sie darauf, um Status, detaillierten Output (oft mit Markdown-Formatierung oder Codeblöcken), eventuelle Quellenangaben (Websuche) und Fehlermeldungen zu sehen.
        *(Platzhalter: Hier könnte ein Screenshot eines Ergebnis-Expanders eingefügt werden)*
    *   **Download (falls zutreffend):** Wenn Agenten Dateien im Format `## FILE: dateiname.ext` generiert haben, erscheint der Abschnitt `📦 Download generierter Dateien` mit einer Liste und einem ZIP-Download-Button.
    *   **Finales Ergebnis:** Der Abschnitt `🏁 Finales Text-Ergebnis` versucht, die relevanteste abschließende Textausgabe des Workflows (typischerweise vom letzten erfolgreichen Agenten, der keine reine Code-Ausgabe produziert hat) zu extrahieren und anzuzeigen.
7.  **Sidebar nutzen:** Die Seitenleiste links bietet Zusatzinformationen:
    *   Aktueller Modus und verwendete Konfigurationsdatei.
    *   Übersicht der Agenten im (geladenen oder generierten) Workflow (Tabelle und vollständiges JSON).
    *   Liste der global verfügbaren `Tools`.
    *   Verwendetes Gemini-Modell (`DEFAULT_MODEL_ID`).

---

## 🌐 Anwendungsfälle & Potenziale

Dieses Framework ist mehr als nur ein Code-Generator. Seine flexible Architektur eröffnet ein breites Spektrum an Anwendungsfällen:

*   **Automatisierte Code-Generierung & Refactoring:** Erstellung von Code in verschiedenen Sprachen, Ergänzung von Tests, Dokumentation, Code-Optimierung und Migration.
*   **Komplexe Problemlösung:** Zerlegung großer Probleme in Teilaufgaben, die von spezialisierten Agenten (Planer, Rechercheur, Kritiker, Implementierer) bearbeitet werden.
*   **Forschungsassistenz:** Automatisierte Literaturrecherche, Zusammenfassung von Papern, Extraktion von Kernaussagen, Generierung von Hypothesen.
*   **Datenanalyse & Berichterstellung:** Erstellung von Skripten zur Datenbereinigung und -analyse, Generierung von Visualisierungen (über Code), Verfassen von Ergebnisberichten.
*   **Inhaltsgenerierung & Marketing:** Erstellung von Blogartikeln, Marketingtexten, Social-Media-Posts durch eine Kette von Agenten (Ideenfinder, Texter, Korrektor, SEO-Optimierer).
*   **Lern- und Lehrwerkzeug:** Demonstration von KI-Konzepten, Erstellung personalisierter Lernpfade oder Übungsaufgaben.
*   **Plugin- & Tool-Entwicklung:** Wie im Beispiel `plugin_developer_config.json` gezeigt, kann der Workflow selbst zur Entwicklung neuer Tools genutzt werden.
*   **Simulation & Szenarienplanung:** Modellierung von Interaktionen oder Prozessen durch spezialisierte Agenten.

**Potenziale für Forschung und Entwicklung:**

*   Untersuchung optimaler Agenten-Kommunikationsstrategien.
*   Entwicklung fortgeschrittener Planungs- und Koordinationsmechanismen für Agenten.
*   Erforschung von Selbstheilungs- und Adaptionsfähigkeiten in Multi-Agenten-Workflows.
*   Integration mit externen Datenbanken und APIs über benutzerdefinierte Tools.
*   Benchmarking verschiedener LLMs in Multi-Agenten-Szenarien.
*   Entwicklung von Benutzeroberflächen zur visuellen Workflow-Erstellung.

---

## 📚 Glossar

*(Das Glossar bleibt weitgehend identisch zur vorherigen Version, da es die Kernbegriffe bereits gut abdeckt. Ggf. könnten spezifische Begriffe wie "Meta-KI" oder "Orchestrierung" hinzugefügt werden, aber die vorhandenen Definitionen decken viel ab.)*

*   **Agent:** Eine Instanz eines KI-Modells (hier Gemini), die eine spezifische Rolle oder Aufgabe innerhalb eines Workflows mit eigenen Anweisungen und potenziellen Werkzeugen (Tools) übernimmt.
*   **API Key:** Ein geheimer Schlüssel zur Authentifizierung bei einem Dienst (hier Google GenAI).
*   **Dynamischer Workflow Generator:** Ein spezieller Agent (Meta-KI), der die Konfiguration (JSON) für einen *anderen* Workflow basierend auf einer Nutzeranforderung erstellt.
*   **Function Calling:** Fähigkeit eines LLMs, vordefinierte externe Funktionen (Tools) aufzurufen.
*   **Gemini:** Googles Familie von Large Language Models (LLMs).
*   **Google GenAI:** Plattform und API von Google für generative KI-Modelle.
*   **JSON:** Daten-Austauschformat zur Definition von Agenten-Workflows.
*   **Kontext:** Informationen (Text, Bilder, vorherige Nachrichten), die einem LLM zur Verfügung gestellt werden.
*   **LLM (Large Language Model):** Tiefes neuronales Netzwerk für Textverständnis und -generierung.
*   **Markdown:** Auszeichnungssprache zur Textformatierung.
*   **Multi-Agenten-System:** System, in dem mehrere Agenten interagieren.
*   **Orchestrierung:** Die Steuerung und Koordination der Ausführung und Interaktion von Agenten in einem Workflow.
*   **Pattern Matching (PEP 634):** Python-Feature (`match/case`) zur Strukturprüfung.
*   **Prompt:** Eingabeaufforderung an ein LLM.
*   **Session State (Streamlit):** Mechanismus zum Speichern von Daten über Interaktionen hinweg.
*   **Streamlit:** Python-Framework für Webanwendungen.
*   **System Instruction:** Grundlegende Anweisung, die Rolle und Verhalten eines Agenten definiert.
*   **Tool:** Eine externe (Python-)Funktion, die ein Agent aufrufen kann.
*   **Union Types (PEP 604):** Python-Syntax (`|`) für Typ-Annotationen mit mehreren Möglichkeiten.
*   **Workflow:** Definierte Abfolge von Agentenaktionen zur Erledigung einer Aufgabe.

---

## ❓ FAQ (Häufig gestellte Fragen)

*   **F: Woher bekomme ich einen Google AI API Key?**
    *   A: Besuchen Sie die [Google AI Studio Webseite](https://aistudio.google.com/app/apikey), melden Sie sich an und erstellen Sie einen neuen API-Schlüssel.

*   **F: Welche Gemini-Modelle werden unterstützt?**
    *   A: Standardmäßig `gemini-2.0-pro-exp-02-05`. Andere Gemini-Modelle (wie Pro), die über `google.genai` verfügbar sind und Function Calling unterstützen, sollten ebenfalls funktionieren. Die Modell-ID ist in `DEFAULT_MODEL_ID` festgelegt.

*   **F: Warum ist mein Workflow fehlgeschlagen oder liefert seltsame Ergebnisse?**
    *   A: Mögliche Gründe:
        *   **API-Key:** Ungültig, fehlt oder Kontingent überschritten.
        *   **Netzwerk:** Probleme bei der Verbindung zur Google API.
        *   **Konfiguration:** Fehler im JSON (Syntax, fehlende Felder, ungültige `receives_messages_from`-Namen). Prüfen Sie die Fehlermeldungen beim Laden/Validieren.
        *   **Prompt/Anweisung:** Die `system_instruction` oder die Nutzeranfrage war unklar, mehrdeutig oder zu komplex für das Modell.
        *   **LLM-Beschränkungen:** Das Modell kann "halluzinieren" (falsche Fakten erfinden), Anweisungen ignorieren oder im Kontextfenster begrenzt sein, besonders bei sehr langen Dialogen oder vielen Dateien.
        *   **Sicherheitsfilter:** Die Antwort des Modells wurde von Google blockiert (Inhaltsfilterung). Die Fehlermeldung im Agenten-Expander gibt oft Hinweise (`block_reason`).
        *   **Tool-Fehler:** Eine aufgerufene Python-Funktion (Tool) hat einen Fehler verursacht (z.B. ungültige Eingabe für den Rechner).
        *   **Dynamische Generierung:** Der Generator-Agent hat einen suboptimalen oder fehlerhaften Workflow entworfen.

*   **F: Wie sicher ist die `simple_calculator`-Funktion mit `eval()`?**
    *   A: **Nicht sicher für Produktionsumgebungen!** `eval()` kann beliebigen Code ausführen, wenn die Eingabe nicht streng kontrolliert wird. Die aktuelle Implementierung hat eine sehr einfache Zeichenvalidierung, die aber **nicht** robust gegen raffinierte Angriffe ist. Für den produktiven Einsatz *muss* dies durch einen sicheren mathematischen Ausdrucksparser (z.B. mit Bibliotheken wie `asteval` oder `numexpr` oder einem eigenen Parser) ersetzt werden. Es dient hier nur als einfaches Beispiel für ein Tool.

*   **F: Wie kann ich eigene Tools hinzufügen?**
    *   A: Folgen Sie den Schritten im Abschnitt "Architektur & Konzepte" unter "Function Calling & Tool-Erweiterbarkeit": 1. Python-Funktion definieren. 2. Zu `AVAILABLE_TOOLS` hinzufügen. 3. Parameter-Schema im `Agent Runner` definieren. 4. Tool-Namen in der `callable_tools`-Liste des Agenten-JSON referenzieren.

*   **F: Gibt es Grenzen für die Komplexität von Workflows oder die Menge an Daten?**
    *   A: Ja. LLMs haben ein **Kontextfenster** (maximale Menge an Text, die sie auf einmal verarbeiten können). Sehr lange Konversationen (durch viele Agentenschritte), sehr umfangreiche Systemanweisungen oder große hochgeladene Dateien können dieses Limit überschreiten, was zu Fehlern oder Informationsverlust führt. Die maximale Anzahl an Function Calls pro Agentenschritt ist ebenfalls begrenzt (`max_function_calls`), um Endlosschleifen zu verhindern.

*   **F: Kann ich den Output eines Agenten direkt als Datei speichern lassen, ohne `## FILE:`?**
    *   A: Die aktuelle Implementierung sucht explizit nach dem `## FILE: dateiname.ext\n```...````-Muster, um Dateien für den ZIP-Download zu extrahieren. Eine direkte Dateispeicherung während der Ausführung ist nicht implementiert, könnte aber als neues Tool (z.B. `save_to_file(filename, content)`) hinzugefügt werden.

---

## 🤝 Beitragende & Entwicklung

Dieses Projekt ist als Grundlage für weitere Forschung und Entwicklung gedacht. Beiträge sind willkommen!

**Mögliche Richtungen für zukünftige Entwicklung:**

*   **Visueller Workflow-Editor:** Eine grafische Oberfläche zur Erstellung und Bearbeitung von Workflows.
*   **Parallele Agenten-Ausführung:** Echte parallele Verarbeitung von Agenten derselben Runde (wo Abhängigkeiten es erlauben).
*   **Fortgeschrittene Agenten-Kommunikation:** Implementierung komplexerer Nachrichtenformate oder gemeinsamer Speicherbereiche ("Blackboards").
*   **Feinabstimmung spezialisierter Agenten:** Training von Modellen für spezifische Rollen im Workflow.
*   **Verbesserte Fehlerbehandlung & Resilienz:** Mechanismen zur automatischen Wiederholung oder zur Umleitung bei Agentenfehlern.
*   **Sicherheits-Hardening:** Ersetzen von `eval()` und Implementierung robusterer Validierungen.
*   **Integration weiterer Tools & APIs:** Anbindung an Datenbanken, externe Dienste etc.
*   **Benchmarking & Evaluierung:** Systematische Bewertung der Workflow-Performance für verschiedene Aufgaben und Modelle.

Wenn Sie beitragen möchten, beachten Sie bitte [eventuelle CONTRIBUTION GUIDELINES - ggf. Link einfügen] oder eröffnen Sie ein Issue oder eine Pull Request im Repository [ggf. Link einfügen].

---

## 📄 Lizenz



**Beispiel:**
```
Dieses Projekt ist unter der MIT-Lizenz lizenziert. Die Details finden Sie in der Datei LICENSE.md.
```

---

