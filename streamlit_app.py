# -*- coding: utf-8 -*-
"""
Dieses Script implementiert einen KI-Workflow-Generator und -Runner mit Streamlit.
Es verwendet unter anderem das strukturierte Pattern Matching (PEP 634-636) und den Union-Typ-Operator (PEP 604).
Außerdem wird großer Wert auf klare, einfache und lesbare Strukturen gelegt.
Dieses Skript beinhaltet zusätzlich eine RPM-Funktion, die es erlaubt, die Anfragen an das Gemini Modell
auf eine bestimmte Anzahl pro Minute zu begrenzen.
"""

# Importiere alle notwendigen Bibliotheken
import streamlit as st
import google.genai as genai
from google.genai.types import Part, Tool, GenerateContentConfig, GoogleSearch, FunctionDeclaration, FunctionResponse
from dotenv import load_dotenv
import os
import json
from typing import List, Dict, Any, Callable
import io
from PIL import Image
import datetime
import re
import zipfile
import traceback
import time  # Wird für die RPM-Implementierung benötigt

# --- Konstanten ---
# Lade API-Key aus .env Datei (falls vorhanden)
load_dotenv()
API_KEY = os.getenv("API_KEY")
# Standard-Modell für die Generierung
DEFAULT_MODEL_ID = "gemini-2.0-flash-thinking-exp"  # Alternativ: "gemini-2.0-pro-exp-02-05"
# Spezielle Namen und Dateien für den Generator-Workflow
GENERATOR_WORKFLOW_NAME = "Dynamischer Workflow Generator"
GENERATOR_CONFIG_FILE = "generator_agent_config.json"

# --- RPM Funktionalität ---
def rpm_limiter(func: Callable) -> Callable:
    """
    Decorator, der sicherstellt, dass die dekorierte Funktion nicht öfter als eine bestimmte Anzahl
    von Aufrufen pro Minute ausgeführt wird.
    
    Der maximale Wert wird in st.session_state['rpm_limit'] gespeichert.
    Falls das Limit erreicht wird, wartet die Funktion, bis 60 Sekunden seit dem letzten Reset vergangen sind.
    """
    def wrapper(*args, **kwargs):
        # Hole den aktuellen RPM-Limit-Wert; Standard ist 30 Anfragen pro Minute.
        rpm_limit = st.session_state.get("rpm_limit", 30)
        current_time = time.time()
        # Initialisiere Timer und Aufrufzähler, falls sie noch nicht existieren.
        if "rpm_last_reset" not in st.session_state:
            st.session_state.rpm_last_reset = current_time
            st.session_state.rpm_calls = 0

        # Falls seit dem letzten Reset mehr als 60 Sekunden vergangen sind, setze den Zähler zurück.
        if current_time - st.session_state.rpm_last_reset >= 60:
            st.session_state.rpm_last_reset = current_time
            st.session_state.rpm_calls = 0

        # Wenn das Limit erreicht wurde, berechne die Wartezeit und pausiere.
        if st.session_state.rpm_calls >= rpm_limit:
            wait_time = 60 - (current_time - st.session_state.rpm_last_reset)
            st.warning(f"RPM Limit erreicht. Warte {wait_time:.2f} Sekunden, bis neue Anfragen gesendet werden können.")
            time.sleep(wait_time)
            st.session_state.rpm_last_reset = time.time()
            st.session_state.rpm_calls = 0

        # Erhöhe den Zähler für die aktuellen Anfragen
        st.session_state.rpm_calls += 1
        return func(*args, **kwargs)
    return wrapper

@rpm_limiter
def limited_generate_content(client: genai.Client, model: str, contents: List[Part], config: GenerateContentConfig) -> Any:
    """
    Wrapper um den API-Aufruf an das Gemini Modell zu rate-limiten.
    Verwendet den in st.session_state gesetzten RPM-Wert.
    
    Parameter:
      - client: Instanz des genai.Client.
      - model: Modellbezeichnung als String.
      - contents: Liste von Part-Objekten, die den Input darstellen.
      - config: Konfiguration für die Generierung (z.B. Temperatur).
    
    Rückgabe:
      - Antwort des API-Aufrufs.
    """
    return client.models.generate_content(model=model, contents=contents, config=config)

# --- Hilfsfunktionen (Tools für Agenten) ---
def get_current_datetime() -> str:
    """Gibt das aktuelle Datum und die Uhrzeit im ISO-Format zurück."""
    return datetime.datetime.now().isoformat()

def simple_calculator(expression: str) -> str:
    """
    Berechnet einen einfachen mathematischen Ausdruck (+, -, *, /).
    Sicherheitshinweis: eval() ist hier nur zur Demonstration! In produktiven Systemen
    sollte eine sicherere Methode (z.B. ein dedizierter Parser) verwendet werden,
    um Code-Injection-Risiken zu vermeiden.
    """
    try:
        allowed_chars = "0123456789+-*/.() "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Das Ergebnis von '{expression}' ist {result}"
        else:
            return "Ungültige Zeichen im Ausdruck."
    except Exception as e:
        return f"Fehler bei der Berechnung von '{expression}': {e}"

# --- Tool Registry (Verzeichnis der verfügbaren Tools) ---
AVAILABLE_TOOLS: Dict[str, Callable] = {
    "get_current_datetime": get_current_datetime,
    "calculator": simple_calculator
    # Hier können weitere Tools hinzugefügt werden
}

# --- Konfigurations- und Hilfsfunktionen ---
def load_agent_config(file_path: str, is_generator_config: bool = False) -> List[Dict[str, Any]] | None:
    """
    Lädt eine Agentenkonfiguration aus einer JSON-Datei.
    Sortiert die Agenten nach 'round'. Gibt eine Liste der Agenten-Dictionaries oder None bei Fehlern zurück.
    Zeigt Fehlermeldungen in Streamlit an.
    """
    if not file_path or not os.path.exists(file_path):
        error_key = f"error_config_not_found_{file_path}"
        if error_key not in st.session_state:
             st.error(f"❌ Konfigurationsdatei '{file_path or 'Kein Pfad angegeben'}' nicht gefunden.")
             st.session_state[error_key] = True
        return None

    error_key = f"error_config_not_found_{file_path}"
    if error_key in st.session_state:
        del st.session_state[error_key]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        if not isinstance(config_data, list):
             st.error(f"❌ Fehler: Konfigurationsdatei '{file_path}' enthält keine Liste von Agenten.")
             return None

        if not is_generator_config:
            try:
                config_data.sort(key=lambda x: x.get('round', float('inf')))
            except TypeError:
                st.error(f"❌ Fehler beim Sortieren der Agenten in '{file_path}'. Stellen Sie sicher, dass 'round' eine Zahl ist.")
                return None

        return config_data

    except json.JSONDecodeError as e:
        st.error(f"❌ Fehler beim Parsen der JSON-Datei '{file_path}': {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unbekannter Fehler beim Laden der Konfiguration '{file_path}': {e}")
        st.error(traceback.format_exc())
        return None

def validate_config_list(config_list: List[Dict[str, Any]], source_description: str = "Konfiguration") -> List[Dict[str, Any]] | None:
    """
    Validiert eine Liste von Agenten-Dictionaries mittels Pattern Matching (match-case).
    Prüft auf erforderliche Schlüssel ('name', 'round', 'system_instruction') und deren Typen.
    Gibt die Liste der validen Agenten zurück oder None, wenn keine validen Agenten gefunden wurden.
    """
    if not config_list:
        st.error(f"❌ Fehler: Die übergebene {source_description} ist leer.")
        return None

    valid_config_found = False
    validated_agents: List[Dict[str, Any]] = []
    any_invalid = False

    for agent_dict in config_list:
        if not isinstance(agent_dict, dict):
            st.warning(f"Überspringe Eintrag in '{source_description}', da es kein Dictionary ist: {agent_dict}")
            any_invalid = True
            continue

        match agent_dict:
            case { "name": str(), "round": int(), "system_instruction": str(), **other_keys }:
                validated_agents.append(agent_dict)
                valid_config_found = True
            case _:
                st.warning(f"Agent '{agent_dict.get('name', 'Unbekannt')}' in '{source_description}' hat eine ungültige Struktur oder fehlende/falsche Typen für 'name'(str), 'round'(int) oder 'system_instruction'(str). Agent wird ignoriert.")
                any_invalid = True

    if any_invalid:
         st.error(f"❌ Mindestens ein Agent in '{source_description}' war ungültig und wurde ignoriert.")

    if not valid_config_found:
         st.error(f"❌ Keine validen Agenten in '{source_description}' gefunden.")
         return None

    try:
        validated_agents.sort(key=lambda x: x.get('round', float('inf')))
    except TypeError:
       st.error(f"❌ Fehler beim Sortieren der validierten Agenten aus '{source_description}'.")
       return None

    return validated_agents

def get_grounding_info(candidate: Any) -> str | None:
    """
    Extrahiert Grounding-Informationen (Websuche) aus einem API-Antwort-Kandidaten.
    """
    try:
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding:
            search_entry_point = getattr(grounding, "search_entry_point", None)
            if search_entry_point:
                 uri = getattr(search_entry_point, "uri", None) or getattr(search_entry_point, "rendered_content", None)
                 if uri: return f"Quelle (URI): {uri}"
            web_search_queries = getattr(grounding, "web_search_queries", None)
            if web_search_queries:
                 titles = [q.display_title for q in web_search_queries if hasattr(q, 'display_title') and q.display_title]
                 if titles: return f"Suchanfragen verwendet: {', '.join(titles)}"
                 else: return "Websuche wurde verwendet (keine spezifischen Titel)."
    except AttributeError:
        pass
    except Exception as e:
        st.warning(f"⚠️ Fehler beim Extrahieren der Grounding-Infos: {e}")
    return None

def parse_generator_output(response_text: str) -> tuple[List[Dict[str, Any]] | None, str | None]:
    """
    Versucht, eine JSON-Liste für Agentenkonfigurationen aus dem Antworttext des Generators zu extrahieren.
    Bereinigt übliche LLM-Artefakte wie Markdown-Code-Zäune.
    Gibt ein Tupel zurück: (config_list | None, error_message | None).
    """
    try:
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:].strip()
        elif cleaned_text.startswith("```"):
             cleaned_text = cleaned_text[3:].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3].strip()

        config_list = json.loads(cleaned_text)

        if not isinstance(config_list, list):
            return None, "Generator-Antwort ist keine JSON-Liste."
        if not all(isinstance(item, dict) for item in config_list):
             first_bad_item = next((item for item in config_list if not isinstance(item, dict)), None)
             return None, f"Generator-Liste enthält Elemente, die keine Dictionaries sind (z.B.: {str(first_bad_item)[:100]}...).'"
        return config_list, None

    except json.JSONDecodeError as e:
        error_msg = f"Generator-Antwort ist kein valides JSON: {e}\n\nEmpfangener Text (Anfang):\n```\n{response_text[:500]}...\n```"
        return None, error_msg
    except Exception as e:
         return None, f"Unerwarteter Fehler beim Parsen der Generator-Antwort: {e}"

# --- Hauptfunktion für den Streamlit-Tab ---
def build_tab(api_key: str | None = None):
    """
    Hauptfunktion, die den KI Workflow Generator & Runner in der Streamlit-Weboberfläche aufbaut.
    Hier werden Einstellungen, Dateiupload, Workflow-Auswahl und Agenten-Ausführung realisiert.
    """
    st.set_page_config(page_title="KI Workflow Generator & Runner", layout="wide")
    api_key = api_key or API_KEY
    if not api_key:
        st.error("❌ Kein API-Key gefunden.")
        return

    st.title("🤖 KI Workflow Generator & Runner")
    st.markdown("Wähle einen vordefinierten Workflow ODER lass die KI einen Workflow für deine Aufgabe generieren!")

    # --- Workflow-Auswahl ---
    supported_workflows = {
        GENERATOR_WORKFLOW_NAME: GENERATOR_CONFIG_FILE,
        "Python Aufgabe": "agents_config_python.json",
        "C++ Aufgabe": "agents_config_cpp.json",
        "Java Aufgabe": "agents_config_java.json",
        "JavaScript Aufgabe": "agents_config_javascript.json",
        "Python Plugin Entwickler": "plugin_developer_config.json"
    }
    if 'selected_workflow' not in st.session_state:
        st.session_state.selected_workflow = list(supported_workflows.keys())[0]
    selected_workflow_name = st.selectbox("Wähle den Workflow:", options=list(supported_workflows.keys()), key="selected_workflow")
    agent_config_file_path = supported_workflows.get(selected_workflow_name)
    is_generator_mode = (selected_workflow_name == GENERATOR_WORKFLOW_NAME)

    # --- Konfiguration laden (für Sidebar) ---
    config_for_sidebar = None
    if agent_config_file_path:
        config_for_sidebar = load_agent_config(agent_config_file_path)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Einstellungen & Infos")
        st.info(f"Modus: **{selected_workflow_name}**")
        if agent_config_file_path:
            st.markdown(f"Konfig: `{agent_config_file_path}`")
            if config_for_sidebar is not None:
                validated_sidebar_config = validate_config_list(config_for_sidebar, f"'{agent_config_file_path}' (für Sidebar)")
                if validated_sidebar_config:
                    st.success(f"✅ Geladen ({len(validated_sidebar_config)} Agenten).")
                    if not is_generator_mode:
                         st.subheader("Agentenübersicht:")
                         agent_summary = [{"R": a.get("round"), "Name": a.get("name"), "Desc": a.get("description", "-")} for a in validated_sidebar_config]
                         st.dataframe(agent_summary, use_container_width=True, hide_index=True, column_config={"R": st.column_config.NumberColumn(width="small"), "Name": st.column_config.TextColumn(width="medium"), "Desc": st.column_config.TextColumn(width="large")})
                         with st.expander("Vollständige JSON"):
                             st.json(validated_sidebar_config)
                else:
                    st.error("❌ Konfig für Sidebar ungültig.")
            else:
                pass
        else:
            st.warning("Kein Konfigurationspfad definiert.")
        st.divider()
        st.subheader("Tools (für Agenten)")
        st.json(list(AVAILABLE_TOOLS.keys()))
        model_id = DEFAULT_MODEL_ID
        st.caption(f"Modell: `{model_id}`")
        
        # --- RPM Einstellungen in der Sidebar ---
        st.divider()
        st.subheader("RPM Einstellungen")
        rpm_limit = st.number_input("Maximale Anfragen pro Minute (RPM):", min_value=1, max_value=120, value=30, step=1, key="rpm_limit_input")
        st.session_state.rpm_limit = rpm_limit

    # --- Session State initialisieren ---
    if 'message_store' not in st.session_state:
        st.session_state.message_store = {}
    if 'agent_results_display' not in st.session_state:
        st.session_state.agent_results_display = []
    if 'last_question_processed' not in st.session_state:
        st.session_state.last_question_processed = ""
    if 'uploaded_files_data' not in st.session_state:
        st.session_state.uploaded_files_data = []
    if 'last_workflow_processed' not in st.session_state:
        st.session_state.last_workflow_processed = ""

    # --- Eingabe ---
    question_label = f"📝 Aufgabe für '{selected_workflow_name}':"
    if is_generator_mode:
        question_label = "📝 Ziel für Workflow-Generierung:"
    question = st.text_area(question_label, key="task_description")
    uploaded_files = st.file_uploader("📎 Dateien hochladen (Kontext):", type=["png", "jpg", "jpeg", "webp", "txt", "py", "md", "csv", "json", "html", "css", "js", "yaml", "sh", "java", "cpp", "h", "cs", "go", "rb", "php"], accept_multiple_files=True, key="file_uploader")

    # --- Datei-Verarbeitung und Anzeige ---
    current_uploaded_files_data = []
    if uploaded_files:
        st.write("Neu hochgeladene Dateien:")
        for uploaded_file in uploaded_files:
            try:
                file_bytes = uploaded_file.getvalue()
                file_data = {"name": uploaded_file.name, "type": uploaded_file.type, "bytes": file_bytes}
                current_uploaded_files_data.append(file_data)
                if uploaded_file.type.startswith("image/"):
                    st.image(file_bytes, caption=f"{uploaded_file.name}", width=100)
                else:
                    st.caption(f"- `{uploaded_file.name}` ({uploaded_file.type})")
            except Exception as file_e:
                 st.error(f"Fehler beim Verarbeiten der Datei '{uploaded_file.name}': {file_e}")
        st.session_state.uploaded_files_data = current_uploaded_files_data
        st.success(f"{len(current_uploaded_files_data)} Datei(en) bereit.")
    elif st.session_state.uploaded_files_data:
         st.write(f"Vorhandene Dateien ({len(st.session_state.uploaded_files_data)}):")
         with st.expander("Dateien verwalten", expanded=False):
             indices_to_remove = []
             for idx in range(len(st.session_state.uploaded_files_data) -1, -1, -1):
                 file_data = st.session_state.uploaded_files_data[idx]
                 col1, col2 = st.columns([0.8, 0.2])
                 with col1:
                    if file_data["type"].startswith("image/"):
                        st.image(file_data["bytes"], caption=f"{file_data['name']}", width=100)
                    else:
                        st.caption(f"- `{file_data['name']}` ({file_data['type']})")
                 with col2:
                    if st.button(f"❌", key=f"remove_file_{file_data['name']}_{idx}", help=f"'{file_data['name']}' entfernen"):
                         indices_to_remove.append(idx)
             if indices_to_remove:
                 indices_to_remove.sort(reverse=True)
                 for index in indices_to_remove:
                     st.session_state.uploaded_files_data.pop(index)
                 st.rerun()
             if not st.session_state.uploaded_files_data:
                 st.info("Keine Dateien vorhanden.")

    # --- Start-Button ---
    button_label = f"🚀 '{selected_workflow_name}'-Workflow starten"
    if is_generator_mode:
        button_label = "🧬 Workflow generieren & ausführen"
    if st.button(button_label, key="start_button"):

        if not question and not st.session_state.uploaded_files_data:
            st.warning("Bitte Aufgabe beschreiben oder Dateien hochladen.")
            st.stop()

        st.session_state.message_store = {}
        st.session_state.agent_results_display = []
        st.session_state.last_question_processed = question
        st.session_state.last_workflow_processed = selected_workflow_name

        results_placeholder = st.empty()
        final_agents_config = None
        prompt_for_execution = question

        # --- Phase 1: Workflow-Generierung ---
        if is_generator_mode:
            with st.spinner("🧠 Workflow-Generator arbeitet..."):
                generator_config_list = load_agent_config(GENERATOR_CONFIG_FILE, is_generator_config=True)
                generator_config = validate_config_list(generator_config_list, f"'{GENERATOR_CONFIG_FILE}'") if generator_config_list else None
                if generator_config is None or not generator_config:
                    st.error("Generator-Konfig ungültig.")
                    st.stop()

                generator_input_parts: List[Part] = []
                generator_agent_conf = generator_config[0]
                generator_input_parts.append(Part(text=f"System Anweisung ({generator_agent_conf.get('name')}):\n{generator_agent_conf.get('system_instruction')}\n---"))
                generator_input_parts.append(Part(text=f"Nutzeranfrage/Ziel:\n{question}"))
                if st.session_state.uploaded_files_data:
                    generator_input_parts.append(Part(text="\n\n--- START KONTEXT DATEIEN ---"))
                    for file_data in st.session_state.uploaded_files_data:
                        file_name, file_type, file_bytes = file_data["name"], file_data["type"], file_data["bytes"]
                        if file_type.startswith("image/"):
                            image_part = Part(inline_data={"mime_type": file_type, "data": file_bytes})
                            generator_input_parts.append(Part(text=f"\nBild: `{file_name}`"))
                            generator_input_parts.append(image_part)
                        else:
                            try:
                                file_content = file_bytes.decode('utf-8')
                            except Exception as decode_e:
                                file_content = f"[Fehler beim Dekodieren: {decode_e}]"
                                st.warning(f"Datei '{file_name}' ({file_type}) ignoriert.")
                            file_text_part = Part(text=(f"\n--- START DATEI: `{file_name}` ---\n{file_content}\n--- ENDE DATEI: `{file_name}` ---"))
                            generator_input_parts.append(file_text_part)
                    generator_input_parts.append(Part(text="\n--- ENDE KONTEXT DATEIEN ---"))

                generator_output = "[Generator nicht geantwortet]"
                generator_success = False
                try:
                    client = genai.Client(api_key=api_key)
                    response = limited_generate_content(
                        client=client,
                        model=f"models/{model_id}",
                        contents=generator_input_parts,
                        config=GenerateContentConfig(temperature=generator_agent_conf.get("temperature", 0.5))
                    )
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        generator_output = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
                        if generator_output:
                            generator_success = True
                        else:
                            generator_output = "[Generator gab leere Antwort]"
                    else:
                        feedback = getattr(response, 'prompt_feedback', None)
                        block_reason = getattr(feedback, 'block_reason', "Unbekannt") if feedback else "Unbekannt"
                        block_msg = getattr(feedback, 'block_reason_message', "Keine Details") if feedback else "Keine Details"
                        generator_output = f"[Fehler: Generator-Antwort ungültig. Grund: {block_reason}. Nachricht: {block_msg}]"
                        st.error(f"Generator-Fehler: {generator_output}")
                except Exception as gen_e:
                    st.error(f"❌ Kritischer Generator-Fehler: {gen_e}")
                    st.error(traceback.format_exc())
                    st.stop()

                st.session_state.agent_results_display.append({
                    "agent": generator_agent_conf.get("name"),
                    "status": "Erfolgreich" if generator_success else "Fehlgeschlagen",
                    "output": generator_output,
                    "sources": None,
                    "details": "Output des Workflow Generators"
                })
                if not generator_success:
                    st.error("Workflow-Generierung fehlgeschlagen.")
                    st.stop()

                generated_config_list, parse_error = parse_generator_output(generator_output)
                if parse_error:
                    st.error("Fehler Parsen Generator-Antwort:")
                    st.error(parse_error)
                    st.code(generator_output, language='text')
                    st.stop()
                if not generated_config_list:
                    st.error("Generator gab leere Konfig zurück.")
                    st.code(generator_output, language='text')
                    st.stop()

                validated_generated_config = validate_config_list(generated_config_list, "generierter Konfiguration")
                if validated_generated_config is None:
                    st.error("Generierte Konfiguration ungültig. Prozess gestoppt.")
                    st.subheader("Fehlerhafter JSON-Output des Generators:")
                    st.code(generator_output, language="json")
                    st.stop()
                else:
                    final_agents_config = validated_generated_config
                    st.success(f"✅ Workflow mit {len(final_agents_config)} Agenten generiert!")
                    st.sidebar.subheader("Dynamisch generierte Agenten:")
                    gen_agent_summary = [{"R": a.get("round"), "Name": a.get("name"), "Desc": a.get("description", "-")} for a in final_agents_config]
                    st.sidebar.dataframe(gen_agent_summary, use_container_width=True, hide_index=True,
                                         column_config={"R": st.column_config.NumberColumn(width="small"),
                                                        "Name": st.column_config.TextColumn(width="medium"),
                                                        "Desc": st.column_config.TextColumn(width="large")})
                    with st.sidebar.expander("Generierte JSON"):
                        st.json(final_agents_config)
                    results_placeholder.info("Führe generierten Workflow aus...")

        # --- Phase 2: Ausführung (Normaler Modus) ---
        else:
            config_to_validate = load_agent_config(agent_config_file_path)
            if config_to_validate:
                final_agents_config = validate_config_list(config_to_validate, f"'{agent_config_file_path}'")
            if final_agents_config is None:
                st.error(f"Vordefinierte Konfig für '{selected_workflow_name}' ungültig/nicht geladen.")
                st.stop()
            results_placeholder.info(f"Führe Workflow '{selected_workflow_name}' aus...")

        # --- Haupt-Agenten-Ausführung ---
        if final_agents_config:
            with st.spinner(f"Agenten arbeiten..."):
                try:
                    client = genai.Client(api_key=api_key)
                    overall_success = True
                    for agent_index, agent_conf in enumerate(final_agents_config):
                        agent_name = agent_conf.get("name", f"Agent_{agent_index+1}")
                        system_instruction = agent_conf.get("system_instruction", "-")
                        enable_web_search = agent_conf.get("enable_web_search", False)
                        temperature = agent_conf.get("temperature")
                        receives_from = agent_conf.get("receives_messages_from", [])
                        accepts_files = agent_conf.get("accepts_files", False)
                        callable_tool_names = agent_conf.get("callable_tools", [])
                        results_placeholder.info(f"🧠 Agent: **{agent_name}** ({agent_index + 1}/{len(final_agents_config)})...")
                        all_sources_found = True

                        # 1. Input vorbereiten
                        current_input_parts: List[Part] = []
                        current_input_parts.append(Part(text=f"System Anweisung ({selected_workflow_name} - Rolle: {agent_name}):\n{system_instruction}\n---"))
                        is_first_relevant_agent = not receives_from or all(source not in st.session_state.message_store for source in receives_from)
                        if is_first_relevant_agent:
                            current_input_parts.append(Part(text=f"Nutzeranfrage:\n{prompt_for_execution}"))
                            if accepts_files and st.session_state.uploaded_files_data:
                                current_input_parts.append(Part(text="\n\n--- START KONTEXT DATEIEN ---"))
                                for file_data in st.session_state.uploaded_files_data:
                                    file_name, file_type, file_bytes = file_data["name"], file_data["type"], file_data["bytes"]
                                    if file_type.startswith("image/"):
                                        image_part = Part(inline_data={"mime_type": file_type, "data": file_bytes})
                                        current_input_parts.append(Part(text=f"\nBild: `{file_name}`"))
                                        current_input_parts.append(image_part)
                                    else:
                                        try:
                                            file_content = file_bytes.decode('utf-8')
                                        except Exception as decode_e:
                                            st.warning(f"Datei '{file_name}' ({file_type}) ignoriert (Decode-Fehler): {decode_e}")
                                            file_content = "[Inhalt nicht lesbar/dekodierbar]"
                                        file_text_part = Part(text=(f"\n--- START DATEI: `{file_name}` ---\n{file_content}\n--- ENDE DATEI: `{file_name}` ---"))
                                        current_input_parts.append(file_text_part)
                                current_input_parts.append(Part(text="\n--- ENDE KONTEXT DATEIEN ---"))
                        else:
                            previous_outputs_text = []
                            for source_agent_name in receives_from:
                                if source_agent_name in st.session_state.message_store:
                                    previous_outputs_text.append(f"Ergebnis '{source_agent_name}':\n{st.session_state.message_store[source_agent_name]}")
                                else:
                                    error_msg = f"Input von '{source_agent_name}' fehlt. Überspringe '{agent_name}'."
                                    st.warning(error_msg)
                                    st.session_state.agent_results_display.append({"agent": agent_name, "status": "Übersprungen", "details": error_msg, "output": "[Input fehlt]"})
                                    all_sources_found = False
                                    break
                            if not all_sources_found:
                                overall_success = False
                                continue
                            input_from_previous = "\n\n---\n\n".join(previous_outputs_text)
                            current_input_parts.append(Part(text=f"Vorherige Ergebnisse:\n{input_from_previous}\n---\nDeine Aufgabe:"))

                        # 2. Konfiguration der Tools
                        agent_tools_list = []
                        current_agent_func_declarations = []
                        if enable_web_search:
                            agent_tools_list.append(Tool(google_search=GoogleSearch()))
                        if callable_tool_names:
                            for tool_name in callable_tool_names:
                                if tool_name in AVAILABLE_TOOLS:
                                    func = AVAILABLE_TOOLS[tool_name]
                                    description = func.__doc__.splitlines()[0] if func.__doc__ else f"Tool: {tool_name}"
                                    params_schema = {}
                                    if tool_name == "calculator":
                                        params_schema = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
                                    elif tool_name == "get_current_datetime":
                                        params_schema = {"type": "object", "properties": {}}
                                    current_agent_func_declarations.append(FunctionDeclaration(name=tool_name, description=description, parameters=params_schema))
                                else:
                                    st.warning(f"Tool '{tool_name}' für '{agent_name}' nicht in AVAILABLE_TOOLS.")
                            if current_agent_func_declarations:
                                agent_tools_list.append(Tool(function_declarations=current_agent_func_declarations))
                        gen_config_args = {"response_modalities": ["TEXT"]}
                        if agent_tools_list:
                            gen_config_args["tools"] = agent_tools_list
                        if temperature is not None:
                            gen_config_args["temperature"] = temperature
                        agent_specific_config = GenerateContentConfig(**gen_config_args)

                        # 3. API-Aufruf / Function Calling Loop
                        max_function_calls = 5
                        call_count = 0
                        final_agent_output = ""
                        agent_success_flag = False
                        grounding_info = None
                        conversation_history = list(current_input_parts)
                        should_skip = (agent_conf.get("name", "").startswith("Planner") and accepts_files and not st.session_state.uploaded_files_data and not question.strip())

                        while call_count < max_function_calls:
                            if should_skip:
                                st.info(f"'{agent_name}' übersprungen.")
                                st.session_state.agent_results_display.append({"agent": agent_name, "status": "Übersprungen", "output": "[Keine Frage/Dateien]"})
                                agent_success_flag = True
                                final_agent_output = "[Keine Frage/Dateien]"
                                break
                            try:
                                effective_model_for_call = f"models/{model_id}"
                                response = limited_generate_content(
                                    client=client,
                                    model=effective_model_for_call,
                                    contents=conversation_history,
                                    config=agent_specific_config
                                )
                                candidate = response.candidates[0] if response.candidates else None
                                function_call = None
                                if candidate and hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    first_part = candidate.content.parts[0]
                                if hasattr(first_part, 'function_call'):
                                    function_call = first_part.function_call
                                if function_call and hasattr(function_call, 'name'):
                                    tool_name = function_call.name
                                    tool_args = dict(function_call.args) if hasattr(function_call, 'args') else {}
                                    st.info(f"'{agent_name}' -> Tool `{tool_name}`...")
                                    if tool_name in AVAILABLE_TOOLS:
                                        tool_function = AVAILABLE_TOOLS[tool_name]
                                        try:
                                            function_result = tool_function(**tool_args)
                                            st.success(f"Tool `{tool_name}` OK.")
                                            function_response_part = Part(function_response=FunctionResponse(name=tool_name, response={"content": str(function_result)}))
                                            conversation_history.append(first_part)
                                            conversation_history.append(function_response_part)
                                            call_count += 1
                                            continue
                                        except Exception as func_exc:
                                            st.error(f"Tool `{tool_name}` Fehler: {func_exc}")
                                            final_agent_output = f"[Tool Fehler {tool_name}: {func_exc}]"
                                            agent_success_flag = False
                                            break
                                    else:
                                        st.error(f"Unbekanntes Tool `{tool_name}`")
                                        final_agent_output = f"[Unbekanntes Tool {tool_name}]"
                                        agent_success_flag = False
                                        break
                                else:
                                    if candidate and hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                                        text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                                        if text_parts:
                                            final_agent_output = "\n".join(text_parts).strip()
                                            grounding_info = get_grounding_info(candidate)
                                            agent_success_flag = True
                                        else:
                                            feedback = getattr(response, 'prompt_feedback', None)
                                            block_reason = getattr(feedback, 'block_reason', "Unbekannt") if feedback else "Unbekannt"
                                            block_msg = getattr(feedback, 'block_reason_message', "Keine Details") if feedback else "Keine Details"
                                            final_agent_output = f"[Fehler: Keine gültige Antwort. Grund: {block_reason}. Nachricht: {block_msg}]"
                                            st.error(f"'{agent_name}': {final_agent_output}")
                                            agent_success_flag = False
                                    else:
                                        feedback = getattr(response, 'prompt_feedback', None)
                                        block_reason = getattr(feedback, 'block_reason', "Unbekannt") if feedback else "Unbekannt"
                                        block_msg = getattr(feedback, 'block_reason_message', "Keine Details") if feedback else "Keine Details"
                                        final_agent_output = f"[Fehler: Keine gültige Antwort. Grund: {block_reason}. Nachricht: {block_msg}]"
                                        st.error(f"'{agent_name}': {final_agent_output}")
                                        agent_success_flag = False
                                    break
                            except Exception as e:
                                st.error(f"❌ Kritischer Fehler bei '{agent_name}': {e}")
                                st.error(traceback.format_exc())
                                final_agent_output = f"[Kritischer Fehler: {e}]"
                                agent_success_flag = False
                                overall_success = False
                                break

                        if call_count >= max_function_calls:
                            st.warning(f"Agent '{agent_name}' hat Limit für Funktionsaufrufe ({max_function_calls}) erreicht.")
                            if not final_agent_output:
                                final_agent_output = "[Limit erreicht]"
                            agent_success_flag = False

                        if final_agent_output:
                            if agent_success_flag and "[Keine Frage/Dateien]" not in final_agent_output:
                                st.session_state.message_store[agent_name] = final_agent_output
                            already_skipped = any(r['agent'] == agent_name and r['status'] == 'Übersprungen' and "[Fehlender Input]" in r.get('output','') for r in st.session_state.agent_results_display)
                            if not already_skipped:
                                current_status = "Erfolgreich" if agent_success_flag else "Fehlgeschlagen"
                                if "[Übersprungen" in final_agent_output or "[Keine Frage/Dateien]" in final_agent_output:
                                    current_status = "Übersprungen"
                                st.session_state.agent_results_display.append({
                                    "agent": agent_name,
                                    "status": current_status,
                                    "output": final_agent_output,
                                    "sources": grounding_info,
                                    "details": None
                                })
                        if not agent_success_flag and not (should_skip or "[Keine Frage/Dateien]" in final_agent_output):
                            overall_success = False

                except Exception as e:
                     st.error(f"❌ Unerwarteter Fehler im Hauptprozess: {e}")
                     st.error(traceback.format_exc())
                     overall_success = False

            results_placeholder.empty()
            st.markdown("---")
            if not st.session_state.agent_results_display:
                st.warning("Keine Agenten ausgeführt.")
            elif overall_success:
                st.success("✅ Workflow erfolgreich abgeschlossen.")
            else:
                st.error("❌ Workflow mit Fehlern/Warnungen abgeschlossen.")

            st.subheader("Ergebnisse der einzelnen Agenten:")
            displayed_agents = set()
            for result in st.session_state.agent_results_display:
                agent_name = result.get('agent', 'Unbekannter Agent')
                if agent_name in displayed_agents:
                    continue
                displayed_agents.add(agent_name)
                status = result.get('status', 'Unbekannt')
                output = result.get('output', '[Kein Output]')
                sources = result.get('sources')
                details = result.get('details')
                status_icon = '❓'
                if status == 'Erfolgreich':
                    status_icon = '✅'
                elif status == 'Fehlgeschlagen':
                    status_icon = '❌'
                elif status in ['Warnung', 'Übersprungen']:
                    status_icon = '⚠️'
                expander_title = f"{status_icon} Agent: **{agent_name}** ({status})"
                expand_default = (status != 'Übersprungen')
                with st.expander(expander_title, expanded=expand_default):
                    st.markdown("##### Output:")
                    is_likely_code_output = "```" in output or (status == 'Erfolgreich' and any(kw in agent_name.lower() for kw in ["coder", "architect", "refiner"]))
                    if agent_name == "WorkflowGenerator" and status == 'Erfolgreich':
                        st.code(output, language="json")
                    elif is_likely_code_output and status == 'Erfolgreich':
                        lang_match = re.search(r"```(\w+)", output)
                        lang_name = selected_workflow_name.split()[0].lower().replace("plugin", "python")
                        lang = lang_match.group(1) if lang_match else lang_name
                        code_content = re.sub(r"```\w*\n?", "", output, count=1)
                        code_content = re.sub(r"\n?```$", "", code_content)
                        st.code(code_content.strip(), language=lang, line_numbers=True)
                    else:
                        st.markdown(output)
                    if sources:
                        st.markdown("##### Quellen/Infos:")
                        st.caption(f"{sources}")
                    if details:
                        st.info(f"Details: {details}")

            st.markdown("---")
            st.subheader("📦 Download generierter Dateien")
            project_files = {}
            file_pattern = re.compile(r"## FILE: \s*([\w\.\-\/]+\.\w+)\s*\n```(?:[\w\+\#\-\.]*\n)?(.*?)```", re.DOTALL | re.MULTILINE)
            for result in st.session_state.agent_results_display:
                if result.get("agent") != "WorkflowGenerator" and result.get("status") == "Erfolgreich" and result.get("output"):
                    matches = file_pattern.findall(result["output"])
                    for filename, content in matches:
                        project_files[filename.strip()] = content.strip() + "\n"
            if project_files:
                st.write(f"Generierte Dateien für **'{selected_workflow_name}'**:")
                st.markdown("\n".join([f"- `{fname}`" for fname in sorted(project_files.keys())]))
                zip_buffer = io.BytesIO()
                try:
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_f:
                        for filename, content in project_files.items():
                            zip_f.writestr(filename, content.encode('utf-8'))
                    zip_bytes = zip_buffer.getvalue()
                    download_filename = f"{selected_workflow_name.lower().replace(' ','_')}_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.zip"
                    st.download_button(label=f"⬇️ '{selected_workflow_name}' Ergebnisse als ZIP", data=zip_bytes, file_name=download_filename, mime="application/zip", key="download_zip_button")
                except Exception as zip_e:
                    st.error(f"Fehler beim Zippen: {zip_e}")
                    st.error(traceback.format_exc())
            else:
                st.info("Keine Dateien zum Zippen gefunden.")

            st.markdown("---")
            st.subheader("🏁 Finales Text-Ergebnis")
            final_successful_output = "[Kein spezifisches textuelles Endergebnis gefunden]"
            final_agent_name = None
            for res in reversed(st.session_state.agent_results_display):
                output = res.get("output", "")
                status = res.get("status")
                agent = res.get("agent")
                if agent != "WorkflowGenerator" and status == "Erfolgreich" and output and "[Kein Output]" not in output and "[Keine Frage/Dateien]" not in output:
                    is_likely_just_files = file_pattern.fullmatch(output.strip()) is not None or output.strip().startswith("## FILE:")
                    if (not is_likely_just_files or any(kw in agent for kw in ["Planner", "Reviewer", "Packager"])):
                        final_successful_output = output
                        final_agent_name = agent
                        break
                    elif final_agent_name is None:
                        final_successful_output = f"[Letzter Output war Code von Agent '{agent}']"
                        final_agent_name = agent
            final_title_suffix = f"(von Agent: **{final_agent_name}**)" if final_agent_name else ""
            st.markdown(f"**{final_title_suffix}**")
            if "[Letzter Output war Code von Agent" in final_successful_output:
                st.info(final_successful_output)
            else:
                st.markdown(final_successful_output)

# --- Hauptausführungspunkt ---
if __name__ == "__main__":
    build_tab()
