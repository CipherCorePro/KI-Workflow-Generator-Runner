# -*- coding: utf-8 -*-
"""
Dieses Script implementiert einen KI-Workflow-Generator und -Runner mit Streamlit.

"""

# Importiere alle notwendigen Bibliotheken
import streamlit as st
import google.genai as genai
from google.genai.types import Part, Tool, GenerateContentConfig, GoogleSearch, FunctionDeclaration, FunctionResponse
# === ZUSÄTZLICHE IMPORTS NUR FÜR DECORATOR ===
import time
from typing import List, Dict, Any, Callable, Optional, Tuple
# === ENDE ZUSÄTZLICHE IMPORTS ===
from dotenv import load_dotenv
import os
import json
import io
from PIL import Image
import datetime
import re
import zipfile
import traceback

# --- Konstanten ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    st.error("❌ API_KEY nicht gefunden. Bitte in .env setzen.")
    st.stop()
DEFAULT_MODEL_ID = "gemini-2.5-pro-exp-03-25" # Angepasst, Original: "gemini-2.0-flash-thinking-exp-01-21"
GENERATOR_WORKFLOW_NAME = "Dynamischer Workflow Generator"
GENERATOR_CONFIG_FILE = "generator_agent_config.json"

# === HINZUGEFÜGT: RPM Funktionalität  ===
def rpm_limiter(func: Callable) -> Callable:
    """
    Decorator für RPM-Limitierung. Liest Limit aus st.session_state['rpm_limit_input'].
    """
    def wrapper(*args, **kwargs):
        rpm_limit = st.session_state.get("rpm_limit_input", 10) # Key aus Sidebar verwenden
        current_time = time.time()
        if "rpm_last_reset" not in st.session_state:
            st.session_state.rpm_last_reset = current_time
            st.session_state.rpm_calls = 0

        if current_time - st.session_state.rpm_last_reset >= 60:
            st.session_state.rpm_last_reset = current_time
            st.session_state.rpm_calls = 0
            try: st.sidebar.caption(f"RPM Reset @ {datetime.datetime.now():%H:%M:%S}")
            except: pass # Fehler ignorieren, falls Sidebar nicht mehr da ist

        if st.session_state.rpm_calls >= rpm_limit:
            wait_time = 60.0 - (current_time - st.session_state.rpm_last_reset)
            if wait_time > 0:
                 st.warning(f"🚦 RPM Limit ({rpm_limit}/min) erreicht. Warte {wait_time:.1f} Sekunden...")
                 time.sleep(wait_time)
                 st.session_state.rpm_last_reset = time.time()
                 st.session_state.rpm_calls = 0
                 try: st.sidebar.caption(f"RPM Reset @ {datetime.datetime.now():%H:%M:%S}")
                 except: pass

        # Rufe die eigentliche Funktion auf
        # WICHTIG: Fehler innerhalb der Funktion werden hier NICHT abgefangen,
        # das muss außerhalb im aufrufenden Code geschehen!
        result = func(*args, **kwargs)

        st.session_state.rpm_calls += 1
        if "rpm_limit_input" in st.session_state:
            try: st.sidebar.caption(f"RPM: {st.session_state.rpm_calls}/{rpm_limit} (Reset in {max(0, 60 - (time.time() - st.session_state.rpm_last_reset)):.0f}s)")
            except: pass
        return result
    return wrapper

@rpm_limiter
def limited_generate_content(client: genai.Client, model: str, contents: List[Part], config: GenerateContentConfig) -> Any:
    """
    Wrapper um client.models.generate_content, um den @rpm_limiter anzuwenden.
    Parameter und Rückgabe wie client.models.generate_content.
    """
    # Der eigentliche API-Call, der durch den Decorator begrenzt wird
    return client.models.generate_content(model=model, contents=contents, config=config)
# === ENDE RPM Funktionalität ===


# --- Hilfsfunktionen (Tools für Agenten ) ---
def get_current_datetime() -> str:
    """Gibt das aktuelle Datum und die Uhrzeit im ISO-Format zurück."""
    return datetime.datetime.now().isoformat()

def simple_calculator(expression: str) -> str:
    """Berechnet einfachen mathematischen Ausdruck (+, -, *, /). Unsicher!"""
    try:
        allowed_chars = "0123456789+-*/.() "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Das Ergebnis von '{expression}' ist {result}"
        else:
            return "Ungültige Zeichen im Ausdruck."
    except Exception as e:
        return f"Fehler bei der Berechnung von '{expression}': {e}"

# --- Tool Registry  ---
AVAILABLE_TOOLS: Dict[str, Callable] = {
    "get_current_datetime": get_current_datetime,
    "calculator": simple_calculator
}

# --- Konfigurations- und Hilfsfunktionen , Typ-Hinweise ---
def load_agent_config(file_path: str, is_generator_config: bool = False) -> Optional[List[Dict[str, Any]]]:
    """Lädt Agentenkonfiguration aus JSON."""
    if not file_path or not os.path.exists(file_path):
        error_key = f"error_config_nf_{file_path}";
        if error_key not in st.session_state: st.error(f"❌ Konfig '{file_path or '?'}' nicht gefunden."); st.session_state[error_key] = True
        return None
    error_key = f"error_config_nf_{file_path}";
    if error_key in st.session_state: del st.session_state[error_key]
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        if not isinstance(data, list): st.error(f"❌ Konfig '{file_path}' ist keine Liste."); return None
        if not is_generator_config:
            try: data.sort(key=lambda x: x.get('round', float('inf')))
            except TypeError: st.error(f"❌ Sortierfehler '{file_path}'. 'round' muss Zahl sein."); return None
        return data
    except json.JSONDecodeError as e: st.error(f"❌ JSON-Fehler in '{file_path}': {e}"); return None
    except Exception as e: st.error(f"❌ Ladefehler Konfig '{file_path}': {e}"); st.error(traceback.format_exc()); return None

def validate_config_list(config_list: List[Dict[str, Any]], source_desc: str = "Konfiguration") -> Optional[List[Dict[str, Any]]]:
    """Validiert Agenten-Struktur und Typen - aus Original."""
    if not config_list: st.error(f"❌ Fehler: {source_desc} ist leer."); return None
    valid_agents: List[Dict[str, Any]] = []; invalid = False; valid_found = False
    for i, agent_dict in enumerate(config_list):
        if not isinstance(agent_dict, dict): st.warning(f"Überspringe #{i+1} in '{source_desc}' (kein dict)."); invalid = True; continue
        match agent_dict:
            case { "name": str(), "round": int(), "system_instruction": str(), **other_keys }: # Original Pattern
                valid_agents.append(agent_dict); valid_found = True
            case _:
                st.warning(f"Agent '{agent_dict.get('name', f'Unbekannt #{i+1}')}' hat ungültige Struktur/Typen ('name':str, 'round':int, 'system_instruction':str). Ignoriert.")
                invalid = True
    if invalid: st.warning(f"⚠️ Mindestens ein Agent in '{source_desc}' war ungültig.")
    if not valid_found: st.error(f"❌ Keine validen Agenten in '{source_desc}'."); return None
    try: valid_agents.sort(key=lambda x: x.get('round', float('inf')))
    except TypeError: st.error(f"❌ Sortierfehler validierte Agenten.")
    return valid_agents

def get_grounding_info(candidate: Any) -> Optional[str]:
    """Extrahiert Grounding-Infos (Websuche) - aus Original."""
    try:
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding:
            search_entry = getattr(grounding, "search_entry_point", None)
            if search_entry:
                 uri = getattr(search_entry, "uri", None) or getattr(search_entry, "rendered_content", None)
                 if uri: return f"Quelle (URI): {uri}"
            queries = getattr(grounding, "web_search_queries", [])
            if queries:
                 titles = [q.display_title for q in queries if hasattr(q, 'display_title') and q.display_title]
                 if titles: return f"Suchanfragen verwendet: {', '.join(titles)}"
                 else: return "Websuche wurde verwendet (keine spezifischen Titel)."
    except AttributeError: pass
    except Exception as e: st.warning(f"⚠️ Fehler Grounding-Infos: {e}")
    return None

def parse_generator_output(response_text: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Parst Generator-Output (JSON-Liste von Agenten) - aus Original."""
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        elif cleaned.startswith("```"): cleaned = cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if not cleaned: return None, "Generator-Antwort war nach Bereinigung leer."
        config_list = json.loads(cleaned)
        if not isinstance(config_list, list): return None, "Generator-Antwort ist keine JSON-Liste."
        if not all(isinstance(item, dict) for item in config_list):
             bad = next((item for item in config_list if not isinstance(item, dict)), None)
             return None, f"Enthält Nicht-Dictionaries (z.B.: {str(bad)[:100]}...).'"
        return config_list, None
    except json.JSONDecodeError as e:
        error_msg = f"Generator-Antwort ist kein valides JSON: {e}\n\nEmpfangen (Anfang):\n```\n{response_text[:500]}...\n```"
        return None, error_msg
    except Exception as e: return None, f"Unerwarteter Fehler beim Parsen der Generator-Antwort: {e}"

# --- Hauptfunktion für den Streamlit-Tab ---
def build_tab():
    """Baut die Streamlit UI und steuert den Workflow."""
    st.set_page_config(page_title="KI Workflow Generator & Runner", layout="wide")

    st.title("🤖 KI Workflow Generator & Runner")
    st.markdown("Wähle Workflow ODER lass KI Workflow generieren!")

    supported_workflows = { name: file for name, file in [
        (GENERATOR_WORKFLOW_NAME, GENERATOR_CONFIG_FILE),
        ("Python Aufgabe", "agents_config_python.json"), ("C++ Aufgabe", "agents_config_cpp.json"),
        ("Java Aufgabe", "agents_config_java.json"), ("JavaScript Aufgabe", "agents_config_javascript.json"),
        ("Python Plugin Entwickler", "plugin_developer_config.json") ] if os.path.exists(file) }
    if not supported_workflows: st.error("Keine Konfigurationsdateien gefunden!"); st.stop()

    if 'selected_workflow' not in st.session_state or st.session_state.selected_workflow not in supported_workflows:
        st.session_state.selected_workflow = list(supported_workflows.keys())[0]
    selected_workflow_name = st.selectbox("Wähle Workflow:", options=list(supported_workflows.keys()), key="selected_workflow")
    agent_config_file_path = supported_workflows[selected_workflow_name]
    is_generator_mode = (selected_workflow_name == GENERATOR_WORKFLOW_NAME)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Einstellungen & Infos")
        st.info(f"Modus: **{selected_workflow_name}**"); st.markdown(f"Konfig: `{agent_config_file_path}`")
        cfg_sidebar = load_agent_config(agent_config_file_path, is_generator_config=is_generator_mode)
        if cfg_sidebar:
            valid_cfg = validate_config_list(cfg_sidebar, f"'{agent_config_file_path}' (Sidebar)")
            if valid_cfg:
                st.success(f"✅ Geladen ({len(valid_cfg)} Agenten).")
                if not is_generator_mode:
                    st.subheader("Agentenübersicht:"); summary = [{"R": a.get("round"), "Name": a.get("name"), "Desc": a.get("description", "-")} for a in valid_cfg]
                    st.dataframe(summary, use_container_width=True, hide_index=True, column_config={"R": st.column_config.NumberColumn("R", width="small"), "Name": "Name", "Desc": "Info"})
                    with st.expander("Vollständige JSON"): st.json(valid_cfg)
        st.divider(); st.subheader("Tools"); st.json(list(AVAILABLE_TOOLS.keys())); model_id = DEFAULT_MODEL_ID; st.caption(f"Modell: `{model_id}`"); st.divider()
        # === RPM EINSTELLUNG ===
        st.subheader("RPM Einstellungen")
        rpm_limit_value = st.number_input(
            "Maximale Anfragen pro Minute (RPM):",
            min_value=1, max_value=120,
            value=st.session_state.get("rpm_limit_input", 10),
            step=1, key="rpm_limit_input", # Key für Widget und Session State
            help="Steuert, wie viele API-Aufrufe pro Minute maximal gesendet werden. Standard: 30"
        )
        # === ENDE RPM EINSTELLUNG ===

    # --- Session State  ---
    if 'message_store' not in st.session_state: st.session_state.message_store = {}
    if 'agent_results_display' not in st.session_state: st.session_state.agent_results_display = []
    if 'last_question_processed' not in st.session_state: st.session_state.last_question_processed = ""
    if 'uploaded_files_data' not in st.session_state: st.session_state.uploaded_files_data = []
    if 'last_workflow_processed' not in st.session_state: st.session_state.last_workflow_processed = ""

    # --- Eingabe (aus Original) ---
    q_label = f"📝 Aufgabe für '{selected_workflow_name}':" if not is_generator_mode else "📝 Ziel für Workflow-Generierung:"
    question = st.text_area(q_label, key="task_description")
    up_files = st.file_uploader("📎 Dateien hochladen (Kontext):", type=["png", "jpg", "jpeg", "webp", "txt", "py", "md", "csv", "json", "html", "css", "js", "yaml", "sh", "java", "cpp", "h", "cs", "go", "rb", "php"], accept_multiple_files=True, key="file_uploader")

    # --- Datei Handling (aus Original) ---
    current_uploaded_files_data = []
    if up_files:
        st.write("Neu hochgeladene Dateien:")
        for uf in up_files:
            try:
                fb = uf.getvalue(); fd = {"name": uf.name, "type": uf.type, "bytes": fb}
                current_uploaded_files_data.append(fd)
                if uf.type.startswith("image/"): st.image(fb, caption=f"{uf.name}", width=100)
                else: st.caption(f"- `{uf.name}` ({uf.type})")
            except Exception as e: st.error(f"Fehler '{uf.name}': {e}")
        st.session_state.uploaded_files_data = current_uploaded_files_data
        st.success(f"{len(current_uploaded_files_data)} Datei(en) bereit.")
    elif st.session_state.uploaded_files_data:
         st.write(f"Vorhandene Dateien ({len(st.session_state.uploaded_files_data)}):")
         with st.expander("Dateien verwalten", expanded=False):
             rm_indices = [];
             for idx in range(len(st.session_state.uploaded_files_data) -1, -1, -1):
                 fd = st.session_state.uploaded_files_data[idx]
                 col1, col2 = st.columns([0.8, 0.2])
                 with col1:
                    if fd["type"].startswith("image/"): st.image(fd["bytes"], caption=f"{fd['name']}", width=100)
                    else: st.caption(f"- `{fd['name']}` ({fd['type']})")
                 with col2:
                    if st.button(f"❌", key=f"rm_{idx}_{fd['name']}", help=f"'{fd['name']}' entfernen"): rm_indices.append(idx)
             if rm_indices:
                 rm_indices.sort(reverse=True)
                 for index in rm_indices:
                      if 0 <= index < len(st.session_state.uploaded_files_data): st.session_state.uploaded_files_data.pop(index)
                 st.rerun()
             if not st.session_state.uploaded_files_data: st.info("Keine Dateien vorhanden.")

    # --- Start Button ---
    btn_label = f"🚀 '{selected_workflow_name}'-Workflow starten" if not is_generator_mode else "🧬 Workflow generieren & ausführen"
    if st.button(btn_label, key="start_button"):
        if not question and not st.session_state.uploaded_files_data: st.warning("Bitte Aufgabe beschreiben oder Dateien hochladen."); st.stop()

        st.session_state.message_store = {}; st.session_state.agent_results_display = []
        st.session_state.last_question_processed = question; st.session_state.last_workflow_processed = selected_workflow_name
        results_placeholder = st.empty(); final_agents_config = None
        prompt_exec = question; gen_agent_conf = None

        # --- Phase 1: Generator  ---
        if is_generator_mode:
            results_placeholder.info("⏳ Starte Generator...")
            with st.spinner("🧠 Workflow-Generator arbeitet..."):
                cfg_list = load_agent_config(GENERATOR_CONFIG_FILE, True)
                validated_cfg = validate_config_list(cfg_list, f"'{GENERATOR_CONFIG_FILE}'") if cfg_list else None
                if not validated_cfg: st.error("Generator-Konfig ungültig."); results_placeholder.empty(); st.stop()
                gen_agent_conf = validated_cfg[0]

                gen_parts: List[Part] = [Part(text=f"System Anweisung ({gen_agent_conf.get('name')}):\n{gen_agent_conf.get('system_instruction')}\n---"), Part(text=f"Nutzeranfrage/Ziel:\n{prompt_exec}")]
                if st.session_state.uploaded_files_data:
                    gen_parts.append(Part(text="\n\n--- START KONTEXT DATEIEN ---"))
                    for fd in st.session_state.uploaded_files_data:
                        fn, ft, fb = fd["name"], fd["type"], fd["bytes"]
                        if ft.startswith("image/"):
                            try: image_part = Part(inline_data={"mime_type": ft, "data": fb}); gen_parts.extend([Part(text=f"\nBild: `{fn}`"), image_part])
                            except Exception as e: st.warning(f"⚠️ Bild '{fn}' Ignoriert: {e}")
                        else:
                            try: content = fb.decode('utf-8'); gen_parts.append(Part(text=f"\n--- START DATEI: `{fn}` ---\n{content}\n--- ENDE DATEI: `{fn}` ---"))
                            except Exception as decode_e: content = f"[Fehler beim Dekodieren: {decode_e}]"; st.warning(f"Datei '{fn}' ({ft}) ignoriert."); gen_parts.append(Part(text=f"\n--- START DATEI: `{fn}` ---\n{content}\n--- ENDE DATEI: `{fn}` ---"))
                    gen_parts.append(Part(text="\n--- ENDE KONTEXT DATEIEN ---"))

                gen_output = "[Generator nicht geantwortet]"; gen_success = False
                try:
                    client = genai.Client(api_key=API_KEY)
                    # === Verwende limited_generate_content ===
                    response = limited_generate_content(
                        client=client,
                        model=f"models/{model_id}",
                        contents=gen_parts,
                        config=GenerateContentConfig(temperature=gen_agent_conf.get("temperature", 0.5))
                    )
                  
                    # Antwort auswerten 
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        gen_output = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
                        if gen_output: gen_success = True
                        else: gen_output = "[Generator gab leere Antwort]"; gen_success = False
                    else:
                         feedback = getattr(response, 'prompt_feedback', None); block_reason = getattr(feedback, 'block_reason', "?"); block_msg = getattr(feedback, 'block_reason_message', "?")
                         gen_output = f"[Fehler: Gen-Antwort ungültig. Grund: {block_reason}. Nachricht: {block_msg}]"; gen_success = False; st.error(f"Generator-Fehler: {gen_output}")

                # Allgemeine Fehlerbehandlung 
                except Exception as gen_e:
                    st.error(f"❌ Kritischer Generator-Fehler: {gen_e}")
                    st.error(traceback.format_exc())
                    st.stop()

                st.session_state.agent_results_display.append({"agent": gen_agent_conf.get('name'), "status": "Erfolgreich" if gen_success else "Fehlgeschlagen", "output": gen_output, "sources": None, "details": "Output des Workflow Generators"})
                if not gen_success: st.error("Workflow-Generierung fehlgeschlagen."); st.stop()

                results_placeholder.info("⚙️ Parse & Validiere Workflow..."); parsed_cfg, parse_err = parse_generator_output(gen_output)
                if parse_err: st.error(f"Parse Fehler:\n{parse_err}"); st.code(gen_output, 'text'); st.stop()
                if not parsed_cfg: st.error("Generator gab leere Konfig zurück."); st.code(gen_output, 'text'); st.stop()

                validated_cfg = validate_config_list(parsed_cfg, "generierter Konfiguration")
                if not validated_cfg: st.error("❌ Generierte Konfig ungültig."); st.subheader("Fehlerhafter JSON-Output:"); st.code(gen_output, language="json"); st.stop()

                final_agents_config = validated_cfg; st.success(f"✅ Workflow ({len(final_agents_config)} Agenten) generiert!");
                st.sidebar.subheader("Dynamisch generierte Agenten:"); summary = [{"R": a.get("round"), "Name": a.get("name"), "Desc": a.get("description", "-")} for a in final_agents_config]
                st.sidebar.dataframe(summary, use_container_width=True, hide_index=True, column_config={"R": st.column_config.NumberColumn(width="small"), "Name": "Name", "Desc": "Info"}) # Original
                with st.sidebar.expander("Generierte JSON"): st.json(final_agents_config)
                results_placeholder.info("Führe generierten Workflow aus...")

        # --- Phase 2: Ausführung ---
        else:
            config_to_validate = load_agent_config(agent_config_file_path)
            if config_to_validate: final_agents_config = validate_config_list(config_to_validate, f"'{agent_config_file_path}'")
            if not final_agents_config: st.error(f"Vordefinierte Konfig für '{selected_workflow_name}' ungültig/nicht geladen."); st.stop()
            results_placeholder.info(f"Führe Workflow '{selected_workflow_name}' aus...")

        # --- Haupt-Agenten-Schleife ---
        if final_agents_config:
            overall_success = True
            try: client = genai.Client(api_key=API_KEY) # Client Init aus Original
            except Exception as e: st.error(f"❌ Client Init Fehler: {e}"); st.stop()

            with st.spinner("⏳ Agenten arbeiten..."):
                try:
                    for agent_index, agent_conf in enumerate(final_agents_config):
                        # Variablenzuweisung
                        agent_name = agent_conf.get("name", f"Agent_{agent_index+1}"); round_ = agent_conf.get("round", "N/A")
                        system_instruction = agent_conf.get("system_instruction", "-"); temperature = agent_conf.get("temperature")
                        enable_web_search = agent_conf.get("enable_web_search", False); receives_from = agent_conf.get("receives_messages_from", [])
                        accepts_files = agent_conf.get("accepts_files", False); callable_tool_names = agent_conf.get("callable_tools", [])
                        results_placeholder.info(f"🧠 Agent: **{agent_name}** (R{round_}, {agent_index+1}/{len(final_agents_config)})...")
                        all_sources_found = True

                        # 1. Input (aus Original)
                        current_input_parts: List[Part] = [Part(text=f"System Anweisung ({selected_workflow_name} - Rolle: {agent_name}):\n{system_instruction}\n---")]
                        is_first_relevant_agent = not receives_from or all(src not in st.session_state.message_store for src in receives_from)
                        if is_first_relevant_agent:
                            current_input_parts.append(Part(text=f"Nutzeranfrage:\n{prompt_exec}"))
                            if accepts_files and st.session_state.uploaded_files_data:
                                current_input_parts.append(Part(text="\n\n--- START KONTEXT DATEIEN ---"))
                                for fd in st.session_state.uploaded_files_data:
                                    fn, ft, fb = fd["name"], fd["type"], fd["bytes"]
                                    if ft.startswith("image/"):
                                        try: image_part = Part(inline_data={"mime_type": ft, "data": fb}); current_input_parts.extend([Part(text=f"\nBild: `{fn}`"), image_part])
                                        except Exception as e: st.warning(f"{agent_name}: Bild '{fn}' Fehler: {e}")
                                    else:
                                        try: content = fb.decode('utf-8'); current_input_parts.append(Part(text=f"\n--- START DATEI: `{fn}` ---\n{content}\n--- ENDE DATEI: `{fn}` ---"))
                                        except Exception as decode_e: st.warning(f"{agent_name}: Datei '{fn}' ({ft}) ignoriert (Decode-Fehler): {decode_e}"); content = "[Inhalt nicht lesbar/dekodierbar]"; current_input_parts.append(Part(text=f"\n--- START DATEI: `{fn}` ---\n{content}\n--- ENDE DATEI: `{fn}` ---"))
                                current_input_parts.append(Part(text="\n--- ENDE KONTEXT DATEIEN ---"))
                        else:
                            previous_outputs_text = []
                            for src in receives_from:
                                if src in st.session_state.message_store: previous_outputs_text.append(f"Ergebnis '{src}':\n{st.session_state.message_store[src]}")
                                else:
                                    err = f"Input von '{src}' fehlt! Überspringe '{agent_name}'."; st.warning(err)
                                    st.session_state.agent_results_display.append({"agent": agent_name, "status": "Übersprungen", "output": "[Input fehlt]", "details": err}); all_sources_found = False; break
                            if not all_sources_found: overall_success = False; continue
                            input_from_previous = "\n\n---\n\n".join(previous_outputs_text)
                            current_input_parts.append(Part(text=f"Vorherige Ergebnisse:\n{input_from_previous}\n---\nDeine Aufgabe:"))

                        # 2. Tools
                        agent_tools_list = []
                        current_agent_func_declarations = []
                        if enable_web_search:
                            try: agent_tools_list.append(Tool(google_search=GoogleSearch()))
                            except Exception as e: st.error(f"{agent_name}: Google Search Init Fehler: {e}")
                        if callable_tool_names:
                            for tn in callable_tool_names:
                                if tn in AVAILABLE_TOOLS:
                                    f = AVAILABLE_TOOLS[tn]; desc = f.__doc__.splitlines()[0] if f.__doc__ else f"Tool: {tn}"
                                    schema = {}
                                    if tn == "calculator": schema = {"type": "OBJECT", "properties": {"expression": {"type": "STRING"}}, "required": ["expression"]}
                                    elif tn == "get_current_datetime": schema = {"type": "OBJECT", "properties": {}}                                
                                    current_agent_func_declarations.append(FunctionDeclaration(name=tn, description=desc, parameters=schema))
                                else: st.warning(f"Tool '{tn}' für '{agent_name}' nicht in AVAILABLE_TOOLS.")
                            if current_agent_func_declarations: agent_tools_list.append(Tool(function_declarations=current_agent_func_declarations))

                        # 3. Config 
                        gen_cfg_args = {"response_modalities": ["TEXT"]}
                        if agent_tools_list: gen_cfg_args["tools"] = agent_tools_list
                        if temperature is not None: gen_cfg_args["temperature"] = temperature
                        agent_specific_config = GenerateContentConfig(**gen_cfg_args)

                        # 4. API Loop mit Anpassung für limited_generate_content)
                        max_function_calls = 5 
                        call_count = 0; final_agent_output = ""; agent_success_flag = False; grounding_info = None
                        conversation_history = list(current_input_parts)
                        should_skip = (agent_conf.get("name", "").startswith("Planner") and accepts_files and not st.session_state.uploaded_files_data and not question.strip()) # Aus Original

                        while call_count < max_function_calls:
                            if should_skip: 
                                st.info(f"'{agent_name}' übersprungen.")
                                st.session_state.agent_results_display.append({"agent": agent_name, "status": "Übersprungen", "output": "[Keine Frage/Dateien]"})
                                agent_success_flag = True; final_agent_output = "[Keine Frage/Dateien]"
                                break
                            try:
                                effective_model_for_call = f"models/{model_id}"
                                # === Verwende limited_generate_content ===
                                response = limited_generate_content(
                                    client=client,
                                    model=effective_model_for_call,
                                    contents=conversation_history,
                                    config=agent_specific_config
                                )
                                # === ENDE ANPASSUNG ===
                                candidate = response.candidates[0] if response.candidates else None
                                function_call = None
                                if candidate and hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    first_part = candidate.content.parts[0]
                                    if hasattr(first_part, 'function_call'): function_call = first_part.function_call
                                if function_call and hasattr(function_call, 'name'): # Handling
                                    tool_name = function_call.name; tool_args = dict(function_call.args) if hasattr(function_call, 'args') else {}
                                    st.info(f"'{agent_name}' -> Tool `{tool_name}`...")
                                    if tool_name in AVAILABLE_TOOLS:
                                        tool_function = AVAILABLE_TOOLS[tool_name]
                                        try:
                                            function_result = tool_function(**tool_args); st.success(f"Tool `{tool_name}` OK.")
                                            function_response_part = Part(function_response=FunctionResponse(name=tool_name, response={"content": str(function_result)}))
                                            conversation_history.append(first_part); conversation_history.append(function_response_part); call_count += 1; continue
                                        except Exception as func_exc: st.error(f"Tool `{tool_name}` Fehler: {func_exc}"); final_agent_output = f"[Tool Fehler {tool_name}: {func_exc}]"; agent_success_flag = False; break
                                    else: st.error(f"Unbekanntes Tool `{tool_name}`"); final_agent_output = f"[Unbekanntes Tool {tool_name}]"; agent_success_flag = False; break
                                else: # Text Handling
                                    if candidate and hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                                        text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                                        if text_parts: final_agent_output = "\n".join(text_parts).strip(); grounding_info = get_grounding_info(candidate); agent_success_flag = True
                                        else:
                                            feedback = getattr(response, 'prompt_feedback', None); block_reason = getattr(feedback, 'block_reason', "?"); block_msg = getattr(feedback, 'block_reason_message', "?")
                                            final_agent_output = f"[Fehler: Keine gültige Antwort. Grund: {block_reason}. Nachricht: {block_msg}]"; st.error(f"'{agent_name}': {final_agent_output}"); agent_success_flag = False
                                    else:
                                        feedback = getattr(response, 'prompt_feedback', None); block_reason = getattr(feedback, 'block_reason', "?"); block_msg = getattr(feedback, 'block_reason_message', "?")
                                        final_agent_output = f"[Fehler: Keine gültige Antwortstruktur. Grund: {block_reason}. Nachricht: {block_msg}]"; st.error(f"'{agent_name}': {final_agent_output}"); agent_success_flag = False
                                    break
                            # Allgemeine Fehlerbehandlung 
                            except Exception as e:
                                st.error(f"❌ Kritischer Fehler bei '{agent_name}': {e}")
                                st.error(traceback.format_exc())
                                final_agent_output = f"[Kritischer Fehler: {e}]"
                                agent_success_flag = False
                                overall_success = False
                                break # Breche while Schleife für diesen Agenten ab

                        # Nach der While-Schleife
                        if call_count >= max_function_calls:
                            st.warning(f"Agent '{agent_name}' hat Limit für Funktionsaufrufe ({max_function_calls}) erreicht.")
                            if not final_agent_output: final_agent_output = "[Limit erreicht]"
                            agent_success_flag = False

                        # Ergebnis speichern
                        if final_agent_output:
                            if agent_success_flag and "[Keine Frage/Dateien]" not in final_agent_output: st.session_state.message_store[agent_name] = final_agent_output
                            already_skipped = any(r['agent'] == agent_name and r['status'] == 'Übersprungen' and "[Input fehlt]" in r.get('output','') for r in st.session_state.agent_results_display)
                            if not already_skipped:
                                current_status = "Erfolgreich" if agent_success_flag else "Fehlgeschlagen"
                                if "[Übersprungen" in final_agent_output or "[Keine Frage/Dateien]" in final_agent_output: current_status = "Übersprungen"
                                # Original hatte keine Warnung
                                st.session_state.agent_results_display.append({"agent": agent_name, "status": current_status, "output": final_agent_output, "sources": grounding_info, "details": None })
                        if not agent_success_flag and not (should_skip or "[Keine Frage/Dateien]" in final_agent_output): overall_success = False

                # Exception Handling für die gesamte Agenten-Schleife
                except Exception as e: st.error(f"❌ Unerwarteter Fehler im Hauptprozess: {e}"); st.error(traceback.format_exc()); overall_success = False
            # Ende with spinner

            # --- Finale Statusmeldung & Ergebnisse ---
            results_placeholder.empty(); st.markdown("---")
            if not st.session_state.agent_results_display: st.warning("Keine Agenten ausgeführt.")
            elif overall_success: st.success("✅ Workflow erfolgreich abgeschlossen.")
            else: st.error("❌ Workflow mit Fehlern/Warnungen abgeschlossen.")

            st.subheader("Ergebnisse der einzelnen Agenten:")
            displayed = set() 
            for result in st.session_state.agent_results_display:
                name = result.get('agent', 'Unbekannter Agent'); status = result.get('status', 'Unbekannt')
                if name in displayed: continue; displayed.add(name)
                out = result.get('output', '[Kein Output]'); src = result.get('sources'); det = result.get('details')
                icons = {'Erfolgreich': '✅', 'Fehlgeschlagen': '❌', 'Übersprungen': '⚠️', 'Warnung': '⚠️'}
                icon = icons.get(status, '❓')
                with st.expander(f"{icon} Agent: **{name}** ({status})", expanded=(status != 'Übersprungen')):
                    st.markdown("##### Output:")
                    gen_name = gen_agent_conf['name'] if is_generator_mode and gen_agent_conf else "___"
                    is_gen = name == gen_name
                    is_code = "```" in out or (status == 'Erfolgreich' and any(kw in name.lower() for kw in ["coder", "architect", "refiner"]))
                    if is_gen and status == 'Erfolgreich': st.code(out, language="json")
                    elif is_code and status == 'Erfolgreich':
                        lang_match = re.search(r"```(\w+)", out); lang_name = selected_workflow_name.split()[0].lower().replace("plugin", "python"); lang = lang_match.group(1) if lang_match else lang_name
                        code_content = re.sub(r"```\w*\n?", "", out, count=1); code_content = re.sub(r"\n?```$", "", code_content)
                        st.code(code_content.strip(), language=lang, line_numbers=True)
                    else: st.markdown(out)
                    if src: st.markdown("##### Quellen/Infos:"); st.caption(f"{src}")
                    if det: st.info(f"Details: {det}")

            # --- Download ---
            st.markdown("---"); st.subheader("📦 Download generierter Dateien")
            files = {}
            pattern = re.compile(r"## FILE: \s*([\w\.\-\/]+\.\w+)\s*\n```(?:[\w\+\#\-\.]*\n)?(.*?)```", re.DOTALL | re.MULTILINE)
            gen_name_to_exclude = gen_agent_conf['name'] if is_generator_mode and gen_agent_conf else "___"
            for r in st.session_state.agent_results_display:
                if r.get("agent") != "WorkflowGenerator" and r.get("status") == "Erfolgreich" and r.get("output"): # Original Bedingung
                    for fn, c in pattern.findall(r["output"]): files[fn.strip()] = c.strip() + "\n"
            if files:
                st.write(f"Generierte Dateien für **'{selected_workflow_name}'**:"); st.markdown("\n".join([f"- `{fname}`" for fname in sorted(files.keys())]))
                buf = io.BytesIO()
                try:
                    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for fn, c in files.items(): zf.writestr(fn, c.encode('utf-8'))
                    dl_fn = f"{selected_workflow_name.lower().replace(' ','_')}_output_{datetime.datetime.now():%Y%m%d_%H%M}.zip"
                    st.download_button(label=f"⬇️ '{selected_workflow_name}' Ergebnisse als ZIP", data=buf.getvalue(), file_name=dl_fn, mime="application/zip", key="dl_zip")
                except Exception as e: st.error(f"❌ ZIP Fehler: {e}"); st.error(traceback.format_exc())
            else: st.info("Keine Dateien zum Zippen gefunden.")

            # --- Finales Ergebnis ---
            st.markdown("---"); st.subheader("🏁 Finales Text-Ergebnis")
            final_out = "[Kein spezifisches textuelles Endergebnis gefunden]"; final_name = None
            for r in reversed(st.session_state.agent_results_display):
                agent, out, status = r.get('agent'), r.get('output', ''), r.get('status')
                is_gen = is_generator_mode and gen_agent_conf and agent == gen_agent_conf['name']
                if status == "Erfolgreich" and not is_gen and out and "[Kein Output]" not in out and "[Keine Frage/Dateien]" not in out:
                    is_likely_just_files = pattern.fullmatch(out.strip()) is not None or out.strip().startswith("## FILE:")
                    if (not is_likely_just_files or any(kw in agent for kw in ["Planner", "Reviewer", "Packager"])):
                        final_out, final_name = out, agent; break
                    elif final_name is None: final_out, final_name = f"[Letzter Output war Code von Agent '{agent}']", agent
            st.markdown(f"**(von Agent: **{final_name}**)**" if final_name else "")
            if "[Letzter Output war Code von Agent" in final_out: st.info(final_out)
            else: st.markdown(final_out)

# --- Hauptausführungspunkt ---
if __name__ == "__main__":
    build_tab()