"""
Segment 1: Zentrale Konfiguration, Infrastruktur und Daten-Audit.

Definiert Experiment-Parameter, Pfade, Vorhersageziele, globale Hilfsfunktionen,
laedt den globalen `master_label_df` und stellt den PDF-Sorter bereit.

Limitation: `sort_and_rename_raw_pdfs` ist auf PDFs der ACM Digital Library
(https://dl.acm.org/) ausgelegt. Primaer wird die DOI im Block
"ACM Reference Format" gesucht; scheitert das, greifen generische DOI-Fallbacks.
"""

import os
import json
import random
import re
import shutil
import time
import platform
import hashlib
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# =============================================================================
# 1. EXPERIMENT-PARAMETER
# =============================================================================
DATEN_QUELLE = "pdf_raw"                 # "markdown" | "pdf_raw"
MODELL_NAME = "nomic-embed-text-v2-moe"   # Ollama-Embedding-Modell nomic-embed-text-v2-moe oder mxbai-embed-large
LLM_MODELL_NAME = "llama3.2"              # Ollama-LLM fuer Pipeline B
POOLING_STRATEGIE = "mean"                 # "mean" | "max" | "mean_max_concat"
POOLING_STRATEGIEN = ["mean", "max", "mean_max_concat"]

CHUNK_SIZE = 1200      # Zeichen
CHUNK_OVERLAP = 200    # Zeichen
# Identisch fuer Markdown und PDF, damit sich die Quellen nur in der Extraktion unterscheiden
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

# Task-Praefixe laut Ollama-Doku zu nomic-embed; Klassifikation bettet nur Dokumente ein
EMBEDDING_PREFIXES = {"nomic-embed-text-v2-moe": "search_document: "}
DOCUMENT_PREFIX = EMBEDDING_PREFIXES.get(MODELL_NAME, "")

# Literaturverzeichnis vor dem Chunking entfernen (Anhaenge danach bleiben erhalten).
# Zitierte Repositories und Datensatz-Links fremder Arbeiten wuerden sonst
# wie eigene Artefakte des Papers wirken.
REFERENCES_ENTFERNEN = True

STRATEGIE_NAME = (
    f"recursive_{CHUNK_SIZE}c_{CHUNK_OVERLAP}o"
    + ("_prefixed" if DOCUMENT_PREFIX else "")
    + ("_norefs" if REFERENCES_ENTFERNEN else "")
)

GLOBAL_SEED = 42
REQUEST_TIMEOUT = 120  # Sekunden pro Ollama-Request

# Nachteilen: Chunks, die das Kontextfenster des Embedding-Modells ueberschreiten,
# werden rekursiv halbiert (gleiche Separatoren, Overlap proportional zur Groesse).
NACHTEIL_MIN_CHARS = 100   # Kuerzere Teile werden nicht weiter geteilt
EMBED_MAX_VERSUCHE = 3     # Versuche bei voruebergehenden Ollama-Fehlern

UNTERORDNER_JAHRE = ["2023", "2024", "2025"]

# Annotiert, aber nicht verwendet (655 annotiert, 654 verwendet).
# Die Label-Tabelle enthaelt diese Studien bereits nicht mehr; der Eintrag dokumentiert
# den Ausschluss und verhindert, dass ihre Nummer als unerwartete Luecke gemeldet wird.
EXCLUDED_PAPER_IDS = {
    "2023_Dec_188": "Textfonts ohne Unicode-Zuordnung (Identity-H); Markdown zu 94,5 % U+FFFD, "
                    "PDF-Text systematisch verschoben (Diagnose: Datenvalidierung, Abschnitt Diagnose)",
}

# Luecken in der ACM-Artikelnummerierung, keine fehlenden Studien.
# Vermutete Ursache: Nummer vergeben, Artikel erst in spaeterem Issue erschienen.
KNOWN_ID_GAPS = {
    "2024": [165],
    "2025": [173],
}

# =============================================================================
# 2. ZIELVARIABLEN
# =============================================================================
# Papers, deren Artefakt-Labels der Protokollregel folgen
# (einziger Typ User Study und/oder Literature Review)

KONVENTION_TYPEN = {"User Study", "Literature Review"}

def folgt_konvention(paper_type) -> bool:
    typen = {t.strip() for t in str(paper_type).split(";") if t.strip()}
    return bool(typen) and typen <= KONVENTION_TYPEN

PAPER_TYPE_CATEGORIES = [
    "Algorithm Development", "Hardware Development",
    "Software Development", "User Study",
    "Dataset Paper", "Literature Review",
]

# code_url / data_url werden im Audit zu "URL angegeben" (Yes/No) binarisiert.
# Nicht vorhergesagt: open_access (auf der ACM-Seite direkt sichtbar),
# is_dataset_paper (Titel-Regel), instructions_present (erfordert Repository-Crawling)
PREDICTION_TARGETS = [
    "paper_type",
    "code_required", "code_available", "code_url",
    "data_required", "data_available", "data_url",
    "hardware_required", "software_required",
]

# Geplant, aktuell nicht verwendet.
FUTURE_REPO_TARGETS = ["repo_is_empty", "repo_has_sufficient_readme"]

# Binarisierung der kategorialen Ziele (Yes/No). NaN entspricht "N/A" im Codebook.
# N/A und Unknown sind keine Vorhersageklassen; nach dem Mapping enthaelt jedes
# binaere Ziel ausschliesslich "Yes" oder "No".
#
# *_required (code, data, hardware, software):
#   Unknown -> No   Anforderung im Paper nicht erkennbar.
#   N/A     -> No   Annotationsprotokoll: Studien, die ausschliesslich User Study sind,
#                   erhielten pauschal "No" bzw. "N/A", wo moeglich (48 reine User Studies,
#                   dazu 8 Literature Reviews, 2 Dataset Papers, 2 sonstige bei hardware/
#                   software). N/A steht damit fuer "nicht benoetigt".
#
# *_available (Zielfrage: Ist das Artefakt oeffentlich zugaenglich?):
#   N/A          -> No   Tritt nur bei *_required = No auf: nicht benoetigt,
#                        also auch nicht veroeffentlicht.
#   Partially    -> Yes  Ein Teil des Artefakts ist oeffentlich abrufbar.
#   Upon_request -> No   Codebook definiert "Yes" als oeffentlich verfuegbar;
#                        Zugang auf Anfrage erfuellt das nicht.
#   Unknown      -> No   Annotator konnte innerhalb von ca. 20 Minuten nicht entscheiden.
#                        Verfuegbarkeit wurde nicht nachgewiesen.
#
# *_url: "Yes", wenn eine gueltige URL angegeben ist, sonst "No".
_REQUIRED_MAP = {"Yes": "Yes", "No": "No", "Unknown": "No", "N/A": "No"}
_AVAILABLE_MAP = {
    "Yes": "Yes", "No": "No", "N/A": "No",
    "Partially": "Yes",
    "Upon_request": "No",
    "Unknown": "No",
}
LABEL_MAPPINGS = {
    "code_required": _REQUIRED_MAP,
    "data_required": _REQUIRED_MAP,
    "hardware_required": _REQUIRED_MAP,
    "software_required": _REQUIRED_MAP,
    "code_available": _AVAILABLE_MAP,
    "data_available": _AVAILABLE_MAP,
}

URL_TARGETS = ["code_url", "data_url"]
# Alle Ziele ausser paper_type sind nach der Aufbereitung binaer (Yes/No)
BINARY_TARGETS = [*LABEL_MAPPINGS, *URL_TARGETS]
URL_PATTERN = r"https?://|www\.|github|\.(?:com|org|io|net)\b"


# =============================================================================
# 3. REPRODUZIERBARKEIT
# =============================================================================
def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Setzt Seeds fuer random, NumPy und (falls installiert) PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Fuer deterministische cuBLAS-Operationen; muss vor der ersten CUDA-Operation gesetzt sein
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Warnt bei Operationen ohne deterministische Implementierung, statt abzubrechen
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


set_global_seed(GLOBAL_SEED)


# =============================================================================
# 4. PFADE
# =============================================================================
BASE_DIR = Path(os.getcwd())

STUDIEN_ROOT = BASE_DIR / "daten" / "pdfs"
# Ground Truth: wird nur gelesen, nie veraendert
LABEL_CSV_PATH = BASE_DIR / "IMWUT_Dataset_original.csv"
# Klassifikationsdatensatz: bei jedem Lauf aus dem Original erzeugt (paper_id, year,
# binarisierte Ziele); nicht von Hand bearbeiten
KLASSIFIKATION_CSV_PATH = BASE_DIR / "daten" / "labels" / "IMWUT_Dataset_klassifikation.csv"
MARKDOWN_ROOT = BASE_DIR / "daten" / "markdown"
RAW_PDF_DIR = BASE_DIR / "daten" / "raw_downloads"   # unsortierte Downloads

EMBEDDING_BASE_ROOT = BASE_DIR / "daten" / "embeddings"
CURRENT_EMBEDDING_PATH = EMBEDDING_BASE_ROOT / DATEN_QUELLE / MODELL_NAME / STRATEGIE_NAME

FUSION_BASE_ROOT = BASE_DIR / "daten" / "fusionsdaten"
CURRENT_FUSION_PATH = FUSION_BASE_ROOT / DATEN_QUELLE / POOLING_STRATEGIE / MODELL_NAME / STRATEGIE_NAME

PREDICTIONS_BASE_ROOT = BASE_DIR / "daten" / "predictions_llm"
CURRENT_PREDICTIONS_PATH = PREDICTIONS_BASE_ROOT / DATEN_QUELLE / LLM_MODELL_NAME

PREDICTIONS_ML_ROOT = BASE_DIR / "daten" / "Predictions"
RUN_LOG_ROOT = BASE_DIR / "daten" / "run_configs"

print("Projektstruktur wird initialisiert")
print(f"Quelle: {DATEN_QUELLE} | Embedding: {MODELL_NAME} | LLM: {LLM_MODELL_NAME}")
print(f"Chunking: {STRATEGIE_NAME} | Pooling: {POOLING_STRATEGIE} | Seed: {GLOBAL_SEED}")
print(f"Vorhersageziele: {len(PREDICTION_TARGETS)} Spalten")

# Ordner anlegen (Jahres-Unterordner ausser fuer raw_downloads)
for pfad in [STUDIEN_ROOT, MARKDOWN_ROOT]:
    for jahr in UNTERORDNER_JAHRE:
        (pfad / jahr).mkdir(parents=True, exist_ok=True)
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

# Ergebnisordner (Embeddings, Fusion, Vorhersagen) legen erst die Segmente an, die
# hineinschreiben. So entstehen keine leeren Ordner fuer Konfigurationen ohne Lauf.
RUN_LOG_ROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 5. LABELS LADEN UND DATEN-AUDIT
# =============================================================================
def file_sha256(pfad: Path) -> str:
    """SHA-256 einer Datei, ordnet jeden Lauf einer Labelversion zu."""
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def load_labels() -> pd.DataFrame:
    """Liest die zentrale Label-CSV. utf-8-sig entfernt das BOM vor 'title'."""
    if not LABEL_CSV_PATH.exists():
        raise FileNotFoundError(f"Label-Datei fehlt: {LABEL_CSV_PATH}")
    df = pd.read_csv(LABEL_CSV_PATH, encoding="utf-8-sig")
    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    df["year"] = df["year"].astype(str).str.strip()
    return df


print("\nDaten-Audit")
master_label_df = load_labels()
LABEL_CSV_SHA256 = file_sha256(LABEL_CSV_PATH)
# Annotiert = Tabelle plus dokumentierte Ausschluesse, die nicht mehr in ihr stehen
n_annotiert = len(master_label_df) + len(set(EXCLUDED_PAPER_IDS) - set(master_label_df["paper_id"]))
print(f"Label-Datei: {LABEL_CSV_PATH.name} (SHA-256 {LABEL_CSV_SHA256[:12]}...)")

# A: Duplikate
for spalte in ["paper_id", "doi_url"]:
    dups = master_label_df[spalte][master_label_df[spalte].duplicated() & master_label_df[spalte].notna()]
    if not dups.empty:
        print(f"WARNUNG: Doppelte Werte in {spalte}: {sorted(dups.unique())}")

# B: Luecken im ID-Zaehler, bekannte Nummerierungsluecken werden getrennt ausgewiesen
print("\nID-Nummerierung je Jahr")
theoretisches_maximum = 0
for jahr in UNTERORDNER_JAHRE:
    ids = master_label_df.loc[master_label_df["year"] == jahr, "paper_id"]
    nums = set(ids.str.extract(r"_(\d+)$")[0].dropna().astype(int))
    if not nums:
        print(f"{jahr}: keine Eintraege")
        continue
    max_id = max(nums)
    theoretisches_maximum += max_id
    fehlend = set(range(1, max_id + 1)) - nums
    ausgeschlossen = {int(p.rsplit("_", 1)[1]) for p in EXCLUDED_PAPER_IDS if p.startswith(f"{jahr}_")}
    bekannt = sorted(fehlend & set(KNOWN_ID_GAPS.get(jahr, [])))
    unerwartet = sorted(fehlend - set(bekannt) - ausgeschlossen)
    teile = [f"bekannte Luecken {bekannt}"] if bekannt else []
    teile += [f"ausgeschlossen {sorted(fehlend & ausgeschlossen)}"] if fehlend & ausgeschlossen else []
    status = ", ".join(teile) or "lueckenlos"
    print(f"{jahr}: 1..{max_id}, {len(nums)} Eintraege, {status}")
    if unerwartet:
        print(f"WARNUNG: {jahr} unerwartet fehlende Nummern {unerwartet}")

# C: Ausschluesse anwenden
in_tabelle = set(EXCLUDED_PAPER_IDS) & set(master_label_df["paper_id"])
master_label_df = master_label_df[~master_label_df["paper_id"].isin(EXCLUDED_PAPER_IDS)].reset_index(drop=True)
for p_id, grund in EXCLUDED_PAPER_IDS.items():
    ort = "aus Tabelle entfernt" if p_id in in_tabelle else "nicht in Tabelle"
    print(f"Ausgeschlossen: {p_id} ({grund}; {ort})")

# D: Abgleich in beide Richtungen: Labels ohne PDF und PDFs ohne Label
missing_pdfs, verwaiste_pdfs = [], []
for jahr in UNTERORDNER_JAHRE:
    vorhandene = {f.stem.strip() for f in (STUDIEN_ROOT / jahr).glob("*.pdf")}
    jahr_ids = set(master_label_df.loc[master_label_df["year"] == jahr, "paper_id"])
    missing_pdfs += sorted(jahr_ids - vorhandene)
    # Ausgeschlossene Studien duerfen eine PDF haben
    verwaiste_pdfs += sorted(vorhandene - jahr_ids - set(EXCLUDED_PAPER_IDS))


def _zeige_ids(titel: str, ids: list, limit: int = 20) -> None:
    if ids:
        rest = f" ... (+{len(ids) - limit} weitere)" if len(ids) > limit else ""
        print(f"{titel}: {', '.join(ids[:limit])}{rest}")


# E: Zusammenfassung
print("\nZusammenfassung Datenbasis")
print(f"Theoretisches Maximum (ID-Zaehler): {theoretisches_maximum}")
print(f"Annotiert:                          {n_annotiert}")
print(f"Verwendet:                          {len(master_label_df)}")
print(f"Vorhandene PDFs (verwendet):        {len(master_label_df) - len(missing_pdfs)}")
print(f"Fehlende PDFs:                      {len(missing_pdfs)}")
print(f"PDFs ohne Label-Eintrag:            {len(verwaiste_pdfs)}")
_zeige_ids("Fehlende IDs", missing_pdfs)
_zeige_ids("PDFs ohne Label", verwaiste_pdfs)

# Unveraenderte Ground Truth (alle Spalten) fuer Validierung und deskriptive Statistik
master_label_raw_df = master_label_df.copy()
# Bibliographische Metadaten (DOI, Titel) fuer Ausgaben, nicht fuer die Vorhersage
master_meta_df = master_label_df[["paper_id", "year", "title", "doi_url"]].copy()

# F: Kategoriale Ziele binarisieren
print("\nLabel-Mapping")
for col, mapping in LABEL_MAPPINGS.items():
    if col not in master_label_df.columns:
        continue
    raw = master_label_df[col].fillna("N/A").astype(str).str.strip()
    unbekannt = sorted(set(raw) - set(mapping))
    if unbekannt:
        print(f"WARNUNG: {col} enthaelt nicht gemappte Werte {unbekannt}")
    master_label_df[col] = raw.map(mapping)
    umkodiert = (raw != master_label_df[col]).sum()
    print(f"{col}: {umkodiert} Eintraege umkodiert")

# G: URL-Spalten binarisieren; Eintraege ohne URL (fehlplatzierte Notizen) zaehlen als "No"
for col in URL_TARGETS:
    gefuellt = master_label_df[col].notna()
    ist_url = master_label_df[col].astype(str).str.contains(URL_PATTERN, case=False, regex=True)
    fehler = master_label_df.loc[gefuellt & ~ist_url, "paper_id"].tolist()
    if fehler:
        print(f"WARNUNG: {col} ohne URL bei {fehler}, wird als 'No' gewertet")
    master_label_df[col] = np.where(gefuellt & ist_url, "Yes", "No")

# Kontrolle: binaere Ziele duerfen nur Yes/No enthalten
for col in BINARY_TARGETS:
    rest = master_label_df[col].fillna("NaN").value_counts().drop(["Yes", "No"], errors="ignore")
    if not rest.empty:
        print(f"WARNUNG: {col} enthaelt Werte ausser Yes/No: {rest.to_dict()}")

# H: Klassifikationsdatensatz: nur Schluessel und Ziele. Metadaten (Zitat, Datum, Issue,
# DOI ...) und Freitext-Notizen sind keine Eingaben der Praediktoren; die Notizen
# enthalten zudem die Begruendung der Labels und duerfen nie als Merkmal dienen.
master_label_df = master_label_df[["paper_id", "year", *PREDICTION_TARGETS]].copy()
KLASSIFIKATION_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
master_label_df.to_csv(KLASSIFIKATION_CSV_PATH, index=False)
KLASSIFIKATION_CSV_SHA256 = file_sha256(KLASSIFIKATION_CSV_PATH)
print(f"\nKlassifikationsdatensatz: {KLASSIFIKATION_CSV_PATH} "
      f"({len(master_label_df)} Zeilen, {master_label_df.shape[1]} Spalten, "
      f"SHA-256 {KLASSIFIKATION_CSV_SHA256[:12]}...)")
if file_sha256(LABEL_CSV_PATH) != LABEL_CSV_SHA256:
    raise RuntimeError("Ground-Truth-Datei wurde waehrend des Laufs veraendert.")


# =============================================================================
# 6. OLLAMA-KOMMUNIKATION
# =============================================================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/embeddings"


class KontextZuLangError(Exception):
    """Text ueberschreitet das Kontextfenster des Embedding-Modells (dauerhafter Fehler)."""


def get_embedding(text: str, raise_on_context: bool = False):
    """
    Liefert den Embedding-Vektor fuer `text` oder None bei Fehlern.
    Mit raise_on_context=True wird ein zu langer Text als KontextZuLangError gemeldet.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODELL_NAME, "prompt": DOCUMENT_PREFIX + text},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        print(f"Embedding-Fehler (Verbindung): {e}")
        return None
    if response.status_code != 200:
        if "context length" in response.text:
            if raise_on_context:
                raise KontextZuLangError(f"{len(text)} Zeichen")
        # Ollama liefert den Grund im Antworttext
        print(f"Embedding-Fehler (HTTP {response.status_code}, {len(text)} Zeichen): {response.text[:300]}")
        return None
    try:
        vektor = response.json()["embedding"]
    except (KeyError, ValueError) as e:
        print(f"Embedding-Fehler (Antwort): {e}")
        return None
    if not vektor:
        print(f"Embedding-Fehler: leerer Vektor ({len(text)} Zeichen)")
        return None
    return vektor


def _teile(text: str) -> list:
    """Halbiert einen Text mit den Chunking-Separatoren, Overlap proportional zur Groesse."""
    groesse = (len(text) + 1) // 2
    return RecursiveCharacterTextSplitter(
        chunk_size=groesse,
        chunk_overlap=CHUNK_OVERLAP * groesse // CHUNK_SIZE,
        separators=CHUNK_SEPARATORS,
    ).split_text(text)


def embed_chunk(text: str):
    """
    Vektorisiert einen Chunk. Ist er zu lang fuer das Kontextfenster, wird er
    rekursiv nachgeteilt. Rueckgabe: Liste von {"text", "vector", "nachgeteilt"}
    oder None, wenn ein Teil nicht vektorisiert werden konnte.
    """
    for versuch in range(EMBED_MAX_VERSUCHE):
        try:
            vektor = get_embedding(text, raise_on_context=True)
        except KontextZuLangError:
            if len(text) < NACHTEIL_MIN_CHARS:
                print(f"Embedding-Fehler: Teil mit {len(text)} Zeichen weiterhin zu lang")
                return None
            ergebnis = []
            for teil in _teile(text):
                teil_ergebnis = embed_chunk(teil)
                if teil_ergebnis is None:
                    return None
                ergebnis += [{**t, "nachgeteilt": True} for t in teil_ergebnis]
            return ergebnis
        if vektor:
            return [{"text": text, "vector": vektor, "nachgeteilt": False}]
        time.sleep(2 ** versuch)  # nur voruebergehende Fehler werden wiederholt
    return None


def check_ollama() -> bool:
    """Prueft, ob der Ollama-Server erreichbar ist."""
    try:
        return requests.get(OLLAMA_BASE_URL, timeout=5).status_code == 200
    except requests.RequestException:
        return False


def get_ollama_model_info() -> dict:
    """Liest Name und Digest der lokal installierten Ollama-Modelle (fuer das Run-Log)."""
    try:
        models = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).json().get("models", [])
        return {m.get("name"): m.get("digest") for m in models}
    except (requests.RequestException, ValueError):
        return {}


# =============================================================================
# 7. CHUNKING
# =============================================================================
# Ueberschrift des Literaturverzeichnisses, z. B. "## REFERENCES", "**References**", "REFERENCES"
_REFERENCES_HEADING = re.compile(
    r"^[ \t]*(?:#+[ \t]*)?(?:\*\*)?[ \t]*(?:references|bibliography)[ \t]*(?:\*\*)?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
# Kandidaten fuer den Beginn eines Anhangs, z. B. "## A IMPLEMENTATION DETAILS",
# "## A Coding Usage Guidelines", "## A sEMG Acquisition System", "Appendix A USER COMMANDS",
# "**Appendix A** Surveys", "11 APPENDIX", "9 SUPPLEMENTARY", "## 10 IRB COMPLIANCE DECLARATION",
# "## METHODOLOGICAL TRANSPARENCY & REPRODUCIBILITY APPENDIX (META)"
_ANHANG_WORT = r"(?i:appendix|appendices|supplementary)"
_APPENDIX_KANDIDAT = re.compile(
    r"^[ \t]*(?:"
    # Markdown-Ueberschrift: Buchstaben-Abschnitt beliebiger Schreibweise, nummerierter
    # Abschnitt in Grossbuchstaben ("10 IRB COMPLIANCE DECLARATION") oder Anhang-Wort irgendwo
    rf"#+[ \t]*(?:\*\*|_)?[ \t]*(?:[A-H]\.?[ \t]+\S|\d+\.?[ \t]+[A-Z]{{2}}|[^\n]*\b{_ANHANG_WORT}\b)"
    # Rohtext und Markdown: Anhang-Wort am Zeilenanfang (ggf. nummeriert) oder "A GROSSBUCHSTABE"
    rf"|(?:\*\*|_)?[ \t]*(?:(?:\d+\.?[ \t]+)?{_ANHANG_WORT}\b|[A-H]\.?[ \t]+[A-Z])"
    r")[^\n]{0,150}$",
    re.MULTILINE,
)
# Nur PDF-Rohtext: PyMuPDF gibt Abschnittsbuchstabe und Titel auf getrennten Zeilen aus,
# z. B. "A" / "IMPLEMENTATION DETAILS". Fuer Markdown nicht verwendet, damit die dort
# validierten Schnitte unveraendert bleiben.
_APPENDIX_ROH_ZWEIZEILIG = re.compile(r"^[ \t]*[A-H]\.?[ \t]*\n[ \t]*[A-Z][^\n]{2,150}$", re.MULTILINE)
# Nur PDF-Rohtext: nummerierter Abschnitt auf zwei Zeilen in Grossbuchstaben,
# z. B. "10" / "IRB COMPLIANCE DECLARATION" oder "9" / "SUPPLEMENTARY"
_APPENDIX_ROH_NUMMERIERT = re.compile(r"^[ \t]*\d{1,2}\.?[ \t]*\n[ \t]*[A-Z][A-Z0-9 ,:&\-]{3,150}$", re.MULTILINE)
# Eintrag im Literaturverzeichnis: "[12] ..." (Rohtext) bzw. "- [12] ..." (Markdown)
_REFERENZ_EINTRAG = re.compile(r"^[ \t]*(?:-[ \t]*)?\[(\d+)\]", re.MULTILINE)


def _letzter_eintrag(text: str, ab: int, rohtext: bool):
    """
    Letzter Eintrag des Literaturverzeichnisses.
    Markdown: letzte Zeile, die mit "[n]" beginnt (validierte Regel).
    Rohtext: letzter Eintrag der aufsteigenden Nummernfolge [1], [2], ... (Luecken bis 3
    erlaubt). Im Rohtext beginnen auch Zeilen im Anhang mit Zitaten wie "[71], ..." oder
    Tabellenzellen wie "[12]"; deren Nummern sind nie groesser als die des letzten
    Eintrags und werden so nicht mitgezaehlt.
    """
    letzter = None
    if not rohtext:
        for letzter in _REFERENZ_EINTRAG.finditer(text, ab):
            pass
        return letzter
    aktuell = 0
    for treffer in _REFERENZ_EINTRAG.finditer(text, ab):
        nummer = int(treffer.group(1))
        if aktuell < nummer <= aktuell + 3:
            aktuell, letzter = nummer, treffer
    return letzter


def entferne_literaturverzeichnis(text: str, rohtext: bool = False) -> tuple[str, dict]:
    """
    Entfernt das Literaturverzeichnis ab der letzten passenden Ueberschrift bis
    zum Beginn eines Anhangs bzw. Textende. Ohne erkannte Ueberschrift bleibt
    der Text unveraendert. Funktioniert fuer Markdown und PDF-Rohtext.

    Als Anhang-Beginn gilt der erste Kandidat nach dem letzten Literatureintrag.
    Zeilen innerhalb des Verzeichnisses, die zufaellig wie eine Ueberschrift
    aussehen, werden so nicht als Anhang erkannt.
    """
    treffer = list(_REFERENCES_HEADING.finditer(text))
    if not treffer:
        return text, {"references_gefunden": False, "references_start": None,
                      "references_chars": 0, "anhang_erhalten": False}

    start = treffer[-1].start()
    letzter_eintrag = _letzter_eintrag(text, treffer[-1].end(), rohtext)
    suche_ab = letzter_eintrag.end() if letzter_eintrag else treffer[-1].end()
    muster = [_APPENDIX_KANDIDAT] + ([_APPENDIX_ROH_ZWEIZEILIG, _APPENDIX_ROH_NUMMERIERT] if rohtext else [])
    kandidaten = [k for k in (m.search(text, suche_ab) for m in muster) if k]
    anhang = min(kandidaten, key=lambda k: k.start()) if kandidaten else None
    ende = anhang.start() if anhang else len(text)
    return text[:start] + text[ende:], {
        "references_gefunden": True,
        "references_start": start,
        "references_chars": ende - start,
        "anhang_erhalten": anhang is not None,
    }


def schnitt_signatur(metadata: dict) -> str:
    """Lage des entfernten Literaturverzeichnisses; aendert sie sich, aendern sich die Chunks."""
    return f"{metadata.get('references_start')}:{metadata.get('references_chars')}"


def extract_pdf_structure(filepath):
    """
    Zerlegt ein Dokument in Chunks. Verarbeitet je nach DATEN_QUELLE .md oder .pdf.
    Rueckgabe: (Liste von {"chunk_id", "text"}, Metadaten-Dict)
    """
    filepath = Path(filepath)
    chunks, metadata = [], {}

    if DATEN_QUELLE == "markdown" and filepath.suffix == ".md":
        try:
            text = filepath.read_text(encoding="utf-8")
            metadata = {"format": "markdown", "chars": len(text)}
        except Exception as e:
            print(f"Fehler beim Lesen von {filepath.name}: {e}")
            return chunks, metadata

    elif DATEN_QUELLE == "pdf_raw" and filepath.suffix == ".pdf":
        try:
            # MuPDF-Meldungen betreffen meist Bilder oder Farbprofile; gezaehlt statt ausgegeben
            fitz.TOOLS.mupdf_display_errors(False)
            fitz.TOOLS.reset_mupdf_warnings()
            with fitz.open(filepath) as doc:
                text = "\n".join(page.get_text("text") for page in doc)
                warnungen = fitz.TOOLS.mupdf_warnings(reset=True)
                metadata = {"format": "pdf_raw", "pages": doc.page_count, "chars": len(text),
                            "mupdf_warnungen": len(warnungen.splitlines()) if warnungen else 0}
        except Exception as e:
            print(f"Fehler beim Lesen von {filepath.name}: {e}")
            return chunks, metadata

    else:
        print(f"Konflikt: DATEN_QUELLE='{DATEN_QUELLE}', Dateiendung='{filepath.suffix}'")
        return chunks, metadata

    if REFERENCES_ENTFERNEN:
        text, info = entferne_literaturverzeichnis(text, rohtext=(DATEN_QUELLE == "pdf_raw"))
        metadata.update(info)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    for i, chunk_text in enumerate(splitter.split_text(text)):
        chunks.append({"chunk_id": i, "text": chunk_text})
    return chunks, metadata


# =============================================================================
# 8. PDF-SORTER
# =============================================================================
DOI_PATTERN = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"


def _find_doi(text: str):
    """Sucht eine DOI in absteigender Spezifitaet: ACM-Referenzblock, 10.1145/..., beliebige DOI."""
    patterns = [
        rf"ACM Reference Format.*?({DOI_PATTERN})",
        r"(10\.1145/[-._;()/:A-Z0-9]+)",
        rf"({DOI_PATTERN})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower().rstrip(".,;")
    return None


def sort_and_rename_raw_pdfs():
    """
    Ordnet PDFs aus RAW_PDF_DIR ueber die DOI einer paper_id zu und kopiert sie
    als <paper_id>.pdf in den passenden Jahresordner.
    """
    master_df = load_labels()

    master_df["clean_doi"] = (
        master_df["doi_url"].astype(str)
        .str.replace(r"https?://doi\.org/", "", regex=True)
        .str.strip().str.lower()
    )

    debug_dir = BASE_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(debug_dir / "master_ground_truth.csv", index=False)

    doi_to_paper = {
        row["clean_doi"]: (row["paper_id"], row["year"])
        for _, row in master_df.iterrows()
        if row["clean_doi"] and row["clean_doi"] != "nan"
    }

    raw_pdfs = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not raw_pdfs:
        print(f"Keine unsortierten PDFs in {RAW_PDF_DIR.name}.")
        return

    erfolgreich, fehlerhafte_dateien = 0, []
    for raw_pdf in tqdm(raw_pdfs, desc="Sortiere PDFs"):
        try:
            # Erste zwei Seiten, da der ACM-Referenzblock umbrechen kann
            with fitz.open(raw_pdf) as doc:
                such_text = " ".join(doc[i].get_text("text") for i in range(min(2, len(doc))))
            such_text = such_text.replace("\n", " ")

            doi = _find_doi(such_text)
            if doi is None:
                fehlerhafte_dateien.append(f"{raw_pdf.name} (keine DOI auf Seite 1-2)")
            elif doi not in doi_to_paper:
                fehlerhafte_dateien.append(f"{raw_pdf.name} (DOI '{doi}' nicht in Labels)")
            else:
                paper_id, year = doi_to_paper[doi]
                target_dir = STUDIEN_ROOT / year
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(raw_pdf, target_dir / f"{paper_id}.pdf")
                erfolgreich += 1
        except Exception as e:
            fehlerhafte_dateien.append(f"{raw_pdf.name} (Lesefehler: {e})")

    print(f"\nZugeordnet und kopiert: {erfolgreich}")
    print(f"Nicht zuordenbar:       {len(fehlerhafte_dateien)}")
    for info in fehlerhafte_dateien:
        print(f"  - {info}")


# =============================================================================
# 9. ERGEBNIS-SPEICHERUNG
# =============================================================================
def _ml_output_dir(classifier_name: str) -> Path:
    pfad = PREDICTIONS_ML_ROOT / classifier_name / DATEN_QUELLE / POOLING_STRATEGIE / MODELL_NAME / STRATEGIE_NAME
    pfad.mkdir(parents=True, exist_ok=True)
    return pfad


def _run_filename(classifier_name: str, val_method: str, is_weighted: bool, suffix: str) -> str:
    return (f"{DATEN_QUELLE}_{MODELL_NAME}_{STRATEGIE_NAME}_pool-{POOLING_STRATEGIE}"
            f"_val-{val_method}_weight-{is_weighted}_clf-{classifier_name}{suffix}")


def save_ml_predictions(y_true, y_pred, classifier_name, columns,
                        paper_ids=None, val_method=None, is_weighted=None):
    """Speichert Ground Truth und Vorhersagen je Label als CSV (optional mit paper_id)."""
    val_method = val_method or globals().get("VALIDIERUNGS_METHODE", "unknown")
    is_weighted = is_weighted if is_weighted is not None else globals().get("NUTZE_GEWICHTUNG", False)

    daten = {}
    if paper_ids is not None:
        daten["paper_id"] = list(paper_ids)
    for i, col in enumerate(columns):
        daten[f"{col}_true"] = y_true[:, i]
        daten[f"{col}_pred"] = y_pred[:, i]

    pfad = _ml_output_dir(classifier_name) / _run_filename(classifier_name, val_method, is_weighted, ".csv")
    pd.DataFrame(daten).to_csv(pfad, index=False)
    print(f"Vorhersagen gespeichert: {pfad}")


def save_experiment_log(log_text, classifier_name, val_method=None, is_weighted=None):
    """Speichert einen Text-Report zum Lauf."""
    val_method = val_method or globals().get("VALIDIERUNGS_METHODE", "unknown")
    is_weighted = is_weighted if is_weighted is not None else globals().get("NUTZE_GEWICHTUNG", False)

    pfad = _ml_output_dir(classifier_name) / _run_filename(classifier_name, val_method, is_weighted, "_REPORT.txt")
    pfad.write_text(log_text, encoding="utf-8")
    print(f"Report gespeichert: {pfad}")


def save_run_config(extra: dict | None = None) -> Path:
    """Schreibt alle Experiment-Parameter, Umgebung und Modell-Digests als JSON."""
    config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "daten_quelle": DATEN_QUELLE,
        "embedding_modell": MODELL_NAME,
        "llm_modell": LLM_MODELL_NAME,
        "pooling": POOLING_STRATEGIE,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_separators": CHUNK_SEPARATORS,
        "nachteil_min_chars": NACHTEIL_MIN_CHARS,
        "references_entfernen": REFERENCES_ENTFERNEN,
        "document_prefix": DOCUMENT_PREFIX,
        "excluded_paper_ids": EXCLUDED_PAPER_IDS,
        "label_mappings": LABEL_MAPPINGS,
        "seed": GLOBAL_SEED,
        "label_csv": str(LABEL_CSV_PATH),
        "label_csv_sha256": LABEL_CSV_SHA256,
        "klassifikation_csv": str(KLASSIFIKATION_CSV_PATH),
        "klassifikation_csv_sha256": KLASSIFIKATION_CSV_SHA256,
        "n_annotiert": n_annotiert,
        "n_verwendet": len(master_label_df),
        "prediction_targets": PREDICTION_TARGETS,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ollama_models": get_ollama_model_info(),
    }
    if extra:
        config.update(extra)
    pfad = RUN_LOG_ROOT / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    pfad.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return pfad


print("\nSetup abgeschlossen.")
