import io
import json
import os
import re
import tempfile
import urllib.request
import uuid
from datetime import datetime

import ctranslate2
import streamlit as st
from docx import Document
from fpdf import FPDF
from pypdf import PdfReader
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# Optional ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# ------------------------------------------------------------------ config
MODEL_NAME = "olob0/nllb-200-distilled-600M-ct2-int8_float16"
MAX_CHUNK_CHARS = 300
BATCH_SIZE = 3
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
    "Urdu": "urd_Arab",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
    "Arabic": "arb_Arab",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
}

# ------------------------------------------------------------------ page setup
st.set_page_config(page_title="AIT Global Technologies", page_icon="🌍", layout="centered")

st.markdown("""
<style>
.stApp, [data-testid="stHeader"] { background: #FFFFFF; }
.stApp, .stApp p, .stApp label, .stApp span, .stApp li { color: #1B1B1B; }
.block-container { padding-top: 4.5rem !important; max-width: 820px; }

.title-card {
    background: #0B2A5B;
    border-radius: 12px;
    padding: 14px 22px;
    margin-bottom: 1.2rem;
    text-align: center;
}
.title-card h1 {
    margin: 0;
    padding: 0;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.title-card h1 .ait {
    color: #FFC107 !important;   /* Yellow */
}
.title-card h1 .rest {
    color: #FFFFFF !important;   /* White */
}

.stButton > button, .stDownloadButton > button,
[data-testid="stFileUploader"] button {
    background: #FFC107 !important;
    color: #1B1B1B !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #FFB300 !important;
    color: #000000 !important;
}
.stDownloadButton > button { width: 100%; }
</style>

<div class="title-card">
    <h1>
        <span class="ait">AIT</span>
        <span class="rest"> GLOBAL TECHNOLOGIES</span>
    </h1>
</div>
""", unsafe_allow_html=True)
# ------------------------------------------------------------------ thread
def new_thread_id():
    return "THR-" + uuid.uuid4().hex[:8].upper()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()
if "streamed_text" not in st.session_state:
    st.session_state.streamed_text = ""
if "result" not in st.session_state:
    st.session_state.result = None

# ------------------------------------------------------------------ chromadb / history
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection("translations")

def save_record(thread_id, source_name, src_lang, tgt_lang, source_text, translated_text):
    if HAS_CHROMA:
        col = get_collection()
        col.add(
            ids=[uuid.uuid4().hex],
            documents=[translated_text],
            metadatas=[{
                "thread_id": thread_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_name": source_name,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "source_preview": source_text[:500],
                "chars": len(translated_text),
            }],
        )
    else:
        st.session_state.setdefault("history", []).append({
            "id": uuid.uuid4().hex,
            "doc": translated_text,
            "meta": {
                "thread_id": thread_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_name": source_name,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "source_preview": source_text[:500],
                "chars": len(translated_text),
            },
        })

def load_records(thread_id=None, search=""):
    if HAS_CHROMA:
        col = get_collection()
        where = {"thread_id": thread_id} if thread_id else None
        if search.strip():
            total = col.count()
            if total == 0:
                return []
            res = col.query(query_texts=[search.strip()], n_results=min(20, total), where=where)
            ids, docs, metas = res["ids"][0], res["documents"][0], res["metadatas"][0]
        else:
            res = col.get(where=where)
            ids, docs, metas = res["ids"], res["documents"], res["metadatas"]
        records = [{"id": i, "doc": d, "meta": m} for i, d, m in zip(ids, docs, metas)]
    else:
        records = st.session_state.get("history", [])
        if thread_id:
            records = [r for r in records if r["meta"]["thread_id"] == thread_id]
        if search.strip():
            q = search.strip().lower()
            records = [r for r in records if q in r["doc"].lower() or q in r["meta"]["source_preview"].lower()]

    records.sort(key=lambda r: r["meta"].get("timestamp", ""), reverse=True)
    return records[:50]

# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_model():
    path = snapshot_download(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(path)
    translator = ctranslate2.Translator(path, device="cpu", compute_type="int8", inter_threads=1, intra_threads=2)
    return tokenizer, translator

def translate_batch(texts, src_code, tgt_code):
    tokenizer, translator = load_model()
    tokenizer.src_lang = src_code
    batch = [tokenizer.convert_ids_to_tokens(tokenizer(t, truncation=True, max_length=256).input_ids) for t in texts]
    results = translator.translate_batch(
        batch, target_prefix=[[tgt_code]] * len(batch), beam_size=2, max_decoding_length=300
    )
    outputs = []
    for r in results:
        tokens = r.hypotheses[0]
        if tokens and tokens[0] == tgt_code:
            tokens = tokens[1:]
        outputs.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True))
    return outputs

def translate_stream(text, src_code, tgt_code, on_partial=None):
    """Translate one string, calling on_partial(partial_text) as tokens are generated.
    Streaming needs beam_size=1. Falls back to normal batch translation if the
    installed ctranslate2 version has no `callback` support."""
    tokenizer, translator = load_model()
    tokenizer.src_lang = src_code
    source = tokenizer.convert_ids_to_tokens(tokenizer(text, truncation=True, max_length=256).input_ids)

    generated = []

    def decode(tokens):
        toks = [t for t in tokens if t != tgt_code]
        return tokenizer.decode(tokenizer.convert_tokens_to_ids(toks), skip_special_tokens=True)

    def callback(step):
        if step.token == tgt_code:
            return
        generated.append(step.token)
        # update the UI every 2 tokens to keep Streamlit responsive
        if on_partial and (len(generated) % 2 == 0 or step.is_last):
            on_partial(decode(generated))

    try:
        results = translator.translate_batch(
            [source],
            target_prefix=[[tgt_code]],
            beam_size=1,
            max_decoding_length=300,
            callback=callback,
        )
    except TypeError:
        # Older ctranslate2: no streaming callback available
        out = translate_batch([text], src_code, tgt_code)[0]
        if on_partial:
            on_partial(out)
        return out

    return decode(results[0].hypotheses[0])

# ------------------------------------------------------------------ helpers
def split_long(line):
    if len(line) <= MAX_CHUNK_CHARS:
        return [line]
    sentences = re.split(r"(?<=[.!?।።؟])\s+", line)
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > MAX_CHUNK_CHARS:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks

def read_upload(uploaded):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "txt", "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "txt", "\n".join(p.text for p in doc.paragraphs)
    if name.endswith(".json"):
        return "json", json.loads(data.decode("utf-8", errors="replace"))
    return "txt", data.decode("utf-8", errors="replace")

def map_strings(obj, fn):
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [map_strings(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: map_strings(v, fn) for k, v in obj.items()}
    return obj

def collect_strings(obj, acc):
    map_strings(obj, lambda s: acc.append(s) or s)
    return acc

# ------------------------------------------------------------------ PDF (self-contained Unicode support)
# Fonts are downloaded once at runtime into a temp folder, so no fonts/ folder
# needs to be committed to the repo.
FONT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "noto_fonts_cache")
NOTO_BASE = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/{n}/hinted/ttf/{n}-Regular.ttf"

# Base font (Latin / Cyrillic / Greek) - always loaded
BASE_FONT = ("NotoSans", NOTO_BASE.format(n="NotoSans"))

# Extra script font needed for each target language
SCRIPT_FONTS = {
    "Hindi":     ("NotoDeva",  NOTO_BASE.format(n="NotoSansDevanagari")),
    "Marathi":   ("NotoDeva",  NOTO_BASE.format(n="NotoSansDevanagari")),
    "Telugu":    ("NotoTelu",  NOTO_BASE.format(n="NotoSansTelugu")),
    "Tamil":     ("NotoTaml",  NOTO_BASE.format(n="NotoSansTamil")),
    "Kannada":   ("NotoKnda",  NOTO_BASE.format(n="NotoSansKannada")),
    "Malayalam": ("NotoMlym",  NOTO_BASE.format(n="NotoSansMalayalam")),
    "Bengali":   ("NotoBeng",  NOTO_BASE.format(n="NotoSansBengali")),
    "Urdu":      ("NotoArab",  NOTO_BASE.format(n="NotoSansArabic")),
    "Arabic":    ("NotoArab",  NOTO_BASE.format(n="NotoSansArabic")),
    # CJK: best effort (large files; variable fonts)
    "Chinese (Simplified)": ("NotoSC", "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"),
    "Japanese":             ("NotoJP", "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"),
}
RTL_LANGS = {"Urdu", "Arabic"}


@st.cache_resource(show_spinner="Downloading PDF font (first time only)...")
def get_font_file(family, url):
    """Download a font once and return its local path (or None on failure)."""
    try:
        os.makedirs(FONT_CACHE_DIR, exist_ok=True)
        path = os.path.join(FONT_CACHE_DIR, f"{family}.ttf")
        if not (os.path.exists(path) and os.path.getsize(path) > 10_000):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as f:
                f.write(resp.read())
        return path
    except Exception:
        return None


def _latin1_safe(text):
    """Last-resort: replace characters Helvetica can't encode so fpdf never crashes."""
    return text.encode("latin-1", "replace").decode("latin-1")


@st.cache_data(show_spinner=False)
def make_pdf(lines, target_lang="English"):
    lines = list(lines)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    unicode_ok = False
    base_family, base_url = BASE_FONT
    base_path = get_font_file(base_family, base_url)
    if base_path:
        try:
            pdf.add_font(base_family, "", base_path)
            pdf.set_font(base_family, size=11)
            unicode_ok = True
        except Exception:
            unicode_ok = False

    if unicode_ok:
        fallbacks = []
        if target_lang in SCRIPT_FONTS:
            fam, url = SCRIPT_FONTS[target_lang]
            p = get_font_file(fam, url)
            if p:
                try:
                    pdf.add_font(fam, "", p)
                    fallbacks.append(fam)
                except Exception:
                    pass
        if fallbacks:
            pdf.set_fallback_fonts(fallbacks)
        # Proper shaping for Indic / Arabic scripts (needs `uharfbuzz`)
        try:
            pdf.set_text_shaping(True)
        except Exception:
            pass
    else:
        pdf.set_font("Helvetica", size=11)

    align = "R" if target_lang in RTL_LANGS else "L"

    for line in lines:
        if not line.strip():
            pdf.ln(5)
            continue
        text = line if unicode_ok else _latin1_safe(line)
        try:
            pdf.multi_cell(pdf.epw, 6, text, new_x="LMARGIN", new_y="NEXT", align=align)
        except Exception:
            # Never let one bad line kill the whole download
            pdf.multi_cell(pdf.epw, 6, _latin1_safe(text), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def make_docx(lines):
    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

# ------------------------------------------------------------------ UI
tcol1, tcol2 = st.columns([3, 1])
tcol1.text_input("Thread ID (system-generated)", value=st.session_state.thread_id, disabled=True)
tcol2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
if tcol2.button("New thread"):
    st.session_state.thread_id = new_thread_id()
    st.session_state.result = None
    st.session_state.streamed_text = ""
    st.rerun()

col1, col2 = st.columns(2)
src_name = col1.selectbox("Translate from", list(LANGUAGES), index=0)
tgt_name = col2.selectbox("Translate to", list(LANGUAGES), index=1)

uploaded = st.file_uploader("Upload a file", type=["pdf", "txt", "docx", "json"])
typed = st.text_area("...or paste text", height=120)

st.subheader("Streaming Output")
output_placeholder = st.empty()
output_placeholder.text_area(
    "Output",
    st.session_state.streamed_text,
    height=280,
    key="stream_display",
)

def show(text):
    """Push text into the Streaming Output box (fresh key each call so it always refreshes)."""
    st.session_state.streamed_text = text
    output_placeholder.text_area("Output", text, height=280, key=f"out_{uuid.uuid4().hex}")

if st.button("Translate", type="primary"):
    if src_name == tgt_name:
        st.info("Source and target languages are the same.")
    elif not uploaded and not typed.strip():
        st.warning("Upload a file or paste some text first.")
    else:
        st.session_state.result = None
        st.session_state.streamed_text = ""

        try:
            kind, content = read_upload(uploaded) if uploaded else ("txt", typed)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        src, tgt = LANGUAGES[src_name], LANGUAGES[tgt_name]
        bar = st.progress(0.0, text="Translating...")

        if kind == "json":
            strings = collect_strings(content, [])
            unique = list(dict.fromkeys(s for s in strings if s.strip()))
            done = {}
            accumulated = []

            for n, s in enumerate(unique):
                base = "\n".join(accumulated)
                t = translate_stream(
                    s, src, tgt,
                    on_partial=lambda p, base=base: show(f"{base}\n{p}" if base else p),
                )
                done[s] = t
                accumulated.append(t)
                show("\n".join(accumulated))
                bar.progress((n + 1) / max(len(unique), 1))

            lines = [done.get(s, s) for s in strings]
            source_text = "\n".join(strings)
            result_json = map_strings(content, lambda s: done.get(s, s))
        else:
            src_lines = content.splitlines()
            pieces = [split_long(l.strip()) if l.strip() else [] for l in src_lines]
            flat = [p for chunks in pieces for p in chunks]
            unique = list(dict.fromkeys(s for s in flat if s.strip()))  # document order
            done = {}

            def render(extra=None):
                m = {**done, **(extra or {})}
                rows = [" ".join(m[p] for p in chunks if p in m) for chunks in pieces]
                return "\n".join(rows).rstrip()

            for n, s in enumerate(unique):
                t = translate_stream(
                    s, src, tgt,
                    on_partial=lambda p, s=s: show(render({s: p})),
                )
                done[s] = t
                show(render())
                bar.progress((n + 1) / max(len(unique), 1))

            lines = [" ".join(done.get(p, p) for p in chunks) if chunks else "" for chunks in pieces]
            source_text = content
            result_json = {
                "thread_id": st.session_state.thread_id,
                "source_language": src_name,
                "target_language": tgt_name,
                "translated_text": "\n".join(lines),
            }

        bar.empty()
        final_text = "\n".join(lines)
        st.session_state.streamed_text = final_text
        st.session_state.result = {"lines": lines, "json": result_json, "target": tgt_name}

        try:
            save_record(
                st.session_state.thread_id,
                uploaded.name if uploaded else "pasted text",
                src_name, tgt_name, source_text, final_text,
            )
        except Exception as e:
            st.warning(f"Translation done, but history could not be saved: {e}")

        # No st.rerun() here → UI stays visible with the final result

# ------------------------------------------------------------------ downloads
result = st.session_state.result
if result:
    st.markdown("---")
    st.subheader("Download translated file")
    c1, c2, c3, c4 = st.columns(4)

    try:
        pdf_bytes = make_pdf(tuple(result["lines"]), result.get("target", "English"))
        c1.download_button("📄 PDF", pdf_bytes, "translation.pdf", "application/pdf", use_container_width=True)
    except Exception as e:
        c1.button("📄 PDF (error)", disabled=True, use_container_width=True)
        st.warning(f"PDF could not be generated: {e}")

    c2.download_button("📝 TXT", "\n".join(result["lines"]).encode("utf-8"), "translation.txt", "text/plain", use_container_width=True)
    c3.download_button("📑 DOCX", make_docx(result["lines"]), "translation.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
    c4.download_button("📋 JSON", json.dumps(result["json"], ensure_ascii=False, indent=2).encode("utf-8"),
                       "translation.json", "application/json", use_container_width=True)

# ------------------------------------------------------------------ history
st.divider()
st.subheader("Past records")
if not HAS_CHROMA:
    st.caption("History is session-only (ChromaDB not installed).")

h1, h2 = st.columns([1, 2])
scope = h1.radio("Show", ["This thread", "All threads"], horizontal=True)
search = h2.text_input("Search", placeholder="Type a word...")

try:
    records = load_records(st.session_state.thread_id if scope == "This thread" else None, search)
except Exception as e:
    records = []
    st.warning(f"Could not load records: {e}")

if not records:
    st.caption("No saved records yet.")
else:
    for r in records:
        m = r["meta"]
        title = f"{m.get('timestamp', '')}  ·  {m.get('source_name', '')}  ·  {m.get('source_lang', '')} → {m.get('target_lang', '')}"
        with st.expander(title):
            st.markdown("**Source (preview)**")
            st.text(m.get("source_preview", ""))
            st.markdown("**Translation**")
            st.text(r["doc"])