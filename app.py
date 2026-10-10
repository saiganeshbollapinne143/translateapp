import glob
import io
import json
import os
import re
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
MAX_CHUNK_CHARS = 400
BATCH_SIZE = 4
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
    background: #0B2A5B; border-radius: 12px;
    padding: 14px 22px; margin-bottom: 1.2rem;
}
.title-card h1 {
    color: #FFFFFF !important; font-size: 1.25rem; font-weight: 700;
    margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.title-card h1 .ait { color: #FFC107 !important; }

.stButton > button, .stDownloadButton > button,
[data-testid="stFileUploader"] button {
    background: #FFC107 !important; color: #1B1B1B !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #FFB300 !important; color: #000000 !important;
}
.stDownloadButton > button { width: 100%; }
</style>
<div class="title-card">
    <h1><span class="ait">AIT</span> GLOBAL TECHNOLOGIES</h1>
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

# ------------------------------------------------------------------ PDF (fixed Unicode support)
def find_font():
    candidates = [
        "font.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
    ]
    for p in candidates:
        hits = glob.glob(p)
        if hits and os.path.exists(hits[0]):
            return hits[0]
    return None

def make_pdf(lines):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    font_path = find_font()
    if font_path:
        pdf.add_font("Unicode", "", font_path)
        pdf.set_font("Unicode", size=11)
    else:
        # Last resort – will produce ? for non-Latin text
        pdf.set_font("Helvetica", size=11)
    
    for line in lines:
        if not line.strip():
            pdf.ln(5)
            continue
        # Never force latin-1 replace – let the Unicode font handle it
        pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    
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
            unique.sort(key=len)
            done = {}
            accumulated = []

            for i in range(0, len(unique), BATCH_SIZE):
                batch = unique[i:i + BATCH_SIZE]
                results = translate_batch(batch, src, tgt)
                for s, t in zip(batch, results):
                    done[s] = t
                    accumulated.append(t)
                streamed = "\n".join(accumulated)
                st.session_state.streamed_text = streamed
                output_placeholder.text_area("Output", streamed, height=280, key=f"j_{i}")
                bar.progress(min((i + BATCH_SIZE) / max(len(unique), 1), 1.0))

            lines = [done.get(s, s) for s in strings]
            source_text = "\n".join(strings)
            result_json = map_strings(content, lambda s: done.get(s, s))
        else:
            src_lines = content.splitlines()
            pieces = [split_long(l.strip()) if l.strip() else [] for l in src_lines]
            flat = [p for chunks in pieces for p in chunks]
            unique = list(dict.fromkeys(s for s in flat if s.strip()))
            unique.sort(key=len)
            done = {}

            for i in range(0, len(unique), BATCH_SIZE):
                batch = unique[i:i + BATCH_SIZE]
                results = translate_batch(batch, src, tgt)
                for s, t in zip(batch, results):
                    done[s] = t

                current_lines = [" ".join(done.get(p, p) for p in chunks) if chunks else "" for chunks in pieces]
                streamed = "\n".join(current_lines)
                st.session_state.streamed_text = streamed
                output_placeholder.text_area("Output", streamed, height=280, key=f"t_{i}")
                bar.progress(min((i + BATCH_SIZE) / max(len(unique), 1), 1.0))

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
        st.session_state.result = {"lines": lines, "json": result_json}

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
    
    c1.download_button("📄 PDF", make_pdf(result["lines"]), "translation.pdf", "application/pdf", use_container_width=True)
    c2.download_button("📝 TXT", "\n".join(result["lines"]).encode("utf-8"), "translation.txt", "text/plain", use_container_width=True)
    c3.download_button("📑 DOCX", make_docx(result["lines"]), "translation.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
    c4.download_button("📋 JSON", json.dumps(result["json"], ensure_ascii=False, indent=2).encode("utf-8"),
                       "translation.json", "application/json", use_container_width=True)

    if not find_font():
        st.warning("No Unicode font found. PDF may show '?' for non-Latin text. Install Noto Sans or DejaVu fonts.")

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