import io
import json
import os
import re
import tempfile
import time
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
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "5"
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
MAX_UPLOAD_MB = 5
LARGE_THRESHOLD = 100      # more unique chunks than this -> batched mode
BIG_BATCH = 8
DISPLAY_LIMIT = 20000      # max characters shown in the output box
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
.title-card h1 .ait  { color: #FFC107 !important; }
.title-card h1 .rest { color: #FFFFFF !important; }

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
    meta = {
        "thread_id": thread_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": source_name,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "source_preview": source_text[:500],
        "chars": len(translated_text),
    }
    doc = translated_text[:5000]  # keep history light; full text is in the downloads
    if HAS_CHROMA:
        get_collection().add(ids=[uuid.uuid4().hex], documents=[doc], metadatas=[meta])
    else:
        st.session_state.setdefault("history", []).append(
            {"id": uuid.uuid4().hex, "doc": doc, "meta": meta}
        )

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

# ------------------------------------------------------------------ duplicate / noise cleaning
def collapse_loops(text, max_n=10):
    """Remove meaningless repetition loops the model sometimes produces
    (e.g. 'the house the house the house' or 'word word word word')."""
    out = []
    for w in text.split(" "):
        out.append(w)
        if len(out) >= 3 and out[-1] == out[-2] == out[-3]:
            del out[-1]                      # same word 3+ times in a row
            continue
        for n in range(2, max_n + 1):        # a phrase immediately repeated
            if len(out) >= 2 * n and out[-n:] == out[-2 * n:-n]:
                del out[-n:]
                break
    return " ".join(out)

def dedupe_lines(lines, remove_global):
    """Drop blank runs, lines with no letters (page numbers, '----'), consecutive
    duplicate lines, and (optionally) any line that already appeared earlier."""
    seen, out, prev = set(), [], None
    for line in lines:
        s = line.strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        if not re.search(r"[^\W\d_]", s):    # no letters at all -> no meaning
            continue
        key = re.sub(r"\s+", " ", s).lower()
        if key == prev:
            continue
        if remove_global and key in seen:
            continue
        seen.add(key)
        prev = key
        out.append(s)
    while out and out[-1] == "":
        out.pop()
    return out

# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_model():
    path = snapshot_download(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(path)
    translator = ctranslate2.Translator(path, device="cpu", compute_type="int8", inter_threads=1, intra_threads=2)
    return tokenizer, translator

def translate_batch(texts, src_code, tgt_code, clean=True):
    tokenizer, translator = load_model()
    tokenizer.src_lang = src_code
    batch = [tokenizer.convert_ids_to_tokens(tokenizer(t, truncation=True, max_length=256).input_ids) for t in texts]
    results = translator.translate_batch(
        batch,
        target_prefix=[[tgt_code]] * len(batch),
        beam_size=2,
        max_decoding_length=300,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
    )
    outputs = []
    for r in results:
        tokens = r.hypotheses[0]
        if tokens and tokens[0] == tgt_code:
            tokens = tokens[1:]
        text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)
        outputs.append(collapse_loops(text) if clean else text)
    return outputs

def translate_stream(text, src_code, tgt_code, on_partial=None, clean=True):
    """Translate one string, calling on_partial(partial_text) as tokens are generated.
    Falls back to normal batch translation if ctranslate2 has no `callback` support."""
    tokenizer, translator = load_model()
    tokenizer.src_lang = src_code
    source = tokenizer.convert_ids_to_tokens(tokenizer(text, truncation=True, max_length=256).input_ids)

    generated = []

    def decode(tokens):
        toks = [t for t in tokens if t != tgt_code]
        out = tokenizer.decode(tokenizer.convert_tokens_to_ids(toks), skip_special_tokens=True)
        return collapse_loops(out) if clean else out

    # Called from CTranslate2's worker thread: must NOT touch st.* elements.
    def callback(step):
        if step.token != tgt_code:
            generated.append(step.token)
        return None

    try:
        futures = translator.translate_batch(
            [source],
            target_prefix=[[tgt_code]],
            beam_size=1,
            max_decoding_length=300,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            callback=callback,
            asynchronous=True,
        )
    except TypeError:
        out = translate_batch([text], src_code, tgt_code, clean)[0]
        if on_partial:
            on_partial(out)
        return out

    future = futures[0]
    last_n = 0
    while not future.done():
        n = len(generated)
        if on_partial and n != last_n:
            on_partial(decode(list(generated)))
            last_n = n
        time.sleep(0.05)

    result = future.result()
    return decode(result.hypotheses[0])

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

def preview(text):
    if len(text) <= DISPLAY_LIMIT:
        return text
    return text[:DISPLAY_LIMIT] + "\n\n... (preview only - use the download buttons for the full text)"

# ------------------------------------------------------------------ PDF (self-contained
st.write("DEBUG 1")   # right after the title-card st.markdown(...) block
st.write("DEBUG 2")   # right before the line: tcol1, tcol2 = st.columns([3, 1])
st.write("DEBUG 3") 