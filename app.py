import hashlib
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

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# ------------------------------------------------------------------ config
MODEL_NAME = "olob0/nllb-200-distilled-600M-ct2-int8_float16"
MAX_PAGES = 100          # PDF page limit
MAX_CHUNK_CHARS = 400    # characters per translation chunk
BATCH_SIZE = 16          # chunks translated together
STREAM_UNDER = 20        # live token streaming only for small inputs
UI_EVERY = 4             # refresh output box every N batches
CKPT_EVERY = 5           # save resume checkpoint every N batches
TAIL_CHARS = 5000        # max live output sent to the browser
STORE_CHUNK = 1000       # characters per stored history chunk
PREVIEW_CHARS = 1000     # preview per history record
MAX_SHOWN = 10           # history records shown at once
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
CKPT_DIR = os.path.join(tempfile.gettempdir(), "translate_ckpt")

LANGUAGES = {
    # priority languages
    "English": "eng_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
    "French": "fra_Latn",
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    # others
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
    "Urdu": "urd_Arab",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
    "Nepali": "npi_Deva",
    "Portuguese": "por_Latn",
    "Dutch": "nld_Latn",
    "Turkish": "tur_Latn",
    "Russian": "rus_Cyrl",
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
.title-card { background: #0B2A5B; border-radius: 12px; padding: 14px 22px;
    margin-bottom: 1.2rem; text-align: center; }
.title-card h1 { margin: 0; padding: 0; font-size: 1.35rem; font-weight: 700;
    letter-spacing: 0.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.title-card h1 .ait { color: #FFC107 !important; }
.title-card h1 .rest { color: #FFFFFF !important; }
.stButton > button, .stDownloadButton > button,
[data-testid="stFileUploader"] button {
    background: #FFC107 !important; color: #1B1B1B !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; }
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #FFB300 !important; color: #000000 !important; }
.stDownloadButton > button { width: 100%; }
</style>
<div class="title-card">
    <h1><span class="ait">AIT</span><span class="rest"> GLOBAL TECHNOLOGIES</span></h1>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ session
def new_thread_id():
    return "THR-" + uuid.uuid4().hex[:8].upper()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()
if "streamed_text" not in st.session_state:
    st.session_state.streamed_text = ""
if "result" not in st.session_state:
    st.session_state.result = None

def tail(text):
    return text if len(text) <= TAIL_CHARS else "…" + text[-TAIL_CHARS:]

# ------------------------------------------------------------------ resume checkpoints
def job_key(content, src, tgt):
    raw = content if isinstance(content, str) else json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(f"{src}|{tgt}|{raw}".encode("utf-8")).hexdigest()

def _ckpt_file(key):
    return os.path.join(CKPT_DIR, f"{key}.json")

def load_ckpt(key):
    try:
        with open(_ckpt_file(key), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_ckpt(key, done):
    try:
        os.makedirs(CKPT_DIR, exist_ok=True)
        tmp = _ckpt_file(key) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(done, f, ensure_ascii=False)
        os.replace(tmp, _ckpt_file(key))
    except Exception:
        pass

def cleanup_ckpts(max_age_hours=24):
    try:
        for name in os.listdir(CKPT_DIR):
            p = os.path.join(CKPT_DIR, name)
            if time.time() - os.path.getmtime(p) > max_age_hours * 3600:
                os.remove(p)
    except Exception:
        pass

cleanup_ckpts()

# ------------------------------------------------------------------ history (chunked)
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection("translations_v2")

def _and(*conds):
    conds = [c for c in conds if c]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}

def save_record(thread_id, source_name, src_lang, tgt_lang, source_text, translated_text):
    record_id = uuid.uuid4().hex
    meta = {
        "record_id": record_id,
        "thread_id": thread_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": source_name,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "source_preview": source_text[:300],
        "chars": len(translated_text),
    }
    if HAS_CHROMA:
        col = get_collection()
        chunks = [translated_text[i:i + STORE_CHUNK] for i in range(0, len(translated_text), STORE_CHUNK)] or [""]
        for start in range(0, len(chunks), 100):
            part = chunks[start:start + 100]
            col.add(
                ids=[f"{record_id}-{start + j}" for j in range(len(part))],
                documents=part,
                metadatas=[{**meta, "chunk_index": start + j} for j in range(len(part))],
            )
    else:
        st.session_state.setdefault("history", []).append({"meta": meta, "full": translated_text})

def load_records(thread_id=None, search=""):
    items = []
    if HAS_CHROMA:
        col = get_collection()
        if col.count() == 0:
            return []
        tw = {"thread_id": thread_id} if thread_id else None
        if search.strip():
            res = col.query(query_texts=[search.strip()], n_results=min(30, col.count()), where=tw)
            seen = set()
            for d, m in zip(res["documents"][0], res["metadatas"][0]):
                if m["record_id"] in seen:
                    continue
                seen.add(m["record_id"])
                items.append({"id": m["record_id"], "text": d, "meta": m, "match": True})
        else:
            res = col.get(where=_and({"chunk_index": 0}, tw), limit=200)
            for d, m in zip(res["documents"], res["metadatas"]):
                items.append({"id": m["record_id"], "text": d, "meta": m, "match": False})
    else:
        q = search.strip().lower()
        for r in st.session_state.get("history", []):
            if thread_id and r["meta"]["thread_id"] != thread_id:
                continue
            if q:
                pos = r["full"].lower().find(q)
                if pos < 0:
                    continue
                items.append({"id": r["meta"]["record_id"], "text": r["full"][max(0, pos - 200):pos + 800],
                              "meta": r["meta"], "match": True})
            else:
                items.append({"id": r["meta"]["record_id"], "text": r["full"][:PREVIEW_CHARS],
                              "meta": r["meta"], "match": False})
    if not search.strip():
        items.sort(key=lambda r: r["meta"].get("timestamp", ""), reverse=True)
    return items[:MAX_SHOWN]

def fetch_full(record_id):
    if HAS_CHROMA:
        res = get_collection().get(where={"record_id": record_id})
        pairs = sorted(zip([m["chunk_index"] for m in res["metadatas"]], res["documents"]))
        return "".join(d for _, d in pairs)
    for r in st.session_state.get("history", []):
        if r["meta"]["record_id"] == record_id:
            return r["full"]
    return ""

# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_model():
    path = snapshot_download(MODEL_NAME)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(path)
    translator = ctranslate2.Translator(
        path, device="cpu", compute_type="int8",
        inter_threads=1, intra_threads=os.cpu_count() or 2,
    )
    return tokenizer, translator

def translate_batch(texts, src_code, tgt_code):
    tokenizer, translator = load_model()
    tokenizer.src_lang = src_code
    ids = tokenizer(texts, truncation=True, max_length=256).input_ids
    batch = [tokenizer.convert_ids_to_tokens(i) for i in ids]
    results = translator.translate_batch(
        batch,
        target_prefix=[[tgt_code]] * len(batch),
        beam_size=1,
        max_decoding_length=256,
        max_batch_size=2048,
        batch_type="tokens",
    )
    outputs = []
    for r in results:
        tokens = r.hypotheses[0]
        if tokens and tokens[0] == tgt_code:
            tokens = tokens[1:]
        outputs.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True))
    return outputs

def translate_stream(text, src_code, tgt_code, on_partial=None):
    tokenizer, translator = load_model()
    tokenizer.src_lang = src_code
    source = tokenizer.convert_ids_to_tokens(tokenizer(text, truncation=True, max_length=256).input_ids)
    generated = []

    def decode(tokens):
        toks = [t for t in tokens if t != tgt_code]
        return tokenizer.decode(tokenizer.convert_tokens_to_ids(toks), skip_special_tokens=True)

    def callback(step):  # runs in a worker thread: no st.* calls here
        if step.token != tgt_code:
            generated.append(step.token)
        return None

    try:
        futures = translator.translate_batch(
            [source], target_prefix=[[tgt_code]], beam_size=1,
            max_decoding_length=256, callback=callback, asynchronous=True,
        )
    except TypeError:
        out = translate_batch([text], src_code, tgt_code)[0]
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
    return decode(future.result().hypotheses[0])

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
        return "txt", "\n\n".join((p.extract_text() or "") for p in reader.pages[:MAX_PAGES])
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

# ------------------------------------------------------------------ PDF output
FONT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "noto_fonts_cache")
NOTO_BASE = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/{n}/hinted/ttf/{n}-Regular.ttf"
BASE_FONT = ("NotoSans", NOTO_BASE.format(n="NotoSans"))  # Latin + Cyrillic + Greek
SCRIPT_FONTS = {
    "Hindi":     ("NotoDeva", NOTO_BASE.format(n="NotoSansDevanagari")),
    "Marathi":   ("NotoDeva", NOTO_BASE.format(n="NotoSansDevanagari")),
    "Nepali":    ("NotoDeva", NOTO_BASE.format(n="NotoSansDevanagari")),
    "Telugu":    ("NotoTelu", NOTO_BASE.format(n="NotoSansTelugu")),
    "Tamil":     ("NotoTaml", NOTO_BASE.format(n="NotoSansTamil")),
    "Kannada":   ("NotoKnda", NOTO_BASE.format(n="NotoSansKannada")),
    "Malayalam": ("NotoMlym", NOTO_BASE.format(n="NotoSansMalayalam")),
    "Bengali":   ("NotoBeng", NOTO_BASE.format(n="NotoSansBengali")),
    "Gujarati":  ("NotoGujr", NOTO_BASE.format(n="NotoSansGujarati")),
    "Punjabi":   ("NotoGuru", NOTO_BASE.format(n="NotoSansGurmukhi")),
    "Urdu":      ("NotoArab", NOTO_BASE.format(n="NotoSansArabic")),
    "Arabic":    ("NotoArab", NOTO_BASE.format(n="NotoSansArabic")),
    "Chinese (Simplified)": ("NotoSC", "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"),
    "Japanese":             ("NotoJP", "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"),
}
RTL_LANGS = {"Urdu", "Arabic"}

@st.cache_resource(show_spinner="Downloading PDF font (first time only)...")
def get_font_file(family, url):
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
    return text.encode("latin-1", "replace").decode("latin-1")

@st.cache_data(show_spinner=False, max_entries=1)
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
_langs = list(LANGUAGES)
src_name = col1.selectbox("Translate from", _langs, index=_langs.index("English"))
tgt_name = col2.selectbox("Translate to", _langs, index=_langs.index("Telugu"))

uploaded = st.file_uploader("Upload a file (PDF: first 100 pages)", type=["pdf", "txt", "docx", "json"])
typed = st.text_area("...or paste text", height=120)

st.subheader("Streaming Output")
output_placeholder = st.empty()
output_placeholder.text_area("Output", tail(st.session_state.streamed_text), height=280, key="stream_display")

def show(text):
    st.session_state.streamed_text = text
    output_placeholder.text_area("Output", tail(text), height=280, key=f"out_{uuid.uuid4().hex}")

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
        key = job_key(content, src, tgt)
        done = load_ckpt(key)

        if kind == "json":
            strings = collect_strings(content, [])
            unique = list(dict.fromkeys(s for s in strings if s.strip()))
            todo = [s for s in unique if s not in done]
            if done and todo:
                st.info(f"Resuming: {len(done)} strings already done, {len(todo)} left.")

            order = sorted(todo, key=len)
            for b, i in enumerate(range(0, len(order), BATCH_SIZE)):
                batch = order[i:i + BATCH_SIZE]
                for s, t in zip(batch, translate_batch(batch, src, tgt)):
                    done[s] = t
                bar.progress(min(i + BATCH_SIZE, len(order)) / max(len(order), 1))
                if b % CKPT_EVERY == 0:
                    save_ckpt(key, done)
            save_ckpt(key, done)

            lines = [done.get(s, s) for s in strings]
            source_text = "\n".join(strings)
            result_json = map_strings(content, lambda s: done.get(s, s))
            show("\n".join(lines))
        else:
            src_lines = content.splitlines()
            pieces = [split_long(l.strip()) if l.strip() else [] for l in src_lines]
            flat = [p for chunks in pieces for p in chunks]
            unique = list(dict.fromkeys(s for s in flat if s.strip()))
            todo = [s for s in unique if s not in done]
            if done and todo:
                st.info(f"Resuming: {len(done)} chunks already done, {len(todo)} left.")

            def render(extra=None):
                m = {**done, **(extra or {})}
                rows = [" ".join(m[p] for p in chunks if p in m) for chunks in pieces]
                return "\n".join(rows).rstrip()

            if len(todo) <= STREAM_UNDER:
                for n, s in enumerate(todo):
                    t = translate_stream(s, src, tgt, on_partial=lambda p, s=s: show(render({s: p})))
                    done[s] = t
                    show(render())
                    bar.progress((n + 1) / max(len(todo), 1))
            else:
                order = sorted(todo, key=len)  # similar lengths = less padding
                total = len(order)
                for b, i in enumerate(range(0, total, BATCH_SIZE)):
                    batch = order[i:i + BATCH_SIZE]
                    for s, t in zip(batch, translate_batch(batch, src, tgt)):
                        done[s] = t
                    count = min(i + BATCH_SIZE, total)
                    bar.progress(count / total, text=f"Translating {count}/{total} chunks")
                    if b % UI_EVERY == 0:
                        show(render())
                    if b % CKPT_EVERY == 0:
                        save_ckpt(key, done)
            save_ckpt(key, done)
            show(render())

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
            with st.spinner("Saving to history..."):
                save_record(
                    st.session_state.thread_id,
                    uploaded.name if uploaded else "pasted text",
                    src_name, tgt_name, source_text, final_text,
                )
        except Exception as e:
            st.warning(f"Translation done, but history could not be saved: {e}")

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
    if search.strip():
        st.caption(f"Best matches (up to {MAX_SHOWN}). Open a record to see the matching passage.")
    for r in records:
        m = r["meta"]
        title = f"{m.get('timestamp', '')}  ·  {m.get('source_name', '')}  ·  {m.get('source_lang', '')} → {m.get('target_lang', '')}"
        with st.expander(title):
            st.markdown("**Source (preview)**")
            st.text(m.get("source_preview", "")[:300])
            st.markdown("**Matching passage**" if r["match"] else "**Translation (preview)**")
            st.text(r["text"][:PREVIEW_CHARS])
            st.caption(f"{m.get('chars', 0):,} characters in full translation")

            pkey = f"prep_{r['id']}"
            if st.session_state.get(pkey):
                st.download_button(
                    "⬇️ Download full translation", fetch_full(r["id"]).encode("utf-8"),
                    file_name=f"{os.path.splitext(m.get('source_name', 'translation'))[0]}_{m.get('target_lang', '')}.txt",
                    key=f"dl_{r['id']}",
                )
            else:
                st.button("Prepare full download", key=f"btn_{r['id']}",
                          on_click=lambda k=pkey: st.session_state.__setitem__(k, True))