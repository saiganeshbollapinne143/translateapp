import glob
import io
import json
import os
import re

import streamlit as st
import torch
from docx import Document
from fpdf import FPDF
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
MAX_CHUNK_CHARS = 400
BATCH_SIZE = 4

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

torch.set_num_threads(2)

# ---------------------------------------------------------------- page + style
st.set_page_config(page_title="Document Translator", page_icon="🌍", layout="centered")

st.markdown(
    """
    <style>
    .stApp, [data-testid="stHeader"] { background: #FFFFFF; }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li { color: #1B1B1B; }
    .block-container { padding-top: 4.5rem !important; max-width: 820px; }

    .title-card {
        background: #0B2A5B; border-radius: 12px;
        padding: 14px 22px; margin-bottom: 1.2rem;
    }
    .title-card h1 {
        color: #FFFFFF !important; font-size: 1.25rem; font-weight: 600;
        margin: 0; padding: 0; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis;
    }

    .stButton > button, .stDownloadButton > button,
    [data-testid="stFileUploader"] button {
        background: #FFC107 !important; color: #1B1B1B !important;
        border: none !important; border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFileUploader"] button:hover {
        background: #FFB300 !important; color: #000000 !important;
    }
    .stDownloadButton > button { width: 100%; }
    </style>
    <div class="title-card"><h1>🌍 Document Translator &nbsp;·&nbsp; PDF, TXT, DOCX, JSON</h1></div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model


def translate_batch(texts, src_code, tgt_code):
    tokenizer, model = load_model()
    tokenizer.src_lang = src_code
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=256
    )
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
            max_new_tokens=300,
            num_beams=2,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def translate_many(strings, src_code, tgt_code, progress):
    """Translate a list of strings (batched, with de-duplication)."""
    unique = list(dict.fromkeys(s for s in strings if s.strip()))
    unique.sort(key=len)  # similar lengths per batch = less padding
    done = {}
    for i in range(0, len(unique), BATCH_SIZE):
        batch = unique[i : i + BATCH_SIZE]
        for src, tgt in zip(batch, translate_batch(batch, src_code, tgt_code)):
            done[src] = tgt
        progress.progress(min((i + BATCH_SIZE) / max(len(unique), 1), 1.0))
    return done


# ------------------------------------------------------------------- read files
def split_long(line):
    """Split a long line into chunks of <= MAX_CHUNK_CHARS at sentence ends."""
    if len(line) <= MAX_CHUNK_CHARS:
        return [line]
    sentences = re.split(r"(?<=[.!?।。؟])\s+", line)
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
    """Apply fn to every string VALUE in a JSON structure (keys untouched)."""
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


# ----------------------------------------------------------------- build files
def find_font():
    if os.path.exists("font.ttf"):
        return "font.ttf"
    patterns = [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans*-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
    ]
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def make_pdf(lines):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    font = find_font()
    if font:
        pdf.add_font("U", "", font)
        pdf.set_font("U", size=11)
    else:
        pdf.set_font("Helvetica", size=11)
    for line in lines:
        if not line.strip():
            pdf.ln(5)
            continue
        if not font:
            line = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def make_docx(lines):
    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# -------------------------------------------------------------------------- UI
col1, col2 = st.columns(2)
src_name = col1.selectbox("Translate from", list(LANGUAGES), index=0)
tgt_name = col2.selectbox("Translate to", list(LANGUAGES), index=1)

uploaded = st.file_uploader("Upload a file", type=["pdf", "txt", "docx", "json"])
typed = st.text_area("...or paste text", height=120)

if st.button("Translate", type="primary"):
    if src_name == tgt_name:
        st.info("Source and target languages are the same.")
    elif not uploaded and not typed.strip():
        st.warning("Upload a file or paste some text first.")
    else:
        try:
            kind, content = read_upload(uploaded) if uploaded else ("txt", typed)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        src, tgt = LANGUAGES[src_name], LANGUAGES[tgt_name]
        bar = st.progress(0.0, text="Translating...")

        if kind == "json":
            strings = collect_strings(content, [])
            done = translate_many(strings, src, tgt, bar)
            result_json = map_strings(content, lambda s: done.get(s, s))
            lines = [done.get(s, s) for s in strings]
        else:
            src_lines = content.splitlines()
            pieces = [split_long(l.strip()) if l.strip() else [] for l in src_lines]
            flat = [p for chunks in pieces for p in chunks]
            done = translate_many(flat, src, tgt, bar)
            lines = [" ".join(done.get(p, p) for p in chunks) for chunks in pieces]
            result_json = {
                "source_language": src_name,
                "target_language": tgt_name,
                "translated_text": "\n".join(lines),
            }

        bar.empty()
        st.session_state["result"] = {"lines": lines, "json": result_json}

result = st.session_state.get("result")
if result:
    st.subheader("Translation")
    st.text_area("Output", "\n".join(result["lines"]), height=220)

    st.markdown("**Download**")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("PDF", make_pdf(result["lines"]), "translation.pdf", "application/pdf")
    c2.download_button("TXT", "\n".join(result["lines"]).encode("utf-8"), "translation.txt", "text/plain")
    c3.download_button(
        "DOCX",
        make_docx(result["lines"]),
        "translation.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    c4.download_button(
        "JSON",
        json.dumps(result["json"], ensure_ascii=False, indent=2).encode("utf-8"),
        "translation.json",
        "application/json",
    )
    if not find_font():
        st.caption("PDF note: no Unicode font found, so non-Latin text may show as '?'. See setup notes.")