# -----------------------------------------------------------------------------
# SQLITE PATCH FOR CHROMADB COMPATIBILITY (MUST BE AT TOP)
# -----------------------------------------------------------------------------
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import io, os, re, uuid, textwrap
from datetime import datetime
import streamlit as st
import torch
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -----------------------------------------------------------------------------
# SECRETS & CONFIGURATION MANAGEMENT
# -----------------------------------------------------------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
COMPANY_NAME = st.secrets.get("COMPANY_NAME", "AIT GLOBAL")
APP_TITLE = st.secrets.get("APP_TITLE", "AIT GLOBAL TECHNOLOGIES - TRANSLATOR")
MODEL_NAME = st.secrets.get("MODEL_NAME", "facebook/nllb-200-distilled-600M")
DEFAULT_BATCH_SIZE = int(st.secrets.get("DEFAULT_BATCH_SIZE", 16))

st.set_page_config(page_title="AI Translator", page_icon="🌍", layout="wide")

# Custom CSS styling
st.markdown(
    f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1100px; }}
.stButton>button {{ border-radius: 10px; font-weight: 600; }}
.banner {{
    background-color: #0B2D5C;
    padding: 14px 6px;
    border-radius: 10px;
    text-align: center;
    white-space: nowrap;
    font-size: clamp(12px, 2.5vw, 30px);
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 20px;
    width: 100%;
    box-sizing: border-box;
}}
.stream-box {{
    background-color: #ffffff;
    border: 1px solid #dcdcdc;
    border-radius: 8px;
    padding: 20px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 13.5px;
    line-height: 1.5;
    white-space: pre-wrap;
    min-height: 250px;
    max-height: 500px;
    overflow-y: auto;
    color: #111111;
}}
</style>
<div class="banner">
    <span style="color:#FFD700">{COMPANY_NAME}</span>
    <span style="color:#FFFFFF"> TECHNOLOGIES - TRANSLATOR</span>
</div>
""",
    unsafe_allow_html=True,
)

LANGS = {
    "English": "eng_Latn",
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Chinese": "zho_Hans",
    "Arabic": "arb_Arab",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Italian": "ita_Latn",
}

WIN = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
FONTS = {
    "Tamil": ["Nirmala.ttf"],
    "Hindi": ["Nirmala.ttf"],
    "Telugu": ["Nirmala.ttf"],
    "Chinese": ["msyh.ttc", "simsun.ttc"],
    "Japanese": ["YuGothR.ttc", "msgothic.ttc"],
    "Korean": ["malgun.ttf"],
    "Arabic": ["arial.ttf", "tahoma.ttf"],
}
DEFAULT_FONTS = ["arial.ttf", "calibri.ttf", "segoeui.ttf"]

# -----------------------------------------------------------------------------
# CHROMADB & MODEL CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def get_chroma_collection():
    """Initializes local ChromaDB client and collection."""
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="translation_history")

@st.cache_resource
def load_model():
    """Loads tokenizer and Seq2Seq model into memory with GPU optimizations."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, low_cpu_mem_usage=True
    ).eval()
    
    if torch.cuda.is_available():
        device = "cuda"
        model = model.to(device=device, dtype=torch.float16)
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
            except Exception:
                pass
    else:
        device = "cpu"
        torch.set_num_threads(os.cpu_count() or 4)
        
    return tok, model, device

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def read_file(f):
    """Extracts text while maintaining structural dividers and spacing."""
    data, n = f.getvalue(), f.name.lower()
    if n.endswith(".txt"):
        return data.decode("utf-8-sig", errors="replace")
    if n.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages_text = []
        for i, p in enumerate(reader.pages):
            txt = p.extract_text() or ""
            pages_text.append(f"================================================================================ PAGE {i+1}\n" + txt)
        return "\n\n".join(pages_text)
    
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)

def is_structural_line(line):
    """Detects structural components to bypass translation."""
    s = line.strip()
    if not s or set(s) in [{'-'}, {'='}, {'*'}] or re.match(r"^(=+|-+)\s*(PAGE|PAGINA)\s*\d+.*", s, re.IGNORECASE) or re.match(r"^\d+[\.\)]?$", s):
        return True
    return False

def split_sentences(line, max_chars=512):
    """Splits sentences into length-controlled chunks."""
    out = []
    for s in re.split(r"(?<=[.!?।。！？؟])\s*", line):
        s = s.strip()
        if s:
            out += textwrap.wrap(s, max_chars, break_long_words=False) if len(s) > max_chars else [s]
    return out

def translate_stream(text, src, tgt, batch_size, progress_bar):
    """Batches translation calls for optimal performance while yielding updates."""
    tok, model, device = load_model()
    tok.src_lang = LANGS[src]
    
    bos = tok.lang_code_to_id[LANGS[tgt]] if hasattr(tok, "lang_code_to_id") else tok.convert_tokens_to_ids(LANGS[tgt])
    lines = text.splitlines()
    translated_lines = [None] * len(lines)
    
    batch_units = []
    for idx, line in enumerate(lines):
        if is_structural_line(line):
            translated_lines[idx] = line
        else:
            prefix, content = "", line
            bullet_match = re.match(r"^(\s*[-•*]\s*)(.*)", line)
            if bullet_match:
                prefix, content = bullet_match.group(1), bullet_match.group(2)
            
            sentences = split_sentences(content)
            batch_units.append((idx, prefix, sentences))

    total_units = len(batch_units) if batch_units else 1
    processed_count = 0

    for i in range(0, len(batch_units), batch_size):
        chunk = batch_units[i : i + batch_size]
        flat_sentences, sentence_map = [], []

        for item_idx, (_, _, sentences) in enumerate(chunk):
            for s in sentences:
                flat_sentences.append(s)
                sentence_map.append(item_idx)

        if flat_sentences:
            enc = tok(
                flat_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.inference_mode():
                ids = model.generate(
                    **enc,
                    forced_bos_token_id=bos,
                    num_beams=1,
                    max_new_tokens=512,
                )

            decoded = tok.batch_decode(ids, skip_special_tokens=True)

            reconstructed = [[] for _ in chunk]
            for dec_text, chunk_item_idx in zip(decoded, sentence_map):
                reconstructed[chunk_item_idx].append(dec_text)

            for item_idx, (orig_line_idx, prefix, _) in enumerate(chunk):
                translated_lines[orig_line_idx] = prefix + " ".join(reconstructed[item_idx])

        processed_count += len(chunk)
        progress_bar.progress(min(1.0, processed_count / total_units))
        yield "\n".join([line if line is not None else "" for line in translated_lines])

# -----------------------------------------------------------------------------
# PERSISTENCE & EXPORT FUNCTIONS
# -----------------------------------------------------------------------------
def save_to_chroma(thread_id, source, target, original, translated):
    collection = get_chroma_collection()
    collection.add(
        documents=[translated],
        metadatas=[{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_lang": source,
            "target_lang": target,
            "original_text": original[:1000]
        }],
        ids=[thread_id]
    )

def fetch_from_chroma(thread_id):
    collection = get_chroma_collection()
    res = collection.get(ids=[thread_id])
    if res and res["ids"]:
        return {"translated": res["documents"][0], "metadata": res["metadatas"][0]}
    return None

def fetch_all_chroma_records():
    return get_chroma_collection().get()

def get_registered_font(lang):
    candidate_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ] + [os.path.join(WIN, f) for f in FONTS.get(lang, []) + DEFAULT_FONTS]
    
    for path in candidate_fonts:
        if os.path.exists(path):
            try:
                font_name = f"CustomFont_{uuid.uuid4().hex[:6]}"
                pdfmetrics.registerFont(TTFont(font_name, path))
                return font_name
            except Exception:
                pass
    return "Helvetica"

def make_pdf(text, lang):
    style = ParagraphStyle(
        "ClientTextFormat",
        parent=getSampleStyleSheet()["Normal"],
        fontName=get_registered_font(lang),
        fontSize=9.5,
        leading=13.5,
    )
    
    rtl = lang == "Arabic"
    if rtl:
        try:
            import arabic_reshaper; from bidi.algorithm import get_display
        except ImportError:
            rtl = False

    out = io.BytesIO()
    story = []
    
    for line in text.splitlines():
        if re.match(r"^(=+|-+)\s*(PAGE|PAGINA)\s*\d+.*", line, re.IGNORECASE):
            story.append(Spacer(1, 6))
            story.append(PageBreak())

        if rtl and line.strip():
            line = get_display(arabic_reshaper.reshape(line))
            
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") or " "
        story.append(Paragraph(safe, style))
        story.append(Spacer(1, 2))

    doc = SimpleDocTemplate(out, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)
    return out.getvalue()

def make_docx(text):
    out, doc = io.BytesIO(), Document()
    for line in text.splitlines():
        if re.match(r"^(=+|-+)\s*(PAGE|PAGINA)\s*\d+.*", line, re.IGNORECASE):
            doc.add_page_break()
            doc.add_paragraph(line)
        else:
            doc.add_paragraph(line)
    doc.save(out)
    return out.getvalue()

# -----------------------------------------------------------------------------
# STREAMLIT UI & EVENT HANDLERS
# -----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("current_thread_id", None)
ss.setdefault("current_record", None)

c1, c2 = st.columns(2)
source = c1.selectbox("Input language", list(LANGS), index=0)
target = c2.selectbox("Translate into", list(LANGS), index=1)
uploaded = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])
prompt = st.text_area("Or enter text to translate", height=120)
batch_size = st.select_slider("Batch size (higher = faster processing)", [4, 8, 16, 32, 64], value=DEFAULT_BATCH_SIZE)

if st.button("🚀 Translate", type="primary"):
    text = read_file(uploaded) if uploaded else prompt
    if not text.strip():
        st.error("Upload a file or enter text first.")
    elif source == target:
        st.warning("Choose different input and output languages.")
    else:
        try:
            with st.spinner("Initializing accelerated translation engine..."):
                load_model()

            bar = st.progress(0)
            output_container = st.empty()

            final_text = ""
            for partial_translation in translate_stream(text, source, target, batch_size, bar):
                final_text = partial_translation
                output_container.markdown(
                    f'<div class="stream-box">{final_text}</div>',
                    unsafe_allow_html=True,
                )

            bar.empty()
            
            auto_thread_id = f"TR-{uuid.uuid4().hex[:10].upper()}"
            save_to_chroma(auto_thread_id, source, target, text, final_text)
            
            ss.current_thread_id = auto_thread_id
            ss.current_record = {"translated": final_text, "target": target}
            st.rerun()
        except Exception as e:
            st.error(f"Translation failed: {e}")

if ss.current_thread_id and ss.current_record:
    st.success(f"Translation saved to ChromaDB! System Thread ID: `{ss.current_thread_id}`")
    st.text_area("Final translation", ss.current_record["translated"], height=300)

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ PDF", make_pdf(ss.current_record["translated"], ss.current_record["target"]), "translation.pdf", "application/pdf")
    d2.download_button("⬇️ DOCX", make_docx(ss.current_record["translated"]), "translation.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    d3.download_button("⬇️ TXT", ss.current_record["translated"], "translation.txt", "text/plain")

with st.expander("ChromaDB History / Import by System Thread ID"):
    tid_in = st.text_input("System Thread ID")
    if st.button("Load thread"):
        if tid_in.strip():
            record = fetch_from_chroma(tid_in.strip())
            if record:
                ss.current_thread_id = tid_in.strip()
                ss.current_record = {
                    "translated": record["translated"],
                    "target": record["metadata"]["target_lang"]
                }
                st.rerun()
            else:
                st.warning("Thread ID not found in ChromaDB database.")
                
    st.subheader("Saved Records")
    records = fetch_all_chroma_records()
    if records and records["ids"]:
        for tid, meta in zip(records["ids"], records["metadatas"]):
            st.write(f"`{tid}` — {meta.get('timestamp', 'N/A')} — {meta.get('source_lang', '')} → {meta.get('target_lang', '')}")
    else:
        st.write("No saved translations found in ChromaDB.")