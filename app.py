
import io
import time
from datetime import datetime

import streamlit as st
import torch
import pandas as pd
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============== AIT GLOBAL TECHNOLOGIES ==============
st.set_page_config(
    page_title="AIT GLOBAL TECHNOLOGIES | AI Translator",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color:#f7f9fc;}
.brand {background:#102b50;color:white;padding:24px;
        border-radius:12px;margin-bottom:18px;}
.brand h1 {color:white;margin:0;}
.brand p {color:#dce8ff;margin:6px 0 0 0;}
.stButton>button {background:#1457a6;color:white;
                  border-radius:8px;font-weight:bold;}
</style>
<div class="brand">
<h1>AIT GLOBAL TECHNOLOGIES</h1>
<p>AI TRANSLATOR | Multilingual Document Translation Platform</p>
</div>
""", unsafe_allow_html=True)

st.caption("Translate documents and text using the NLLB-200 multilingual model.")

MODEL = "facebook/nllb-200-distilled-600M"

LANGS = {
    "English": "eng_Latn", "Spanish": "spa_Latn",
    "Tamil": "tam_Taml", "Hindi": "hin_Deva",
    "Italian": "ita_Latn", "French": "fra_Latn",
    "German": "deu_Latn", "Portuguese": "por_Latn",
    "Dutch": "nld_Latn", "Russian": "rus_Cyrl",
    "Arabic": "arb_Arab", "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan", "Korean": "kor_Hang",
    "Bengali": "ben_Beng", "Telugu": "tel_Telu",
    "Kannada": "kan_Knda", "Malayalam": "mal_Mlym",
    "Urdu": "urd_Arab", "Thai": "tha_Thai",
    "Vietnamese": "vie_Latn", "Indonesian": "ind_Latn",
    "Turkish": "tur_Latn", "Polish": "pol_Latn",
    "Ukrainian": "ukr_Cyrl", "Swahili": "swh_Latn",
    "Greek": "ell_Grek", "Hebrew": "heb_Hebr",
    "Persian": "pes_Arab", "Romanian": "ron_Latn"
}

torch.set_num_threads(min(4, torch.get_num_threads() or 2))

@st.cache_resource(show_spinner="Loading AI translation model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, use_fast=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.to("cpu")
    model.eval()
    return tokenizer, model

def extract_file(upload):
    ext = upload.name.lower().rsplit(".", 1)[-1]
    raw = upload.getvalue()

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    if ext == "docx":
        doc = Document(io.BytesIO(raw))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    if ext in ("txt", "md"):
        return raw.decode("utf-8-sig", errors="replace")

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(raw), dtype=str).fillna("")
        return df.to_csv(index=False)

    if ext == "xlsx":
        sheets = pd.read_excel(
            io.BytesIO(raw), sheet_name=None, dtype=str
        )
        return "\n\n".join(
            "Sheet: " + name + "\n" + frame.fillna("").to_csv(index=False)
            for name, frame in sheets.items()
        )

    if ext == "pptx":
        prs = Presentation(io.BytesIO(raw))
        parts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text.strip():
                    parts.append(shape.text)
                if shape.has_table:
                    for row in shape.table.rows:
                        parts.append(" | ".join(
                            cell.text for cell in row.cells
                        ))
        return "\n\n".join(parts)

    raise ValueError("Unsupported file format.")

def split_chunks(text, size):
    chunks, current = [], ""
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > size:
            cut = paragraph.rfind(" ", 0, size)
            if cut < size // 2:
                cut = size
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            current = paragraph
        else:
            current = (current + "\n" + paragraph).strip()
    if current:
        chunks.append(current)
    return chunks or [text.strip()]

def translate_chunk(text, tokenizer, model, src, tgt, beams, max_tokens):
    tokenizer.src_lang = src
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512
    )
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt),
            num_beams=beams,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True
        )
    return tokenizer.batch_decode(
        output, skip_special_tokens=True
    )[0].strip()

def make_pdf(text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch
    )
    styles = getSampleStyleSheet()
    story = []
    for line in text.splitlines():
        safe = (line.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
        story.append(Paragraph(safe or " ", styles["Normal"]))
        story.append(Spacer(1, 5))
    doc.build(story)
    return buffer.getvalue()

def make_docx(text):
    buffer = io.BytesIO()
    doc = Document()
    doc.add_heading("AIT GLOBAL TECHNOLOGIES - AI TRANSLATOR", 1)
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(buffer)
    return buffer.getvalue()

def make_pptx(text):
    buffer = io.BytesIO()
    prs = Presentation()
    lines = text.splitlines() or [text]
    for start in range(0, len(lines), 8):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Translated Document"
        frame = slide.placeholders[1].text_frame
        frame.text = "\n".join(lines[start:start + 8])
    prs.save(buffer)
    return buffer.getvalue()

def make_xlsx(text):
    buffer = io.BytesIO()
    pd.DataFrame({"Translated Text": text.splitlines()}).to_excel(
        buffer, index=False
    )
    return buffer.getvalue()

def make_csv(text):
    return pd.DataFrame(
        {"Translated Text": text.splitlines()}
    ).to_csv(index=False).encode("utf-8-sig")

# ============== TRANSLATION INPUT ==============
left, right = st.columns(2)
with left:
    source_name = st.selectbox(
        "Source language", list(LANGS), index=0
    )
with right:
    target_name = st.selectbox(
        "Target language", list(LANGS), index=1
    )

uploaded = st.file_uploader(
    "Upload document",
    type=["pdf", "docx", "txt", "md", "csv", "xlsx", "pptx"]
)

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if uploaded:
    try:
        st.session_state.input_text = extract_file(uploaded)
        st.success(f"Extracted text from {uploaded.name}")
        if not st.session_state.input_text.strip():
            st.warning(
                "No selectable text found. Scanned PDFs need OCR first."
            )
    except Exception as error:
        st.error(f"Could not read the file: {error}")

prompt = st.text_input(
    "Translation instructions",
    value="Translate accurately. Preserve meaning, names, numbers and tone.",
    help="NLLB is a translation model; arbitrary instructions may not be followed."
)

input_text = st.text_area(
    "Input text",
    value=st.session_state.input_text,
    height=220,
    placeholder="Paste text here or upload a supported document."
)
st.session_state.input_text = input_text

with st.expander("⚙️ Speed and quality"):
    chunk_size = st.select_slider(
        "Chunk size (characters)",
        options=[400, 600, 800, 1000, 1200, 1600, 2000],
        value=1200
    )
    beams = st.select_slider(
        "Beam search",
        options=[1, 2, 3, 4],
        value=2,
        help="1 is fastest; 4 may improve quality but takes longer."
    )
    max_tokens = st.select_slider(
        "Maximum output tokens per chunk",
        options=[128, 192, 256, 384, 512],
        value=256
    )

if st.button("🚀 TRANSLATE DOCUMENT", type="primary",
             use_container_width=True):
    if not input_text.strip():
        st.warning("Upload a document or enter text first.")
    elif source_name == target_name:
        st.warning("Select different source and target languages.")
    else:
        try:
            tokenizer, model = load_model()
            chunks = split_chunks(input_text, int(chunk_size))
            results = []
            progress = st.progress(0, text="Starting translation...")
            status = st.empty()
            live = st.empty()
            started = time.time()

            for i, chunk in enumerate(chunks, 1):
                status.info(f"Translating chunk {i} of {len(chunks)}...")
                result = translate_chunk(
                    chunk, tokenizer, model,
                    LANGS[source_name], LANGS[target_name],
                    int(beams), int(max_tokens)
                )
                results.append(result)
                live.text_area(
                    "Live translated output",
                    "\n\n".join(results),
                    height=280,
                    key=f"live_output_{i}_{len(chunks)}"
                )
                progress.progress(
                    i / len(chunks),
                    text=f"Completed {i}/{len(chunks)} chunks"
                )

            final_text = "\n\n".join(results)
            elapsed = time.time() - started
            st.session_state["final_text"] = final_text
            st.success(
                f"Translation complete in {elapsed:.1f} seconds."
            )
        except Exception as error:
            st.error(f"Translation failed: {error}")

# ============== DOWNLOAD TRANSLATED FILES ==============
if st.session_state.get("final_text"):
    final_text = st.session_state["final_text"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.subheader("📥 Download translated document")
    st.caption("Downloads contain translated text; original layouts may not be preserved.")

    a, b, c = st.columns(3)
    with a:
        st.download_button(
            "Download TXT",
            final_text.encode("utf-8"),
            f"translated_{stamp}.txt",
            "text/plain",
            use_container_width=True
        )
        st.download_button(
            "Download PDF",
            make_pdf(final_text),
            f"translated_{stamp}.pdf",
            "application/pdf",
            use_container_width=True
        )
    with b:
        st.download_button(
            "Download DOCX",
            make_docx(final_text),
            f"translated_{stamp}.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        st.download_button(
            "Download XLSX",
            make_xlsx(final_text),
            f"translated_{stamp}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with c:
        st.download_button(
            "Download CSV",
            make_csv(final_text),
            f"translated_{stamp}.csv",
            "text/csv",
            use_container_width=True
        )
        st.download_button(
            "Download PPTX",
            make_pptx(final_text),
            f"translated_{stamp}.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True
        )
        st.download_button(
            "Download Markdown",
            final_text.encode("utf-8"),
            f"translated_{stamp}.md",
            "text/markdown",
            use_container_width=True
        )

st.divider()
st.caption(
    "AIT GLOBAL TECHNOLOGIES | AI Translator | "
    "CPU processing with NLLB-200"
)

