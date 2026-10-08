import os
import uuid
import streamlit as st
import torch
from io import BytesIO
from pypdf import PdfReader
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Translator", page_icon="🌐")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn"
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- SESSION ----------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    if file.name.lower().endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if file.name.lower().endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    return ""

# ---------------- CHUNK TEXT ----------------
def split_text(text, size=500):
    words = text.split()
    chunks = []
    current = ""

    for word in words:
        if len(current) + len(word) + 1 <= size:
            current += (" " if current else "") + word
        else:
            if current:
                chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks

# ---------------- TRANSLATE ----------------
def translate_text(text, target_language):
    chunks = split_text(text)

    results = []

    progress = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(chunks):
        status.write(f"Translating {i + 1} / {len(chunks)}...")

        tokenizer.src_lang = "eng_Latn"

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(DEVICE)

        target_id = tokenizer.convert_tokens_to_ids(target_language)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=1
            )

        translated = tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )[0]

        results.append(translated)
        progress.progress((i + 1) / len(chunks))

    status.empty()
    progress.empty()

    return "\n\n".join(results)

# ---------------- CREATE PDF ----------------
def create_pdf(text):
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    margin = 40
    y = height - margin

    pdf.setFont("Helvetica", 10)

    for paragraph in text.split("\n"):
        words = paragraph.split()
        line = ""

        for word in words:
            test_line = line + " " + word if line else word

            if pdf.stringWidth(test_line, "Helvetica", 10) < width - 2 * margin:
                line = test_line
            else:
                pdf.drawString(margin, y, line)
                y -= 15
                line = word

                if y < margin:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 10)
                    y = height - margin

        if line:
            pdf.drawString(margin, y, line)
            y -= 15

        y -= 5

        if y < margin:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - margin

    pdf.save()
    buffer.seek(0)

    return buffer

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌐 AI Translator")

st.sidebar.write("Thread ID:")
st.sidebar.code(st.session_state.thread_id)

if st.sidebar.button("🆕 New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

# ---------------- MAIN UI ----------------
st.title("🌐 AI Document Translator")

st.write(
    "Translate English text or documents into multiple languages "
    "using NLLB-200."
)

source_language = st.selectbox(
    "Source Language",
    list(LANGUAGES.keys()),
    index=0
)

target_language = st.selectbox(
    "Target Language",
    list(LANGUAGES.keys()),
    index=1
)

input_text = st.text_area(
    "Enter text",
    height=180,
    placeholder="Enter English text here..."
)

uploaded_file = st.file_uploader(
    "Or upload TXT, PDF, or DOCX",
    type=["txt", "pdf", "docx"]
)

# ---------------- TRANSLATE BUTTON ----------------
if st.button("🔄 Translate", type="primary"):

    if uploaded_file:
        text = extract_text(uploaded_file)
    else:
        text = input_text

    if not text.strip():
        st.warning("Please enter text or upload a document.")
        st.stop()

    target_code = LANGUAGES[target_language]

    st.info(f"Translating to {target_language}...")

    translated_text = translate_text(
        text,
        target_code
    )

    st.session_state.messages.append({
        "source": text,
        "translation": translated_text,
        "target": target_language
    })

    st.success("Translation completed!")

    st.subheader("Translated Text")

    st.text_area(
        "Result",
        translated_text,
        height=300
    )

    # ---------------- DOWNLOAD TXT ----------------
    st.download_button(
        "📄 Download TXT",
        translated_text,
        file_name=f"translated_{target_language}.txt",
        mime="text/plain"
    )

    # ---------------- DOWNLOAD PDF ----------------
    pdf_file = create_pdf(translated_text)

    st.download_button(
        "📕 Download Translated PDF",
        pdf_file,
        file_name=f"translated_{target_language}.pdf",
        mime="application/pdf"
    )

# ---------------- HISTORY ----------------
if st.session_state.messages:

    st.sidebar.subheader("📜 Current Conversation")

    for i, msg in enumerate(
        reversed(st.session_state.messages),
        1
    ):
        st.sidebar.write(
            f"{i}. {msg['target']} translation"
        )