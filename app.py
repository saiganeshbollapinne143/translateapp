
import io, re, uuid
from datetime import datetime
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Translator", layout="wide")
st.title("🌍 AIT GLOBAL TECHNOLOGIES - TRANSLATOR")

@st.cache_resource
def load_model():
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()
    return tok, model

LANGS = {
    "English": "eng_Latn", "Tamil": "tam_Taml",
    "Hindi": "hin_Deva", "Telugu": "tel_Telu",
    "French": "fra_Latn", "Spanish": "spa_Latn",
    "German": "deu_Latn", "Chinese": "zho_Hans",
    "Arabic": "arb_Arab", "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang", "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl", "Italian": "ita_Latn"
}

source = st.selectbox("Input language", list(LANGS), index=0)
target = st.selectbox("Translate into", list(LANGS), index=1)
uploaded = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])
prompt = st.text_area("Or enter text to translate", height=150)
max_tokens = st.selectbox("Maximum output tokens per chunk", [256, 512, 800], index=0)
chunk_size = st.selectbox("Input chunk size (characters)", [500, 1000, 1500], index=1)

def read_file(file):
    data = file.getvalue()
    if file.name.lower().endswith(".txt"):
        return data.decode("utf-8-sig", errors="replace")
    if file.name.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)

def make_pdf(text):
    out = io.BytesIO()
    doc = SimpleDocTemplate(out)
    styles = getSampleStyleSheet()
    story = []
    for line in text.splitlines():
        story += [Paragraph(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") or " ", styles["Normal"]), Spacer(1, 5)]
    doc.build(story)
    return out.getvalue()

def make_docx(text):
    out = io.BytesIO()
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(out)
    return out.getvalue()

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🚀 Translate", type="primary"):
    text = read_file(uploaded) if uploaded else prompt
    if not text.strip():
        st.error("Upload a file or enter text first.")
    elif source == target:
        st.warning("Choose different input and output languages.")
    else:
        try:
            with st.spinner("Loading NLLB-200 model (first run may take time)..."):
                tok, model = load_model()
            tok.src_lang = LANGS[source]
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            result, progress = [], st.progress(0)
            output_area = st.empty()
            status = st.empty()
            for i, chunk in enumerate(chunks):
                inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=256)
                with torch.no_grad():
                    ids = model.generate(
                        **inputs,
                        forced_bos_token_id=tok.convert_tokens_to_ids(LANGS[target]),
                        max_new_tokens=max_tokens,
                        num_beams=2
                    )
                result.append(tok.batch_decode(ids, skip_special_tokens=True)[0])
                output_area.text_area("Translation in progress", "\n\n".join(result), height=250)
                progress.progress((i + 1) / len(chunks))
                status.write(f"Translated chunk {i + 1} of {len(chunks)}")
            translated = "\n\n".join(result)
            st.success("Translation completed!")
            st.text_area("Final translation", translated, height=300)
            st.download_button("Download PDF", make_pdf(translated), "translation.pdf", "application/pdf")
            st.download_button("Download DOCX", make_docx(translated), "translation.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            st.session_state.history.append({
                "thread_id": str(uuid.uuid4()), "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": source, "target": target, "chunks": len(chunks)
            })
        except Exception as e:
            st.error(f"Translation failed: {e}")

with st.expander("Translation history"):
    st.write(st.session_state.history or "No translations yet.")

