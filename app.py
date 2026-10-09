import io, uuid, time
from datetime import datetime
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

st.set_page_config(page_title="AI Translator", layout="wide")
st.title("🌍 AIT GLOBAL — AI Translator")

MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {
    "English": "eng_Latn", "Tamil": "tam_Taml",
    "Hindi": "hin_Deva", "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym", "Kannada": "kan_Knda",
    "French": "fra_Latn", "German": "deu_Latn",
    "Spanish": "spa_Latn", "Arabic": "arb_Arab",
    "Chinese": "zho_Hans", "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang", "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl", "Italian": "ita_Latn"
}

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.eval()
    return tok, model

if "history" not in st.session_state:
    st.session_state.history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

st.caption(f"Thread ID: {st.session_state.thread_id}")
source = st.selectbox("Source language", list(LANGS), index=0)
target = st.selectbox("Target language", list(LANGS), index=1)
uploaded = st.file_uploader("Upload TXT file (optional)", type=["txt"])
prompt = st.text_area("Enter text to translate", height=150,
                      placeholder="Type or paste your text here...")
text = uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else prompt

if st.button("🔄 New Thread"):
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    st.session_state.history = []
    st.rerun()

if st.button("🌐 Translate", type="primary", disabled=not text.strip()):
    if source == target:
        st.warning("Choose different source and target languages.")
    else:
        try:
            with st.spinner("Loading translation model (first run may take time)..."):
                tokenizer, model = load_model()

            # Split into manageable chunks, preserving paragraph boundaries.
            chunks, current = [], ""
            for paragraph in text.splitlines():
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                while len(paragraph) > 900:
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(paragraph[:900])
                    paragraph = paragraph[900:]
                if len(current) + len(paragraph) + 1 > 900:
                    if current:
                        chunks.append(current)
                    current = paragraph
                else:
                    current = (current + "\n" + paragraph).strip()
            if current:
                chunks.append(current)
            if not chunks:
                chunks = [text.strip()]

            progress = st.progress(0)
            status = st.empty()
            output = st.empty()
            results = []

            for i, chunk in enumerate(chunks):
                status.info(f"Translating chunk {i+1}/{len(chunks)}...")
                tokenizer.src_lang = LANGS[source]
                inputs = tokenizer(chunk, return_tensors="pt",
                                   truncation=True, max_length=512)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                            LANGS[target]),
                        max_new_tokens=256,
                        num_beams=2
                    )
                result = tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )[0]
                results.append(result)
                progress.progress((i + 1) / len(chunks))
                output.markdown("**Latest translated chunk:**\n\n" + result)
                status.write(f"Completed {i+1}/{len(chunks)} chunks")

            translated = "\n\n".join(results)
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "thread": st.session_state.thread_id,
                "source": source, "target": target,
                "input": text, "output": translated
            })
            st.subheader("✅ Complete Translation")
            st.text_area("Translated text", translated, height=250)
            status.success(f"Finished translating {len(chunks)} chunks!")

            # Create downloadable PDF
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = [Paragraph("AI Translator — Translation", styles["Title"]),
                     Spacer(1, 12),
                     Paragraph(f"{escape(source)} to {escape(target)}", styles["Heading2"]),
                     Spacer(1, 12)]
            for paragraph in translated.split("\n"):
                story.append(Paragraph(escape(paragraph) or " ", styles["BodyText"]))
                story.append(Spacer(1, 6))
            doc.build(story)
            st.download_button(
                "📄 Download translated PDF",
                data=buffer.getvalue(),
                file_name="translated_document.pdf",
                mime="application/pdf"
            )
            st.download_button(
                "⬇️ Download translated TXT",
                data=translated.encode("utf-8"),
                file_name="translated.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Translation failed: {e}")

if st.session_state.history:
    st.subheader("🕘 Translation History")
    for item in st.session_state.history:
        with st.expander(f"{item['time']} | {item['source']} → {item['target']}"):
            st.caption(f"Thread ID: {item['thread']}")
            st.write("**Input:**", item["input"][:1000])
            st.write("**Translation:**", item["output"][:2000])

