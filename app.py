
import io, uuid, time, html
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="AI Translator", layout="wide")
st.title("🌍 AIT GLOBAL Document Translator")

MODEL = "facebook/nllb-200-distilled-600M"
MAX_NEW_TOKENS = 800
LANGS = {
    "English":"eng_Latn", "Tamil":"tam_Taml", "Hindi":"hin_Deva",
    "Telugu":"tel_Telu", "Malayalam":"mal_Mlym", "Kannada":"kan_Knda",
    "French":"fra_Latn", "Spanish":"spa_Latn", "German":"deu_Latn",
    "Arabic":"arb_Arab", "Chinese":"zho_Hans", "Japanese":"jpn_Jpan",
    "Korean":"kor_Hang", "Portuguese":"por_Latn", "Russian":"rus_Cyrl",
    "Italian":"ita_Latn", "Bengali":"ben_Beng", "Urdu":"urd_Arab"
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to("cpu").eval()
    return tokenizer, model

def make_pdf(text):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(html.escape(line) or "&nbsp;", styles["BodyText"])
        for line in text.splitlines()
    ]
    doc.build(story or [Spacer(1, 10)])
    return output.getvalue()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns(2)
source = col1.selectbox("Source language", list(LANGS))
target = col2.selectbox("Target language", list(LANGS), index=1)

uploaded = st.file_uploader("Upload TXT file", type=["txt"])
prompt = st.text_area("Enter text or translation instructions", height=130)
text = uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else prompt

chunk_size = st.slider("Chunk size (characters)", 500, 3000, 1200, 100)
st.caption(f"Thread: {st.session_state.thread_id} | CPU | Max output: 800 tokens/chunk")

if st.button("New thread"):
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    st.rerun()

if st.button("Translate", type="primary", disabled=not text.strip()):
    try:
        tokenizer, model = load_model()
        tokenizer.src_lang = LANGS[source]
        chunks = []
        remaining = text.strip()
        while remaining:
            if len(remaining) <= chunk_size:
                chunks.append(remaining)
                break
            split_at = remaining.rfind(" ", 0, chunk_size)
            if split_at < chunk_size // 2:
                split_at = chunk_size
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].strip()

        progress = st.progress(0)
        status = st.empty()
        live_output = st.empty()
        results = []
        target_id = tokenizer.convert_tokens_to_ids(LANGS[target])

        for i, chunk in enumerate(chunks, 1):
            status.write(f"Translating chunk {i}/{len(chunks)}...")
            inputs = tokenizer(
                chunk, return_tensors="pt", truncation=True,
                max_length=1024
            )
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    forced_bos_token_id=target_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=2,
                    do_sample=False
                )
            results.append(tokenizer.decode(output[0], skip_special_tokens=True))
            live_output.text_area(
                "Translation output",
                "\n\n".join(results),
                height=220,
                key=f"live_{st.session_state.thread_id}_{i}"
            )
            progress.progress(i / len(chunks))

        translated = "\n\n".join(results)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({
            "time": timestamp, "thread": st.session_state.thread_id,
            "source": source, "target": target,
            "input": text, "output": translated
        })
        status.success(f"Translation complete at {timestamp}")
        st.download_button("Download TXT", translated,
                           file_name="translated.txt", mime="text/plain")
        st.download_button("Download PDF", make_pdf(translated),
                           file_name="translated.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Translation failed: {e}")

with st.expander("Translation history"):
    for item in reversed(st.session_state.history):
        st.write(f"{item['time']} | {item['thread']} | {item['source']} → {item['target']}")
        st.download_button(
            "Download result", item["output"],
            file_name=f"translation_{item['thread']}.txt",
            key=f"history_{item['thread']}_{item['time']}"
        )

