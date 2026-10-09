
import os, uuid
import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()
try:
    token = st.secrets.get("HF_TOKEN", "")
except Exception:
    token = ""
if token and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = token

MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {
    "English": "eng_Latn", "Spanish": "spa_Latn",
    "Tamil": "tam_Taml", "Hindi": "hin_Deva",
    "Italian": "ita_Latn", "French": "fra_Latn",
    "German": "deu_Latn", "Portuguese": "por_Latn",
    "Arabic": "arb_Arab", "Chinese": "zho_Hans"
}

st.set_page_config(page_title="AI Translator")
st.title("🌍 NLLB Document Translator")

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    net = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    net.eval()
    return tok, net

if "thread" not in st.session_state:
    st.session_state.thread = str(uuid.uuid4())[:8]
st.caption("Thread ID: " + st.session_state.thread)
if st.button("New Thread"):
    st.session_state.thread = str(uuid.uuid4())[:8]
    st.rerun()

file = st.file_uploader("Upload TXT file", type=["txt"])
src = st.selectbox("Translate from", list(LANGS))
dst = st.selectbox("Translate to", list(LANGS), index=1)

if st.button("Translate", type="primary"):
    if file is None:
        st.warning("Upload a TXT file first.")
    elif src == dst:
        st.warning("Select different languages.")
    else:
        try:
            text = file.getvalue().decode("utf-8", errors="replace").strip()
            if not text:
                st.error("The file is empty.")
                st.stop()

            chunks, buf = [], ""
            for word in text.split():
                if buf and len(buf) + len(word) + 1 > 1200:
                    chunks.append(buf)
                    buf = ""
                buf = f"{buf} {word}".strip()
            if buf:
                chunks.append(buf)

            tok, net = load_model()
            tok.src_lang = LANGS[src]
            target_id = tok.convert_tokens_to_ids(LANGS[dst])
            bar = st.progress(0)
            status = st.empty()
            results = []

            for i, chunk in enumerate(chunks):
                status.write(f"Translating {i+1}/{len(chunks)}...")
                inputs = tok(chunk, return_tensors="pt",
                             truncation=True, max_length=512)
                with torch.inference_mode():
                    output = net.generate(
                        **inputs, forced_bos_token_id=target_id,
                        max_new_tokens=256, num_beams=1
                    )
                results.append(tok.decode(output[0], skip_special_tokens=True))
                bar.progress((i + 1) / len(chunks))

            result = "\n\n".join(results)
            status.success("Translation completed!")
            st.text_area("Translated text", result, height=250)
            st.download_button("Download translation", result,
                               "translated.txt", "text/plain")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")