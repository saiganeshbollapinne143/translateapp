
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="AI Translator", layout="wide")
st.title("🌍 NLLB-200 Document Translator")

MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {
    "English": "eng_Latn", "Spanish": "spa_Latn",
    "Tamil": "tam_Taml", "Hindi": "hin_Deva",
    "Italian": "ita_Latn", "French": "fra_Latn",
    "German": "deu_Latn", "Portuguese": "por_Latn",
    "Arabic": "arb_Arab", "Chinese": "zho_Hans"
}

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    net = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    net.eval()
    return tok, net, device

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

now = datetime.now(ZoneInfo("Asia/Kolkata"))
st.caption(f"📅 Date: {now:%d-%m-%Y} | 🕒 Time: {now:%I:%M:%S %p} IST")
st.caption(f"🧵 Thread ID: {st.session_state.thread_id}")

if st.button("🆕 New Thread"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.translated_text = ""
    st.session_state.history = []
    st.rerun()

prompt = st.text_area(
    "✍️ Translation prompt",
    "Translate accurately into the selected target language. "
    "Preserve meaning, names, numbers, headings and paragraphs.",
    height=100
)

c1, c2 = st.columns(2)
with c1:
    src = st.selectbox("Source language", list(LANGS))
with c2:
    dst = st.selectbox("Target language", list(LANGS), index=1)

file = st.file_uploader("📄 Upload TXT document", type=["txt"])

st.subheader("📄 Live Translation Output")
output = st.empty()
output.text_area("Translated text", value=st.session_state.translated_text,
                 height=500, key="initial_output")

if st.button("🚀 Translate Document", type="primary"):
    if file is None:
        st.warning("Upload a TXT file first.")
    elif src == dst:
        st.warning("Select different languages.")
    elif not prompt.strip():
        st.warning("Enter a translation prompt.")
    else:
        text = file.getvalue().decode("utf-8", errors="replace").strip()
        if not text:
            st.error("The file is empty.")
        else:
            try:
                tok, net, device = load_model()
                tok.src_lang = LANGS[src]
                target_id = tok.convert_tokens_to_ids(LANGS[dst])

                words, chunks, buf = text.split(), [], []
                size = 0
                for word in words:
                    if size + len(word) + 1 > 850 and buf:
                        chunks.append(" ".join(buf))
                        buf, size = [], 0
                    buf.append(word)
                    size += len(word) + 1
                if buf:
                    chunks.append(" ".join(buf))

                results = []
                progress = st.progress(0)
                status = st.empty()
                live = st.empty()

                for i, chunk in enumerate(chunks):
                    status.write(f"Translating {i+1}/{len(chunks)}...")
                    inputs = tok(chunk, return_tensors="pt",
                                 truncation=True, max_length=512).to(device)

                    with torch.inference_mode():
                        generated = net.generate(
                            **inputs,
                            forced_bos_token_id=target_id,
                            max_new_tokens=256,
                            num_beams=1
                        )

                    results.append(tok.decode(
                        generated[0], skip_special_tokens=True))
                    st.session_state.translated_text = "\n\n".join(results)
                    live.text_area("Translation in progress",
                                   value=st.session_state.translated_text,
                                   height=500, key=f"live_{i}")
                    progress.progress((i + 1) / len(chunks))

                completed = datetime.now(ZoneInfo("Asia/Kolkata"))
                st.session_state.history.append({
                    "thread_id": st.session_state.thread_id,
                    "source": src, "target": dst,
                    "date_time": completed.isoformat(),
                    "file": file.name
                })
                status.success(f"Completed at {completed:%d-%m-%Y %I:%M:%S %p} IST")

            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

if st.session_state.translated_text:
    st.download_button(
        "📥 Download Translation",
        data=st.session_state.translated_text,
        file_name=f"translation_{st.session_state.thread_id[:8]}.txt",
        mime="text/plain"
    )

with st.expander("🕘 Translation History"):
    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No completed translations in this session yet.")

