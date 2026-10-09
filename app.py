
import uuid
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES", layout="wide")
st.title("AIT GLOBAL TECHNOLOGIES")


MODEL = "facebook/nllb-200-distilled-600M"

# Priority languages only
LANGS = {
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
    "English": "eng_Latn",
    "Kannada": "kan_Knda"
}

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.to(device).eval()
    return tokenizer, model, device

def now_ist():
    return datetime.now(ZoneInfo("Asia/Kolkata"))

# Session initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "result" not in st.session_state:
    st.session_state.result = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "notice" not in st.session_state:
    st.session_state.notice = ""

def reset_app():
    st.session_state.prompt = ""
    st.session_state.result = ""
    st.session_state.notice = "Prompt and translation reset."

def new_thread():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.prompt = ""
    st.session_state.result = ""
    st.session_state.notice = "New thread created."

now = now_ist()
st.caption(f"Date: {now:%d-%m-%Y} | Time: {now:%I:%M:%S %p} IST")
st.caption(f"Thread ID: {st.session_state.thread_id}")

b1, b2 = st.columns(2)
with b1:
    st.button("🔄 Reset", on_click=reset_app,
              use_container_width=True)
with b2:
    st.button("🆕 New Thread", on_click=new_thread,
              use_container_width=True)

if st.session_state.notice:
    st.info(st.session_state.notice)
    st.session_state.notice = ""

st.divider()

c1, c2 = st.columns(2)
with c1:
    src = st.selectbox("Source Language", list(LANGS), index=6)
with c2:
    dst = st.selectbox("Target Language", list(LANGS), index=0)

st.text_area(
    "✍️ Translation Prompt",
    key="prompt",
    height=100,
    placeholder=(
        "Example: Use formal business language. "
        "Preserve names, numbers, and technical terms."
    ),
    help="NLLB-200 does not reliably follow custom instructions. "
         "This field records your preferences but does not directly "
         "control the model."
)

uploaded_file = st.file_uploader(
    "📄 Upload TXT Document",
    type=["txt"]
)

if st.button("🚀 Translate Document", type="primary",
             use_container_width=True):
    if uploaded_file is None:
        st.warning("Please upload a TXT document.")
    elif src == dst:
        st.warning("Please select different source and target languages.")
    else:
        text = uploaded_file.getvalue().decode(
            "utf-8-sig", errors="replace"
        ).strip()

        if not text:
            st.error("The uploaded document is empty.")
        else:
            try:
                tokenizer, model, device = load_model()
                tokenizer.src_lang = LANGS[src]
                target_id = tokenizer.convert_tokens_to_ids(LANGS[dst])

                sentences = re.split(r"(?<=[.!?。！？])\s+", text)
                chunks = []
                buffer = ""

                for sentence in sentences:
                    if len(buffer) + len(sentence) > 700 and buffer:
                        chunks.append(buffer)
                        buffer = ""
                    buffer = (buffer + " " + sentence).strip()

                if buffer:
                    chunks.append(buffer)

                translated_parts = []
                progress = st.progress(0)
                status = st.empty()
                live = st.empty()

                for i, chunk in enumerate(chunks):
                    status.info(
                        f"Translating part {i + 1}/{len(chunks)} "
                        f"on {device.upper()}..."
                    )

                    inputs = tokenizer(
                        chunk,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(device)

                    with torch.inference_mode():
                        output = model.generate(
                            **inputs,
                            forced_bos_token_id=target_id,
                            max_new_tokens=384,
                            num_beams=2
                        )

                    translated = tokenizer.decode(
                        output[0], skip_special_tokens=True
                    )
                    translated_parts.append(translated)
                    st.session_state.result = "\n\n".join(
                        translated_parts
                    )

                    live.text_area(
                        "📄 Live Translation",
                        value=st.session_state.result,
                        height=220,
                        key=f"live_{st.session_state.thread_id}_{i}"
                    )
                    progress.progress((i + 1) / len(chunks))

                completed = now_ist()
                st.session_state.history.append({
                    "Thread ID": st.session_state.thread_id,
                    "File": uploaded_file.name,
                    "Source": src,
                    "Target": dst,
                    "Parts": len(chunks),
                    "Device": device,
                    "Prompt": st.session_state.prompt,
                    "Completed": completed.strftime(
                        "%d-%m-%Y %I:%M:%S %p IST"
                    )
                })

                # Clear prompt after successful translation
                st.session_state.prompt = ""
                status.success("✅ Translation completed!")

            except Exception as error:
                st.error(f"{type(error).__name__}: {error}")

if st.session_state.result:
    st.divider()
    st.subheader("📄 Final Translation")
    st.text_area(
        "Translated Text",
        value=st.session_state.result,
        height=300
    )
    st.download_button(
        "📥 Download Translation",
        data=st.session_state.result,
        file_name=f"translation_{st.session_state.thread_id[:8]}.txt",
        mime="text/plain",
        use_container_width=True
    )

with st.expander("🕘 Translation History"):
    if st.session_state.history:
        st.dataframe(
            st.session_state.history,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No translations completed in this session.")
