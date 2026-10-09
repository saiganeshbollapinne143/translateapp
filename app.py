
import uuid
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- CONFIG ----------
st.set_page_config(
    page_title="AIT GLOBAL TECHNOLOGIES",
    page_icon="🌍",
    layout="wide"
)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANGS = {
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Telugu": "tel_Telu"
}

def ist_now():
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

@st.cache_resource(show_spinner="Loading NLLB-200 model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if device == "cuda":
        model = model.half()

    model.to(device)
    model.eval()
    return tokenizer, model, device

def split_text(text, limit=250):
    # Split on sentence boundaries, then split oversized sentences safely.
    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    chunks = []
    current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        while len(piece) > limit:
            if current:
                chunks.append(current)
                current = ""

            cut = piece.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit

            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()

        candidate = f"{current} {piece}".strip()

        if current and len(candidate) > limit:
            chunks.append(current)
            current = piece
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks

# ---------- SESSION STATE ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = ""

if "last_filename" not in st.session_state:
    st.session_state.last_filename = "translated.txt"

def new_thread():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.last_result = ""
    st.session_state.last_filename = "translated.txt"

# ---------- UI ----------
st.title("🌍 AIT GLOBAL TECHNOLOGIES")
st.caption("NLLB-200 Translator | Live chunk-by-chunk translation")

left, right = st.columns([3, 1])

with left:
    st.write("**Thread ID**")
    st.code(st.session_state.thread_id)

with right:
    st.button(
        "➕ New Thread",
        on_click=new_thread,
        use_container_width=True
    )

left, right = st.columns(2)

with left:
    source_name = st.selectbox(
        "Source language",
        list(LANGS.keys()),
        index=0
    )

with right:
    target_name = st.selectbox(
        "Target language",
        list(LANGS.keys()),
        index=1
    )

uploaded = st.file_uploader(
    "Upload a TXT document",
    type=["txt"]
)

manual_text = st.text_area(
    "Enter or paste text to translate",
    height=180,
    placeholder="Paste your document text here..."
)

text_input = manual_text
filename = "pasted_text.txt"

if uploaded is not None:
    try:
        text_input = uploaded.getvalue().decode("utf-8-sig")
        filename = uploaded.name
    except UnicodeDecodeError:
        text_input = ""
        st.error("Please save the TXT file as UTF-8 and upload it again.")

# ---------- TRANSLATE ----------
if st.button("🚀 Translate", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Upload a TXT file or paste text first.")

    elif source_name == target_name:
        st.warning("Choose different source and target languages.")

    else:
        progress = st.progress(0)
        status = st.empty()
        live_area = st.empty()

        try:
            status.info("Loading translation model...")
            tokenizer, model, device = load_model()

            chunks = split_text(text_input, limit=250)

            if not chunks:
                st.warning("No text found to translate.")
                st.stop()

            total = len(chunks)
            translated_parts = []
            started = time.time()

            status.info(
                f"Model ready on {device.upper()}. "
                f"Translating {total} chunks..."
            )

            for i, chunk in enumerate(chunks):
                status.info(
                    f"Translating chunk {i + 1}/{total}..."
                )

                tokenizer.src_lang = LANGS[source_name]

                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)

                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                            LANGS[target_name]
                        ),
                        max_new_tokens=256,
                        num_beams=1,
                        do_sample=False
                    )

                translated_chunk = tokenizer.decode(
                    output[0],
                    skip_special_tokens=True
                )

                translated_parts.append(translated_chunk)
                current_result = "\n\n".join(translated_parts)

                # Display completed chunks immediately.
                with live_area.container():
                    st.markdown("### 🔄 Streaming Translation")
                    st.text_area(
                        "Translated text so far",
                        value=current_result,
                        height=280,
                        key=f"live_{st.session_state.thread_id}_{i}"
                    )

                # Progress updates only after each chunk is generated.
                progress.progress((i + 1) / total)

            result = "\n\n".join(translated_parts)
            duration = round(time.time() - started, 2)

            st.session_state.last_result = result
            st.session_state.last_filename = (
                f"translated_{LANGS[target_name]}.txt"
            )

            st.session_state.history.insert(0, {
                "thread": st.session_state.thread_id,
                "file": filename,
                "source": source_name,
                "target": target_name,
                "chunks": total,
                "seconds": duration,
                "time": ist_now()
            })

            status.success(
                f"✅ Translation completed in {duration} seconds."
            )

        except Exception as e:
            status.error("Translation failed.")
            st.exception(e)

# ---------- FINAL OUTPUT ----------
if st.session_state.last_result:
    st.subheader("✅ Completed Translation")

    st.text_area(
        "Final translated document",
        value=st.session_state.last_result,
        height=300,
        key="final_translation"
    )

    st.download_button(
        "📥 Download translated TXT",
        data=st.session_state.last_result.encode("utf-8"),
        file_name=st.session_state.last_filename,
        mime="text/plain",
        use_container_width=True
    )

# ---------- HISTORY ----------
with st.expander("🕘 Translation History"):
    if st.session_state.history:
        for item in st.session_state.history:
            st.markdown(
                f"**{item['source']} → {item['target']}** | "
                f"{item['file']} | {item['seconds']} seconds"
            )
            st.caption(
                f"Thread: {item['thread']} | "
                f"Chunks: {item['chunks']} | "
                f"Completed: {item['time']}"
            )
    else:
        st.caption("No translations yet.")