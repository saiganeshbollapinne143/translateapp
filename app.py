
import re
import uuid
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AIT GLOBAL TECHNOLOGIES Translator",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AIT GLOBAL TECHNOLOGIES")
st.subheader("Quality-Focused AI Document Translator")
st.caption("NLLB-200 | CPU Optimized | Sentence-Aware Chunk Translation")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
CHUNK_SIZE = 700
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 384

LANGS = {
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Telugu": "tel_Telu",
}

# ---------------- SESSION STATE ----------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

if "last_translation" not in st.session_state:
    st.session_state.last_translation = ""

# ---------------- MODEL LOADING ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True
    )
    model.to("cpu")
    model.eval()

    return tokenizer, model

# ---------------- SENTENCE-AWARE CHUNKING ----------------
def split_text(text, tokenizer, limit=CHUNK_SIZE):
    """
    Keep sentences together where possible.
    Use tokenizer length to avoid making chunks too long.
    """
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = (current + " " + sentence).strip() if current else sentence
        token_count = len(
            tokenizer(candidate, add_special_tokens=True)["input_ids"]
        )

        if token_count <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        # Split oversized sentences at word boundaries.
        words = sentence.split()
        piece = ""

        for word in words:
            candidate = (piece + " " + word).strip() if piece else word
            count = len(
                tokenizer(candidate, add_special_tokens=True)["input_ids"]
            )

            if count <= limit:
                piece = candidate
            else:
                if piece:
                    chunks.append(piece)
                piece = word

        if piece:
            current = piece

    if current:
        chunks.append(current)

    return chunks

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Conversation")
    st.caption("Current thread ID")
    st.code(st.session_state.thread_id)

    if st.button("➕ New Thread", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.last_translation = ""
        st.rerun()

    st.divider()
    st.write("**Device:** CPU")
    st.write("**Model:** NLLB-200 distilled 600M")
    st.write("**Priority:** Translation quality")

# ---------------- INPUT UI ----------------
col1, col2 = st.columns(2)

with col1:
    source_name = st.selectbox(
        "Source language",
        list(LANGS.keys()),
        index=0
    )

with col2:
    target_name = st.selectbox(
        "Target language",
        list(LANGS.keys()),
        index=1
    )

uploaded_file = st.file_uploader(
    "Upload a text document (.txt)",
    type=["txt"]
)

uploaded_text = ""
if uploaded_file is not None:
    uploaded_text = uploaded_file.getvalue().decode(
        "utf-8", errors="replace"
    )

source_text = st.text_area(
    "Paste or edit your document",
    value=uploaded_text,
    height=240,
    placeholder="Paste the complete document here..."
)

translate_clicked = st.button(
    "🚀 Translate Document",
    type="primary",
    use_container_width=True
)

# ---------------- TRANSLATION ----------------
if translate_clicked:
    if not source_text.strip():
        st.warning("Please paste text or upload a TXT file.")
        st.stop()

    if source_name == target_name:
        st.warning("Select two different languages.")
        st.stop()

    progress = st.progress(0, text="Starting translation...")
    status = st.empty()
    live_output = st.empty()

    try:
        status.info(
            "Loading the translation model. "
            "The first run may take time to download model files."
        )

        tokenizer, model = load_model()

        tokenizer.src_lang = LANGS[source_name]
        target_token_id = tokenizer.convert_tokens_to_ids(
            LANGS[target_name]
        )

        chunks = split_text(source_text, tokenizer)
        if not chunks:
            st.warning("No translatable text was found.")
            st.stop()

        translated_chunks = []
        total = len(chunks)
        start_time = time.time()

        status.success(
            f"Model ready on CPU. {total} chunk(s) to translate."
        )

        for index, chunk in enumerate(chunks, start=1):
            status.info(
                f"Translating chunk {index}/{total}. "
                "Long chunks can take time on CPU."
            )

            # Do not truncate: chunking has already controlled input length.
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS
            )

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=target_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                    early_stopping=False
                )

            translated = tokenizer.decode(
                generated[0],
                skip_special_tokens=True
            ).strip()

            translated_chunks.append(translated)

            current_output = "\n\n".join(translated_chunks)
            live_output.text_area(
                "Translation output (updates by chunk)",
                value=current_output,
                height=300,
                key=f"live_{st.session_state.thread_id}_{index}"
            )

            percent = int(index * 100 / total)
            elapsed = int(time.time() - start_time)
            progress.progress(
                index / total,
                text=f"{percent}% complete — {index}/{total} chunks — {elapsed}s"
            )

        final_text = "\n\n".join(translated_chunks)
        timestamp = datetime.now(
            ZoneInfo("Asia/Kolkata")
        ).strftime("%d-%m-%Y %I:%M:%S %p IST")

        st.session_state.last_translation = final_text

        st.session_state.history.append({
            "thread": st.session_state.thread_id,
            "time": timestamp,
            "source": source_name,
            "target": target_name,
            "original": source_text,
            "translated": final_text,
            "elapsed": round(time.time() - start_time, 2)
        })

        status.success(
            f"Translation completed in "
            f"{time.time() - start_time:.1f} seconds."
        )

    except Exception as error:
        status.error("Translation failed.")
        st.exception(error)

# ---------------- DOWNLOAD CURRENT RESULT ----------------
if st.session_state.last_translation:
    st.download_button(
        "📥 Download translated document",
        data=st.session_state.last_translation,
        file_name=f"translation_{target_name.lower()}.txt",
        mime="text/plain",
        use_container_width=True
    )

# ---------------- TRANSLATION HISTORY ----------------
st.divider()
st.subheader("🕘 Translation History")

if not st.session_state.history:
    st.caption("Completed translations will appear here.")
else:
    for index, item in enumerate(reversed(st.session_state.history)):
        heading = (
            f"{item['source']} → {item['target']} | "
            f"{item['time']}"
        )

        with st.expander(heading):
            st.caption(f"Thread ID: {item['thread']}")
            st.caption(f"Translation time: {item['elapsed']} seconds")

            st.write("**Original document**")
            st.text(item["original"])

            st.write("**Translated document**")
            st.text(item["translated"])

            st.download_button(
                "📥 Download this translation",
                data=item["translated"],
                file_name=f"translation_history_{index + 1}.txt",
                mime="text/plain",
                key=f"history_{index}_{item['thread']}"
            )
