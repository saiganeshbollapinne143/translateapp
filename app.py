
import uuid
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Translator", page_icon="🌍", layout="wide")
MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANGS = {
    "English": "eng_Latn", "Spanish": "spa_Latn",
    "French": "fra_Latn", "German": "deu_Latn",
    "Italian": "ita_Latn", "Tamil": "tam_Taml",
    "Hindi": "hin_Deva", "Kannada": "kan_Knda",
    "Telugu": "tel_Telu"
}

def ist_now():
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%m-%Y %I:%M:%S %p IST")

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    if device == "cuda":
        model = model.half()
    model.to(device)
    model.eval()
    return tokenizer, model, device

def split_text(text, limit=700):
    # Split by paragraphs and sentences without cutting ordinary words.
    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    chunks, current = [], ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > limit:
            if current:
                chunks.append(current)
                current = ""
            cut = piece.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        candidate = f"{current} {piece}".strip()
        if len(candidate) > limit and current:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

def translate_chunk(text, src, tgt, tokenizer, model, device):
    tokenizer.src_lang = src
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    target_id = tokenizer.convert_tokens_to_ids(tgt)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_new_tokens=384,
            num_beams=1,
            do_sample=False
        )
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

# ---------- SESSION ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

def new_thread():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.last_result = ""

st.title("🌍 AIT GLOBAL TECHNOLOGIES")
st.caption("Translate text documents with live chunk progress.")

c1, c2 = st.columns([3, 1])
with c1:
    st.write("**Thread ID:**", st.session_state.thread_id)
with c2:
    st.button("➕ New Thread", on_click=new_thread, use_container_width=True)

a, b = st.columns(2)
with a:
    source_name = st.selectbox("Source language", list(LANGS), index=0)
with b:
    target_name = st.selectbox("Target language", list(LANGS), index=1)

uploaded = st.file_uploader("Upload a TXT document", type=["txt"])
prompt = st.text_input(
    "Translation notes (saved in history)",
    placeholder="e.g. Preserve headings and use formal business language"
)
manual_text = st.text_area("Or paste text here", height=180)

batch_size = st.select_slider(
    "Processing batch size",
    options=[1, 2, 3, 4],
    value=1,
    help="Larger batches may be faster but use more memory."
)

text_input = manual_text
filename = "pasted_text.txt"
if uploaded is not None:
    try:
        text_input = uploaded.getvalue().decode("utf-8-sig")
        filename = uploaded.name
    except UnicodeDecodeError:
        st.error("This TXT file is not UTF-8 encoded. Save it as UTF-8 and upload again.")
        text_input = ""

if st.button("🚀 Translate", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Upload a TXT file or paste text first.")
    elif source_name == target_name:
        st.warning("Choose two different languages.")
    else:
        try:
            tokenizer, model, device = load_model()
            chunks = split_text(text_input)
            if not chunks:
                st.warning("No translatable text was found.")
                st.stop()

            st.info(f"Device: {device.upper()} · Chunks: {len(chunks)}")
            progress = st.progress(0)
            status = st.empty()
            live_output = st.empty()
            translated_parts = []
            start = time.time()

            for start_idx in range(0, len(chunks), batch_size):
                batch = chunks[start_idx:start_idx + batch_size]
                # Batch inputs for fewer generation calls.
                tokenizer.src_lang = LANGS[source_name]
                encoded = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(device)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                            LANGS[target_name]
                        ),
                        max_new_tokens=384,
                        num_beams=1,
                        do_sample=False
                    )
                translated_parts.extend(
                    tokenizer.batch_decode(generated, skip_special_tokens=True)
                )

                done = len(translated_parts)
                progress.progress(done / len(chunks))
                status.write(f"Translating chunk {done}/{len(chunks)}…")
                live_output.text_area(
                    "Translation in progress",
                    "\n\n".join(translated_parts),
                    height=220,
                    key=f"live_{st.session_state.thread_id}_{start_idx}"
                )

            result = "\n\n".join(translated_parts)
            duration = round(time.time() - start, 2)
            st.session_state.last_result = result
            st.session_state.history.insert(0, {
                "thread": st.session_state.thread_id,
                "file": filename,
                "source": source_name,
                "target": target_name,
                "prompt": prompt,
                "chunks": len(chunks),
                "seconds": duration,
                "time": ist_now()
            })
            status.success(f"Translation completed in {duration} seconds.")
            st.rerun()

        except Exception as e:
            st.error(f"Translation failed: {e}")
            st.exception(e)

if st.session_state.get("last_result"):
    st.subheader("✅ Translated Document")
    st.text_area("Final translation", st.session_state.last_result, height=300)
    st.download_button(
        "📥 Download translated TXT",
        data=st.session_state.last_result.encode("utf-8"),
        file_name=f"translated_{LANGS[target_name]}.txt",
        mime="text/plain",
        use_container_width=True
    )

with st.expander("🕘 Translation History"):
    if st.session_state.history:
        for item in st.session_state.history:
            st.markdown(
                f"**{item['source']} → {item['target']}** | "
                f"{item['file']} | {item['seconds']} sec"
            )
            st.caption(
                f"Thread: {item['thread']} · Chunks: {item['chunks']} · "
                f"{item['time']} · Notes: {item['prompt'] or 'None'}"
            )
    else:
        st.caption("No translations yet.")