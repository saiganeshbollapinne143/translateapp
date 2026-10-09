
import re, uuid, time
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES", page_icon="🌍", layout="wide")
st.title("🌍 AIT GLOBAL TECHNOLOGIES")
st.subheader("AI Document Translator — NLLB-200")

MODEL = "facebook/nllb-200-distilled-600M"
CHUNK_TOKENS, MAX_OUTPUT = 400, 384
LANGS = {
    "English": "eng_Latn", "Spanish": "spa_Latn", "French": "fra_Latn",
    "German": "deu_Latn", "Italian": "ita_Latn", "Tamil": "tam_Taml",
    "Hindi": "hin_Deva", "Kannada": "kan_Knda", "Telugu": "tel_Telu"
}

if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state: st.session_state.history = []
if "result" not in st.session_state: st.session_state.result = ""

@st.cache_resource(show_spinner=False)
def load_model():
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, low_cpu_mem_usage=True)
    model.eval()
    return tok, model

def make_chunks(text, tok, limit=CHUNK_TOKENS):
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        candidate = (current + " " + sentence).strip()
        if len(tok(candidate)["input_ids"]) <= limit:
            current = candidate
        else:
            if current: chunks.append(current)
            current = ""
            words, piece = sentence.split(), ""
            for word in words:
                test = (piece + " " + word).strip()
                if len(tok(test)["input_ids"]) <= limit: piece = test
                else:
                    if piece: chunks.append(piece)
                    piece = word
            current = piece
    if current: chunks.append(current)
    return chunks

with st.sidebar:
    st.write("**Thread ID**")
    st.code(st.session_state.thread_id)
    if st.button("➕ New Thread", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.result = ""
        st.rerun()
    st.caption("Device: CPU")

c1, c2 = st.columns(2)
source = c1.selectbox("Source language", list(LANGS), index=0)
target = c2.selectbox("Target language", list(LANGS), index=1)

prompt = st.text_area(
    "✍️ Writing / Translation Prompt",
    value="Translate the entire text accurately into the target language. Preserve the original meaning, names, numbers, dates, paragraphs, tone, and formatting. Do not summarize, omit, or add information. Return only the translated text.",
    height=110,
    help="Tell the translator how to handle the document, for example: use formal business Spanish and preserve all headings and figures."
)
uploaded = st.file_uploader("Upload TXT document", type=["txt"])
file_text = uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else ""
text = st.text_area("📄 Input text / document", value=file_text, height=200,
                    placeholder="Paste your complete document text here...")

if st.button("🚀 Translate with Streaming Output", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Enter text or upload a TXT file.")
        st.stop()
    if source == target:
        st.warning("Choose different source and target languages.")
        st.stop()

    bar, status, live = st.progress(0), st.empty(), st.empty()
    try:
        status.info("Loading model (first run may download it)...")
        tok, model = load_model()
        tok.src_lang = LANGS[source]
        target_id = tok.convert_tokens_to_ids(LANGS[target])
        chunks = make_chunks(text, tok)
        outputs, start = [], time.time()

        for i, chunk in enumerate(chunks, 1):
            status.info(f"Translating chunk {i}/{len(chunks)}...")
            # Prompt guides intended style. NLLB is not an instruction-following model,
            # so only the document text is sent to the translation model.
            inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
            with torch.inference_mode():
                ids = model.generate(
                    **inputs, forced_bos_token_id=target_id,
                    max_new_tokens=MAX_OUTPUT, num_beams=1,
                    do_sample=False, use_cache=True
                )
            outputs.append(tok.decode(ids[0], skip_special_tokens=True).strip())
            partial = "\n\n".join(outputs)
            live.text_area("🔄 Live translated output", value=partial,
                           height=280, key=f"live_{st.session_state.thread_id}_{i}")
            bar.progress(i / len(chunks),
                         text=f"{i}/{len(chunks)} chunks completed")
        
        st.session_state.result = "\n\n".join(outputs)
        stamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%m-%Y %I:%M:%S %p IST")
        st.session_state.history.append({
            "thread": st.session_state.thread_id, "time": stamp,
            "source": source, "target": target, "prompt": prompt,
            "original": text, "translated": st.session_state.result,
            "seconds": round(time.time() - start, 1)
        })
        status.success(f"Translation complete in {time.time()-start:.1f} seconds.")
    except Exception as e:
        status.error("Translation failed:")
        st.exception(e)

if st.session_state.result:
    st.download_button("📥 Download translated TXT", st.session_state.result,
                       file_name=f"translation_{target.lower()}.txt",
                       mime="text/plain", use_container_width=True)

st.divider()
st.subheader("🕘 Translation History")
if not st.session_state.history:
    st.caption("Your completed translations will appear here.")
else:
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"{item['source']} → {item['target']} | {item['time']}"):
            st.caption(f"Thread: {item['thread']} | Time: {item['seconds']} seconds")
            st.write("**Writing prompt**")
            st.write(item["prompt"])
            st.write("**Original text**")
            st.text(item["original"])
            st.write("**Translation**")
            st.text(item["translated"])
            st.download_button("Download this translation", item["translated"],
                               file_name=f"translation_history_{i+1}.txt",
                               mime="text/plain", key=f"hist_{i}_{item['thread']}")

