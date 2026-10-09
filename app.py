
import io
import uuid
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- APP CONFIG --------------------
st.set_page_config(page_title="AI Document Translator", page_icon="🌍", layout="wide")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DB_PATH = Path("chroma_db")
CHUNK_SIZE = 700
MAX_NEW_TOKENS = 256

LANGUAGES = {
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Arabic": "arb_Arab",
    "Russian": "rus_Cyrl",
    "Telugu": "tel_Telu",
    "Bengali": "ben_Beng",
    "Urdu": "urd_Arab",
    "Marathi": "mar_Deva",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Thai": "tha_Thai",
    "Vietnamese": "vie_Latn",
    "Indonesian": "ind_Latn",
    "Dutch": "nld_Latn",
    "Turkish": "tur_Latn",
    "Polish": "pol_Latn",
    "Swahili": "swh_Latn",
}

st.title("🌍 NLLB-200 AI Document Translator")
st.caption("Translate text and TXT documents with chunk-level progress and conversation history.")

# -------------------- MODEL --------------------
@st.cache_resource(show_spinner="Loading NLLB-200 model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")
    return tokenizer, model

# -------------------- DATABASE --------------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(name="translation_history")

def save_history(thread_id, source, target, original, translated, prompt):
    try:
        collection = get_collection()
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[original[:10000] + "\n\nTRANSLATION:\n" + translated[:10000]],
            metadatas=[{
                "thread_id": thread_id,
                "source": source,
                "target": target,
                "timestamp": timestamp,
                "prompt": prompt[:1000],
            }],
        )
    except Exception as e:
        st.warning(f"History could not be saved: {e}")

# -------------------- TEXT CHUNKING --------------------
def split_text(text, tokenizer, chunk_size=CHUNK_SIZE):
    # Split on words to avoid cutting ordinary text in the middle.
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        candidate = " ".join(current)
        token_count = len(tokenizer(
            candidate, add_special_tokens=False
        )["input_ids"])

        if token_count > chunk_size:
            last_word = current.pop()
            if current:
                chunks.append(" ".join(current))
            current = [last_word]

    if current:
        chunks.append(" ".join(current))

    return chunks or ([text] if text.strip() else [])

# -------------------- TRANSLATION --------------------
def translate_chunk(chunk, tokenizer, model, source_code, target_code):
    tokenizer.src_lang = source_code

    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    target_id = tokenizer.convert_tokens_to_ids(target_code)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            do_sample=False,
        )

    return tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()

def translate_document(text, tokenizer, model, source_code,
                       target_code, prompt, progress, status):
    chunks = split_text(text, tokenizer)
    results = []
    start = time.time()

    for i, chunk in enumerate(chunks, start=1):
        status.info(f"Translating chunk {i}/{len(chunks)}...")
        translated = translate_chunk(
            chunk, tokenizer, model, source_code, target_code
        )
        results.append(translated)

        progress.progress(
            int(i * 100 / len(chunks)),
            text=f"Translated {i}/{len(chunks)} chunks",
        )

    elapsed = time.time() - start
    return "\n\n".join(results), len(chunks), elapsed

# -------------------- SESSION STATE --------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8].upper()
if "history" not in st.session_state:
    st.session_state.history = []
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "last_original" not in st.session_state:
    st.session_state.last_original = ""

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("🧵 Conversation")
    st.code(st.session_state.thread_id)

    if st.button("➕ New Thread", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8].upper()
        st.session_state.history = []
        st.session_state.translated_text = ""
        st.session_state.last_original = ""
        st.rerun()

    st.divider()
    st.subheader("🕘 Current Thread History")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            with st.expander(
                f"{item['timestamp']} · {item['source']} → {item['target']}"
            ):
                st.write("**Input**")
                st.write(item["original"][:1500])
                st.write("**Translation**")
                st.write(item["translated"][:1500])
    else:
        st.caption("Translations in this thread will appear here.")

# -------------------- INPUT UI --------------------
left, right = st.columns(2)
with left:
    source_name = st.selectbox(
        "Source language", list(LANGUAGES.keys()), index=0
    )
with right:
    target_options = [x for x in LANGUAGES if x != source_name]
    default_target = target_options.index("Spanish") if "Spanish" in target_options else 0
    target_name = st.selectbox(
        "Target language", target_options, index=default_target
    )

uploaded_file = st.file_uploader("📄 Upload a TXT document", type=["txt"])

uploaded_text = ""
if uploaded_file is not None:
    uploaded_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    st.success(f"Loaded {uploaded_file.name} · {len(uploaded_text):,} characters")

prompt = st.text_area(
    "Translation instructions",
    value="Translate accurately and naturally. Preserve the original meaning, names, numbers, paragraphs, and tone. Do not summarize or add new information.",
    height=100,
    help="Instructions are saved with history. NLLB is a translation model, so arbitrary prompts are not interpreted like chat-model prompts.",
)

typed_text = st.text_area(
    "✍️ Enter text to translate",
    value="",
    height=180,
    placeholder="Type or paste your document text here...",
)

text_to_translate = typed_text.strip() or uploaded_text.strip()

st.caption(
    f"Input: {len(text_to_translate):,} characters · "
    f"Thread: {st.session_state.thread_id}"
)

translate_clicked = st.button(
    "🌐 Translate Document",
    type="primary",
    use_container_width=True,
    disabled=not bool(text_to_translate),
)

# -------------------- RUN TRANSLATION --------------------
if translate_clicked:
    progress = st.progress(0, text="Preparing translation...")
    status = st.empty()

    try:
        tokenizer, model = load_model()

        translated, chunk_count, elapsed = translate_document(
            text_to_translate,
            tokenizer,
            model,
            LANGUAGES[source_name],
            LANGUAGES[target_name],
            prompt,
            progress,
            status,
        )

        if not translated:
            status.warning("No translated text was returned.")
        else:
            timestamp = datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
            st.session_state.translated_text = translated
            st.session_state.last_original = text_to_translate

            record = {
                "timestamp": timestamp,
                "source": source_name,
                "target": target_name,
                "original": text_to_translate,
                "translated": translated,
                "chunks": chunk_count,
                "elapsed": round(elapsed, 2),
            }
            st.session_state.history.append(record)

            save_history(
                st.session_state.thread_id,
                source_name,
                target_name,
                text_to_translate,
                translated,
                prompt,
            )

            status.success(
                f"Translation completed · {chunk_count} chunks · "
                f"{elapsed:.1f} seconds"
            )

    except Exception as e:
        status.error(f"Translation failed: {type(e).__name__}: {e}")
        st.exception(e)

# -------------------- OUTPUT --------------------
if st.session_state.translated_text:
    st.divider()
    st.subheader("✅ Translated Document")

    st.text_area(
        "Translation output",
        value=st.session_state.translated_text,
        height=300,
        key="translation_display",
    )

    st.download_button(
        "📥 Download translated TXT",
        data=st.session_state.translated_text.encode("utf-8"),
        file_name=f"translation_{st.session_state.thread_id}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.download_button(
        "📥 Download original TXT",
        data=st.session_state.last_original.encode("utf-8"),
        file_name=f"original_{st.session_state.thread_id}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    if st.session_state.history:
        latest = st.session_state.history[-1]
        a, b, c = st.columns(3)
        a.metric("Chunks", latest["chunks"])
        b.metric("Seconds", latest["elapsed"])
        c.metric("Characters", len(latest["translated"]))

st.caption(
    "Note: ChromaDB data stored on Streamlit Cloud's local filesystem may "
    "not persist across app rebuilds or restarts. Export important history."
)
