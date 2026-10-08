import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import re
import time
import uuid
from datetime import datetime

import chromadb
import streamlit as st
import torch
from chromadb.utils import embedding_functions
from docx import Document
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Translation Chat", page_icon="💬", layout="wide")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "/tmp/chroma_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16 if DEVICE == "cuda" else 8
NUM_BEAMS = 4 if DEVICE == "cuda" else 2
MAX_CHUNK_CHARS = 400
MAX_INPUT_TOKENS = 256
PREVIEW_CHARS = 1500

LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
}


@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()

    if DEVICE == "cuda":
        model = model.half().to(DEVICE)

    return tokenizer, model


@st.cache_resource(show_spinner="Opening database...")
def get_chroma():
    client = chromadb.PersistentClient(path=DB_PATH)

    embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device=DEVICE
    )

    translation_memory = client.get_or_create_collection(
        name="translation_memory",
        embedding_function=embedding,
        metadata={"hnsw:space": "cosine"},
    )

    chat_history = client.get_or_create_collection(
        name="chat_history",
        embedding_function=embedding,
        metadata={"hnsw:space": "cosine"},
    )

    return translation_memory, chat_history


translation_memory, chat_history = get_chroma()


def create_thread_id():
    return str(uuid.uuid4())


if "thread_id" not in st.session_state:
    st.session_state.thread_id = create_thread_id()

if "messages" not in st.session_state:
    st.session_state.messages = []


def translate_batch(texts, src_code, tgt_code, tokenizer, model):
    tokenizer.src_lang = src_code
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    results = [""] * len(texts)

    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        batch = [texts[i] for i in idx]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        ).to(DEVICE)

        max_new = min(
            int(inputs["input_ids"].shape[1] * 1.5) + 10,
            400
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new,
                max_length=None,
                num_beams=NUM_BEAMS,
            )

        decoded = tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )

        for i, text in zip(idx, decoded):
            results[i] = text

    return results


def translation_id(text, src_lang, tgt_lang):
    return hashlib.sha1(
        f"{src_lang}|{tgt_lang}|{text}".encode()
    ).hexdigest()


def get_saved_translations(texts, src_lang, tgt_lang):
    if not texts or translation_memory.count() == 0:
        return {}

    ids = [
        translation_id(x, src_lang, tgt_lang)
        for x in texts
    ]

    result = translation_memory.get(
        ids=ids,
        include=["metadatas"]
    )

    saved = {}

    for record_id, metadata in zip(
        result["ids"],
        result["metadatas"]
    ):
        index = ids.index(record_id)
        saved[texts[index]] = metadata["translation"]

    return saved


def save_translation_memory(
    pairs,
    src_lang,
    tgt_lang,
    source=""
):
    if not pairs:
        return

    translation_memory.upsert(
        ids=[
            translation_id(x, src_lang, tgt_lang)
            for x, _ in pairs
        ],
        documents=[
            x for x, _ in pairs
        ],
        metadatas=[
            {
                "translation": y,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "source": source,
                "ts": time.time() + i * 0.001,
            }
            for i, (_, y) in enumerate(pairs)
        ],
    )


def save_chat_message(
    thread_id,
    role,
    content,
    src_lang="",
    tgt_lang="",
    source=""
):
    message_id = str(uuid.uuid4())

    chat_history.upsert(
        ids=[
            hashlib.sha1(
                f"{thread_id}|{message_id}".encode()
            ).hexdigest()
        ],
        documents=[content],
        metadatas=[{
            "thread_id": thread_id,
            "role": role,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "source": source,
            "ts": time.time(),
            "message_id": message_id,
        }],
    )


def load_thread(thread_id):
    if chat_history.count() == 0:
        return []

    result = chat_history.get(
        where={"thread_id": thread_id},
        include=["documents", "metadatas"],
    )

    rows = sorted(
        zip(
            result["documents"],
            result["metadatas"]
        ),
        key=lambda x: x[1].get("ts", 0),
    )

    return [
        {
            "role": metadata["role"],
            "content": document
        }
        for document, metadata in rows
    ]


def get_past_threads():
    if chat_history.count() == 0:
        return []

    result = chat_history.get(
        include=["documents", "metadatas"]
    )

    threads = {}

    for document, metadata in zip(
        result["documents"],
        result["metadatas"]
    ):
        tid = metadata.get("thread_id")

        if not tid:
            continue

        if tid not in threads:
            threads[tid] = {
                "thread_id": tid,
                "title": document,
                "timestamp": metadata.get("ts", 0),
            }

    return sorted(
        threads.values(),
        key=lambda x: x["timestamp"],
        reverse=True,
    )


def run_translation(
    texts,
    src_lang,
    tgt_lang,
    tokenizer,
    model,
    source=""
):
    unique = list(dict.fromkeys(texts))

    saved = get_saved_translations(
        unique,
        src_lang,
        tgt_lang
    )

    todo = [
        x for x in unique
        if x not in saved
    ]

    reused = len(unique) - len(todo)

    if todo:
        translated = translate_batch(
            todo,
            LANGUAGES[src_lang],
            LANGUAGES[tgt_lang],
            tokenizer,
            model
        )

        pairs = list(zip(todo, translated))

        save_translation_memory(
            pairs,
            src_lang,
            tgt_lang,
            source
        )

        saved.update(pairs)

    return [
        saved[x] for x in texts
    ], reused


def extract_text(uploaded):
    name = uploaded.name.lower()

    if name.endswith(".txt"):
        return uploaded.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

    if name.endswith(".pdf"):
        return "\n".join(
            page.extract_text() or ""
            for page in PdfReader(uploaded).pages
        )

    if name.endswith(".docx"):
        return "\n".join(
            p.text
            for p in Document(uploaded).paragraphs
        )

    return ""


def split_into_chunks(text):
    chunks = []

    for paragraph in (
        p.strip()
        for p in text.splitlines()
        if p.strip()
    ):
        sentences = re.split(
            r"(?<=[.!?।])\s+",
            paragraph
        )

        current = ""

        for sentence in sentences:
            words = sentence.split()
            parts = []
            temp = ""

            for word in words:
                if (
                    temp
                    and len(temp) + len(word) + 1
                    > MAX_CHUNK_CHARS
                ):
                    parts.append(temp)
                    temp = word
                else:
                    temp = f"{temp} {word}".strip()

            if temp:
                parts.append(temp)

            for part in parts:
                if (
                    current
                    and len(current) + len(part) + 1
                    > MAX_CHUNK_CHARS
                ):
                    chunks.append(current)
                    current = part
                else:
                    current = f"{current} {part}".strip()

        if current:
            chunks.append(current)

    return chunks


def handle_text(
    text,
    src_lang,
    tgt_lang,
    tokenizer,
    model
):
    with st.spinner("Translating..."):
        (result,), reused = run_translation(
            [text],
            src_lang,
            tgt_lang,
            tokenizer,
            model
        )

    answer = f"**{tgt_lang}:** {result}"

    save_chat_message(
        st.session_state.thread_id,
        "user",
        text,
        src_lang,
        tgt_lang
    )

    save_chat_message(
        st.session_state.thread_id,
        "assistant",
        answer,
        src_lang,
        tgt_lang
    )

    return {
        "role": "assistant",
        "content": answer
    }


def handle_file(
    uploaded,
    src_lang,
    tgt_lang,
    tokenizer,
    model
):
    try:
        text = extract_text(uploaded)

    except Exception as e:
        return {
            "role": "assistant",
            "content": (
                f"Couldn't read **{uploaded.name}**: {e}"
            )
        }

    if not text.strip():
        return {
            "role": "assistant",
            "content": (
                f"No readable text found in "
                f"**{uploaded.name}**."
            )
        }

    chunks = split_into_chunks(text)

    with st.spinner(
        f"Translating {uploaded.name}..."
    ):
        translated, reused = run_translation(
            chunks,
            src_lang,
            tgt_lang,
            tokenizer,
            model,
            uploaded.name
        )

    full = "\n\n".join(translated)

    preview = full[:PREVIEW_CHARS]

    if len(full) > PREVIEW_CHARS:
        preview += "..."

    answer = (
        f"**{tgt_lang} translation of "
        f"{uploaded.name}:**\n\n{preview}"
    )

    if reused:
        answer += (
            f"\n\n<small>{reused} segment(s) "
            f"reused from saved translation "
            f"memory.</small>"
        )

    save_chat_message(
        st.session_state.thread_id,
        "user",
        f"📎 {uploaded.name}",
        src_lang,
        tgt_lang,
        uploaded.name
    )

    save_chat_message(
        st.session_state.thread_id,
        "assistant",
        answer,
        src_lang,
        tgt_lang,
        uploaded.name
    )

    return {
        "role": "assistant",
        "content": answer,
        "download": {
            "data": full.encode("utf-8"),
            "name": (
                f"{uploaded.name.rsplit('.', 1)[0]}"
                f"_{tgt_lang}.txt"
            ),
        },
    }


st.title("💬 Translation Chat")

tokenizer, model = load_model()

if not st.session_state.messages:
    st.session_state.messages = load_thread(
        st.session_state.thread_id
    )


with st.sidebar:
    st.header("⚙️ Settings")

    src_lang = st.selectbox(
        "I write in",
        list(LANGUAGES),
        index=0
    )

    tgt_lang = st.selectbox(
        "Translate to",
        list(LANGUAGES),
        index=1
    )

    if st.button(
        "➕ New Conversation",
        use_container_width=True
    ):
        st.session_state.thread_id = create_thread_id()
        st.session_state.messages = []
        st.rerun()

    st.caption("Current conversation")
    st.code(st.session_state.thread_id)

    st.divider()

    st.header("🗂️ Past Conversations")

    for thread in get_past_threads():
        title = thread["title"][:35]

        date = datetime.fromtimestamp(
            thread["timestamp"]
        ).strftime(
            "%d %b %Y, %I:%M %p"
        )

        if st.button(
            f"💬 {title}\n🕒 {date}",
            key=f"past_{thread['thread_id']}",
            use_container_width=True,
        ):
            st.session_state.thread_id = (
                thread["thread_id"]
            )

            st.session_state.messages = (
                load_thread(thread["thread_id"])
            )

            st.rerun()

    st.divider()

    st.header("🔎 Search Past Translations")

    query = st.text_input(
        "Search by meaning"
    )

    if (
        query
        and translation_memory.count() > 0
    ):
        results = translation_memory.query(
            query_texts=[query],
            n_results=min(
                5,
                translation_memory.count()
            ),
        )

        for (
            document,
            metadata,
            distance
        ) in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            st.markdown(
                f"**{document}**  \n"
                f"→ {metadata['tgt_lang']}: "
                f"{metadata['translation']}  \n"
                f"<small>similarity "
                f"{1-distance:.2f}</small>",
                unsafe_allow_html=True,
            )

    st.caption(
        f"Translation memory: "
        f"{translation_memory.count()}"
    )

    st.caption(
        f"Chat messages: "
        f"{chat_history.count()}"
    )

    st.caption(
        f"Device: {DEVICE.upper()}"
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(
            message["content"],
            unsafe_allow_html=True
        )


prompt = st.chat_input(
    f"Type in {src_lang} or drop a file "
    f"(.txt, .pdf, .docx)...",
    accept_file="multiple",
    file_type=["txt", "pdf", "docx"],
)


if prompt:
    text = (prompt.text or "").strip()
    files = prompt.files or []
    new_messages = []

    if src_lang == tgt_lang:
        with st.chat_message("assistant"):
            st.warning(
                "Source and target languages are "
                "the same. Please select different "
                "languages."
            )

    else:
        if text:
            with st.chat_message("user"):
                st.markdown(text)

            with st.chat_message("assistant"):
                reply = handle_text(
                    text,
                    src_lang,
                    tgt_lang,
                    tokenizer,
                    model
                )

                st.markdown(
                    reply["content"],
                    unsafe_allow_html=True
                )

            new_messages.extend([
                {
                    "role": "user",
                    "content": text
                },
                reply,
            ])

        for uploaded_file in files:
            with st.chat_message("user"):
                st.markdown(
                    f"📎 {uploaded_file.name}"
                )

            with st.chat_message("assistant"):
                reply = handle_file(
                    uploaded_file,
                    src_lang,
                    tgt_lang,
                    tokenizer,
                    model
                )

                st.markdown(
                    reply["content"],
                    unsafe_allow_html=True
                )

                if "download" in reply:
                    st.download_button(
                        "⬇️ Download translation",
                        data=reply["download"]["data"],
                        file_name=reply["download"]["name"],
                        key=f"download_{uuid.uuid4()}",
                    )

            new_messages.extend([
                {
                    "role": "user",
                    "content": (
                        f"📎 {uploaded_file.name}"
                    )
                },
                reply,
            ])

        st.session_state.messages.extend(
            new_messages
        )

        st.rerun()