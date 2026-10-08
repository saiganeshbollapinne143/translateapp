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


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "facebook/nllb-200-distilled-600M"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "chroma_db"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16 if DEVICE == "cuda" else 8
NUM_BEAMS = 4 if DEVICE == "cuda" else 2

MAX_CHUNK_CHARS = 400
MAX_INPUT_TOKENS = 256

HISTORY_LIMIT = 20
PREVIEW_CHARS = 1500

QUANTIZE_ON_CPU = False


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


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading NLLB-200 model...")
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    ).eval()

    if DEVICE == "cuda":

        model = model.half().to(DEVICE)

    elif QUANTIZE_ON_CPU:

        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

    return tokenizer, model


# ============================================================
# CHROMA DATABASE
# ============================================================

@st.cache_resource(show_spinner="Opening Chroma database...")
def get_collection():

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device=DEVICE,
    )

    return client.get_or_create_collection(
        name="translations",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ============================================================
# TRANSLATION
# ============================================================

def translate_batch(
    texts,
    src_code,
    tgt_code,
    tokenizer,
    model,
    on_progress=None,
):

    tokenizer.src_lang = src_code

    forced_bos = tokenizer.convert_tokens_to_ids(
        tgt_code
    )

    order = sorted(
        range(len(texts)),
        key=lambda i: len(texts[i])
    )

    results = [""] * len(texts)

    for start in range(
        0,
        len(order),
        BATCH_SIZE,
    ):

        idx = order[
            start:start + BATCH_SIZE
        ]

        batch = [
            texts[i]
            for i in idx
        ]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(DEVICE)

        max_new = min(
            int(
                inputs["input_ids"].shape[1] * 1.5
            ) + 10,
            400,
        )

        with torch.inference_mode():

            out = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new,
                num_beams=NUM_BEAMS,
            )

        decoded = tokenizer.batch_decode(
            out,
            skip_special_tokens=True,
        )

        for i, t in zip(
            idx,
            decoded,
        ):

            results[i] = t

        if on_progress:

            on_progress(
                min(
                    start + BATCH_SIZE,
                    len(order),
                ),
                len(order),
            )

    return results


# ============================================================
# TRANSLATION MEMORY
# ============================================================

def make_id(
    text,
    src_lang,
    tgt_lang,
):

    return hashlib.sha1(
        f"{src_lang}|{tgt_lang}|{text}".encode(
            "utf-8"
        )
    ).hexdigest()


def memory_lookup(
    collection,
    texts,
    src_lang,
    tgt_lang,
):

    ids = {
        make_id(
            t,
            src_lang,
            tgt_lang,
        ): t
        for t in texts
    }

    if not ids:
        return {}

    if collection.count() == 0:
        return {}

    found = collection.get(
        ids=list(ids),
        include=["metadatas"],
    )

    return {
        ids[i]: m["translation"]
        for i, m in zip(
            found["ids"],
            found["metadatas"],
        )
    }


def save_records(
    collection,
    pairs,
    src_lang,
    tgt_lang,
    source="",
    thread_id="",
):

    if not pairs:
        return

    now = time.time()

    collection.upsert(

        ids=[
            make_id(
                s,
                src_lang,
                tgt_lang,
            )
            for s, _ in pairs
        ],

        documents=[
            s
            for s, _ in pairs
        ],

        metadatas=[

            {
                "translation": t,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "ts": now + i * 1e-3,
                "source": source,
                "thread_id": thread_id,
            }

            for i, (_, t) in enumerate(pairs)
        ],
    )


def run_translation(
    texts,
    src_lang,
    tgt_lang,
    tokenizer,
    model,
    collection,
    source="",
    thread_id="",
    on_progress=None,
):

    unique = list(
        dict.fromkeys(texts)
    )

    done = memory_lookup(
        collection,
        unique,
        src_lang,
        tgt_lang,
    )

    todo = [
        t
        for t in unique
        if t not in done
    ]

    if todo:

        new = translate_batch(

            todo,

            LANGUAGES[src_lang],

            LANGUAGES[tgt_lang],

            tokenizer,

            model,

            on_progress,
        )

        save_records(

            collection,

            list(
                zip(
                    todo,
                    new,
                )
            ),

            src_lang,
            tgt_lang,

            source,

            thread_id,
        )

        done.update(
            zip(
                todo,
                new,
            )
        )

    return (
        [done[t] for t in texts],
        len(unique) - len(todo),
    )


# ============================================================
# THREAD MANAGEMENT
# ============================================================

def create_thread():

    return str(
        uuid.uuid4()
    )


def get_threads(collection):

    if collection.count() == 0:
        return []

    data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    threads = {}

    for doc, meta in zip(
        data["documents"],
        data["metadatas"],
    ):

        thread_id = meta.get(
            "thread_id"
        )

        if not thread_id:
            continue

        ts = meta.get(
            "ts",
            0,
        )

        if thread_id not in threads:

            threads[thread_id] = {

                "thread_id": thread_id,

                "first_message": doc,

                "last_ts": ts,

                "count": 0,
            }

        threads[
            thread_id
        ]["last_ts"] = max(
            threads[
                thread_id
            ]["last_ts"],
            ts,
        )

        threads[
            thread_id
        ]["count"] += 1

    return sorted(

        threads.values(),

        key=lambda x: x["last_ts"],

        reverse=True,
    )


def load_thread_history(
    collection,
    thread_id,
):

    if collection.count() == 0:
        return []

    data = collection.get(

        where={
            "thread_id": thread_id
        },

        include=[
            "documents",
            "metadatas",
        ],
    )

    rows = sorted(

        zip(
            data["documents"],
            data["metadatas"],
        ),

        key=lambda r: r[1].get(
            "ts",
            0,
        ),
    )

    messages = []

    for doc, meta in rows:

        messages.append({

            "role": "user",

            "content": doc,

        })

        messages.append({

            "role": "assistant",

            "content":
                f"**{meta['tgt_lang']}:** "
                f"{meta['translation']}",

        })

    return messages


# ============================================================
# FILE HANDLING
# ============================================================

def extract_text(uploaded):

    name = uploaded.name.lower()

    if name.endswith(".txt"):

        return uploaded.getvalue().decode(
            "utf-8",
            errors="ignore",
        )

    if name.endswith(".pdf"):

        return "\n".join(

            (
                p.extract_text()
                or ""
            )

            for p in PdfReader(
                uploaded
            ).pages
        )

    if name.endswith(".docx"):

        return "\n".join(

            p.text

            for p in Document(
                uploaded
            ).paragraphs
        )

    return ""


def _hard_wrap(
    sentence,
    max_chars,
):

    parts = []

    current = ""

    for word in sentence.split():

        if (
            current
            and len(current)
            + len(word)
            + 1
            > max_chars
        ):

            parts.append(
                current
            )

            current = word

        else:

            current = (
                f"{current} {word}"
                .strip()
            )

    if current:

        parts.append(
            current
        )

    return parts


def split_into_chunks(
    text,
    max_chars=MAX_CHUNK_CHARS,
):

    chunks = []

    for para in (

        p.strip()

        for p in text.splitlines()

        if p.strip()
    ):

        current = ""

        for s in re.split(
            r"(?<=[.!?।])\s+",
            para,
        ):

            pieces = (

                _hard_wrap(
                    s,
                    max_chars,
                )

                if len(s) > max_chars

                else [s]
            )

            for piece in pieces:

                if (

                    current

                    and len(current)
                    + len(piece)
                    + 1
                    > max_chars
                ):

                    chunks.append(
                        current
                    )

                    current = piece

                else:

                    current = (
                        f"{current} {piece}"
                        .strip()
                    )

        if current:

            chunks.append(
                current
            )

    return chunks


# ============================================================
# MESSAGE HANDLERS
# ============================================================

def handle_text(
    text,
    src_lang,
    tgt_lang,
    tokenizer,
    model,
    collection,
    thread_id,
):

    with st.spinner(
        "Translating..."
    ):

        (result,), _ = run_translation(

            [text],

            src_lang,

            tgt_lang,

            tokenizer,

            model,

            collection,

            thread_id=thread_id,
        )

    return {

        "role": "assistant",

        "content":
            f"**{tgt_lang}:** {result}",
    }


def handle_file(
    f,
    src_lang,
    tgt_lang,
    tokenizer,
    model,
    collection,
    thread_id,
):

    try:

        chunks = split_into_chunks(
            extract_text(f)
        )

    except Exception as e:

        return {

            "role": "assistant",

            "content":
                f"Couldn't read **{f.name}**: {e}",
        }

    if not chunks:

        return {

            "role": "assistant",

            "content":
                f"No readable text found in **{f.name}**.",
        }

    bar = st.progress(
        0.0,
        text=f"Translating {f.name}...",
    )

    def on_progress(
        done,
        total,
    ):

        bar.progress(

            done / total,

            text=
                f"Translating "
                f"{f.name} "
                f"({done}/{total})",
        )

    translated_chunks, reused = run_translation(

        chunks,

        src_lang,

        tgt_lang,

        tokenizer,

        model,

        collection,

        source=f.name,

        thread_id=thread_id,

        on_progress=on_progress,
    )

    bar.empty()

    full = "\n\n".join(
        translated_chunks
    )

    preview = (

        full[:PREVIEW_CHARS]

        + (
            "..."
            if len(full) > PREVIEW_CHARS
            else ""
        )
    )

    note = (

        f"  \n<small>"
        f"{reused} segment(s) reused "
        f"from saved translations"
        f"</small>"

        if reused

        else ""
    )

    return {

        "role": "assistant",

        "content":
            f"**{tgt_lang} translation "
            f"of {f.name}:**\n\n"
            f"{preview}{note}",

        "download": {

            "data":
                full.encode(
                    "utf-8"
                ),

            "name":
                f"{f.name.rsplit('.', 1)[0]}"
                f"_{tgt_lang}.txt",
        },
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(

    page_title="Translation Chat",

    page_icon="💬",

    layout="wide",
)


st.title(
    "💬 Translation Chat"
)


# ============================================================
# LOAD MODEL + DATABASE
# ============================================================

tokenizer, model = load_model()

collection = get_collection()


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = (
        create_thread()
    )


if "messages" not in st.session_state:

    st.session_state.messages = (
        load_thread_history(
            collection,
            st.session_state.thread_id,
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Settings"
    )

    src_lang = st.selectbox(

        "I write in",

        list(LANGUAGES),

        index=0,

        key="src",
    )

    tgt_lang = st.selectbox(

        "Translate to",

        list(LANGUAGES),

        index=1,

        key="tgt",
    )


    def swap_languages():

        st.session_state.src, \
        st.session_state.tgt = (

            st.session_state.tgt,

            st.session_state.src,
        )


    st.button(

        "⇄ Swap languages",

        on_click=swap_languages,
    )


    # ========================================================
    # NEW THREAD
    # ========================================================

    if st.button(

        "➕ New Thread",

        use_container_width=True,
    ):

        st.session_state.thread_id = (
            create_thread()
        )

        st.session_state.messages = []

        st.rerun()


    # ========================================================
    # CURRENT THREAD
    # ========================================================

    st.caption(

        "Current Thread: "
        + st.session_state.thread_id[:8]
        + "..."
    )


    # ========================================================
    # CLEAR CURRENT CHAT VIEW
    # ========================================================

    if st.button(

        "🗑️ Clear Chat View",

        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


    st.caption(

        f"Saved records: "
        f"{collection.count()} "
        f"· Running on "
        f"{DEVICE.upper()}"
    )


    # ========================================================
    # PAST THREADS
    # ========================================================

    st.divider()

    st.header(
        "🗂️ Past Threads"
    )

    threads = get_threads(
        collection
    )

    if threads:

        for thread in threads:

            title = thread[
                "first_message"
            ]

            if len(title) > 35:

                title = (
                    title[:35]
                    + "..."
                )

            timestamp = (
                datetime.fromtimestamp(
                    thread["last_ts"]
                ).strftime(
                    "%d %b %Y, %I:%M %p"
                )
            )

            label = (
                f"💬 {title}\n"
                f"🕒 {timestamp}"
            )

            if st.button(

                label,

                key=
                    f"thread_"
                    f"{thread['thread_id']}",

                use_container_width=True,
            ):

                st.session_state.thread_id = (
                    thread["thread_id"]
                )

                st.session_state.messages = (
                    load_thread_history(
                        collection,
                        thread["thread_id"],
                    )
                )

                st.rerun()

    else:

        st.caption(
            "No past threads yet."
        )


    # ========================================================
    # SEARCH
    # ========================================================

    st.divider()

    st.header(
        "🔎 Search Past Translations"
    )

    query = st.text_input(
        "Search by meaning"
    )

    if (
        query
        and collection.count() > 0
    ):

        res = collection.query(

            query_texts=[query],

            n_results=min(
                5,
                collection.count(),
            ),
        )

        for doc, meta, dist in zip(

            res["documents"][0],

            res["metadatas"][0],

            res["distances"][0],
        ):

            st.markdown(

                f"**{doc}**  \n"
                f"→ {meta['tgt_lang']}: "
                f"{meta['translation']}  \n"
                f"<small>"
                f"similarity "
                f"{1 - dist:.2f}"
                f"</small>",

                unsafe_allow_html=True,
            )

            st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

for i, msg in enumerate(
    st.session_state.messages
):

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(

            msg["content"],

            unsafe_allow_html=True,
        )

        if "download" in msg:

            st.download_button(

                "⬇️ Download translation",

                data=
                    msg["download"]["data"],

                file_name=
                    msg["download"]["name"],

                key=f"dl_{i}",
            )


# ============================================================
# NEW INPUT
# ============================================================

prompt = st.chat_input(

    f"Type in {src_lang} "
    f"or drop a file "
    f"(.txt, .pdf, .docx)...",

    accept_file="multiple",

    file_type=[
        "txt",
        "pdf",
        "docx",
    ],
)


if prompt:

    text = (
        prompt.text or ""
    ).strip()

    files = (
        prompt.files or []
    )

    new_messages = []


    # ========================================================
    # SAME LANGUAGE CHECK
    # ========================================================

    if src_lang == tgt_lang:

        with st.chat_message(
            "assistant"
        ):

            st.warning(

                "Source and target "
                "languages are the same. "
                "Please change one "
                "in the sidebar."
            )


    else:

        # ====================================================
        # TEXT
        # ====================================================

        if text:

            with st.chat_message(
                "user"
            ):

                st.markdown(text)


            with st.chat_message(
                "assistant"
            ):

                reply = handle_text(

                    text,

                    src_lang,

                    tgt_lang,

                    tokenizer,

                    model,

                    collection,

                    st.session_state.thread_id,
                )


                st.markdown(

                    reply["content"],

                    unsafe_allow_html=True,
                )


            new_messages += [

                {
                    "role": "user",

                    "content": text,
                },

                reply,
            ]


        # ====================================================
        # FILES
        # ====================================================

        for f in files:

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    f"📎 {f.name}"
                )


            with st.chat_message(
                "assistant"
            ):

                reply = handle_file(

                    f,

                    src_lang,

                    tgt_lang,

                    tokenizer,

                    model,

                    collection,

                    st.session_state.thread_id,
                )


                st.markdown(

                    reply["content"],

                    unsafe_allow_html=True,
                )


                if "download" in reply:

                    st.download_button(

                        "⬇️ Download translation",

                        data=
                            reply["download"]["data"],

                        file_name=
                            reply["download"]["name"],

                        key=
                            f"new_dl_"
                            f"{uuid.uuid4()}",
                    )


            new_messages += [

                {
                    "role": "user",

                    "content":
                        f"📎 {f.name}",
                },

                reply,
            ]


        # ====================================================
        # SAVE CURRENT THREAD IN SESSION
        # ====================================================

        st.session_state.messages.extend(
            new_messages
        )

        st.rerun()