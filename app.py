
import io, os, re, json, uuid
from datetime import datetime
import streamlit as st
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="AIT Translator", page_icon="🌐", layout="centered")

# Minimal blue title card
st.markdown("""
<div style="background:#123d78;padding:18px;border-radius:12px;
text-align:center;margin-bottom:18px;">
<span style="color:#ffd43b;font-size:30px;font-weight:800;">AIT</span>
<span style="color:white;font-size:25px;font-weight:700;">
 GLOBAL TECHNOLOGIES</span><br>
<span style="color:white;font-size:17px;">TRANSLATOR</span>
</div>
""", unsafe_allow_html=True)

LANG = {
    "English":"eng_Latn", "Tamil":"tam_Taml", "Hindi":"hin_Deva",
    "Telugu":"tel_Telu", "Kannada":"kan_Knda",
    "Malayalam":"mal_Mlym", "French":"fra_Latn",
    "Spanish":"spa_Latn", "German":"deu_Latn",
    "Chinese":"zho_Hans", "Japanese":"jpn_Jpan",
    "Arabic":"arb_Arab", "Russian":"rus_Cyrl",
    "Bengali":"ben_Beng"
}

# System-generated ID: stable during this session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
st.caption("System-generated Thread ID")
st.code(st.session_state.thread_id)

@st.cache_resource
def get_history_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection("translation_history")

@st.cache_resource
def load_model():
    name = "facebook/nllb-200-distilled-600M"
    try:
        token = st.secrets.get("HF_TOKEN", "")
    except Exception:
        token = ""
    token = token or os.getenv("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(name, token=token or None)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name, token=token or None
    )
    model.eval()
    return tok, model

def read_file(f):
    name = f.name.lower()
    if name.endswith(".txt"):
        return f.getvalue().decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(f).paragraphs)
    if name.endswith(".json"):
        return json.dumps(json.load(f), ensure_ascii=False, indent=2)
    raise ValueError("Unsupported file type.")

def translate_text(text, src, dst, tok, model, bar):
    tok.src_lang = LANG[src]
    words = text.split()
    chunks = [" ".join(words[i:i+800]) for i in range(0, len(words), 800)]
    output = []
    if not chunks:
        return ""
    for i, chunk in enumerate(chunks):
        inputs = tok(chunk, return_tensors="pt", truncation=True,
                     max_length=512)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids(LANG[dst]),
                max_new_tokens=256
            )
        output.append(tok.batch_decode(ids, skip_special_tokens=True)[0])
        bar.progress((i + 1) / len(chunks),
                     text=f"Translating chunk {i+1}/{len(chunks)}")
    return "\n".join(output)

def make_docx(text):
    b = io.BytesIO()
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(b)
    return b.getvalue()

def make_pdf(text):
    b = io.BytesIO()
    doc = SimpleDocTemplate(b, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    for line in text.splitlines():
        safe = (line.replace("&", "&amp;")
                    .replace("<", "&lt;").replace(">", "&gt;"))
        story += [Paragraph(safe or " ", styles["Normal"]), Spacer(1, 5)]
    doc.build(story)
    return b.getvalue()

st.subheader("Translation Settings")
prompt = st.text_area(
    "Input Prompt Template (edit manually)",
    value="Translate accurately from {source_language} to {target_language}. "
          "Preserve the original meaning, names, numbers and paragraph order.",
    height=90
)
uploaded = st.file_uploader(
    "Upload input file", type=["pdf", "docx", "txt", "json"]
)
c1, c2 = st.columns(2)
src = c1.selectbox("Source language", list(LANG), index=0)
dst = c2.selectbox("Target language", list(LANG), index=1)

if st.button("Translate Document", type="primary", use_container_width=True):
    if not uploaded:
        st.warning("Please upload a PDF, DOCX, TXT or JSON file.")
    elif src == dst:
        st.warning("Select different source and target languages.")
    else:
        try:
            source = read_file(uploaded)
            if not source.strip():
                st.error("No readable text found in this file.")
            else:
                with st.spinner("Loading translation model..."):
                    tok, model = load_model()
                bar = st.progress(0, text="Starting translation...")
                translated = translate_text(source, src, dst, tok, model, bar)

                if translated.strip():
                    timestamp = datetime.now().isoformat(timespec="seconds")
                    record = {
                        "thread_id": st.session_state.thread_id,
                        "timestamp": timestamp,
                        "filename": uploaded.name,
                        "source_language": src,
                        "target_language": dst,
                        "prompt_template": prompt,
                        "source_text": source,
                        "translation": translated
                    }
                    # Save every translation to ChromaDB
                    db = get_history_db()
                    db.add(
                        ids=[str(uuid.uuid4())],
                        documents=[json.dumps(record, ensure_ascii=False)],
                        metadatas=[{
                            "thread_id": record["thread_id"],
                            "timestamp": timestamp,
                            "filename": uploaded.name,
                            "source_language": src,
                            "target_language": dst
                        }]
                    )
                    st.session_state.last_result = record
                    st.success("Translation completed and history saved!")
                else:
                    st.warning("No translated text was produced.")
        except Exception as e:
            st.error(f"Error: {e}")

# Downloads
if st.session_state.get("last_result"):
    r = st.session_state.last_result
    result = r["translation"]
    st.subheader("Translated Output")
    st.text_area("Result", result, height=220)
    st.download_button("Download TXT", result, "translation.txt",
                       "text/plain", use_container_width=True)
    st.download_button("Download PDF", make_pdf(result), "translation.pdf",
                       "application/pdf", use_container_width=True)
    st.download_button(
        "Download DOCX", make_docx(result), "translation.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True
    )
    json_output = json.dumps(r, ensure_ascii=False, indent=2)
    st.download_button("Download JSON", json_output, "translation.json",
                       "application/json", use_container_width=True)

# Read saved history from ChromaDB
st.divider()
st.subheader("Past Translation History")
try:
    db = get_history_db()
    records = db.get(include=["documents", "metadatas"])
    entries = []
    for doc in records.get("documents") or []:
        try:
            entries.append(json.loads(doc))
        except (ValueError, TypeError):
            pass
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    if entries:
        thread_ids = list(dict.fromkeys(x.get("thread_id", "") for x in entries))
        selected = st.selectbox("Filter by Thread ID", ["All threads"] + thread_ids)
        shown = entries if selected == "All threads" else [
            x for x in entries if x.get("thread_id") == selected
        ]
        for i, item in enumerate(shown):
            title = (f'{item.get("timestamp", "")} | '
                     f'{item.get("filename", "")}')
            with st.expander(title):
                st.caption("Thread ID: " + item.get("thread_id", ""))
                st.write(
                    item.get("source_language", ""), "→",
                    item.get("target_language", "")
                )
                st.text_area("Saved translation", item.get("translation", ""),
                             height=120, key=f"history_{i}_{item.get('thread_id','')}")
                st.download_button(
                    "Download this record as JSON",
                    json.dumps(item, ensure_ascii=False, indent=2),
                    f"history_{i}.json", "application/json",
                    key=f"download_{i}_{item.get('thread_id','')}"
                )
    else:
        st.info("No history records yet. Translate a document to save one.")
except Exception as e:
    st.warning(f"Could not load history: {e}")
