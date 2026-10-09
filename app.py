
import io, os, re, json, uuid
from datetime import datetime
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="AIT Global Technologies Translator", layout="centered")
st.title("AIT GLOBAL TECHNOLOGIES — TRANSLATOR")

LANG = {"English":"eng_Latn","Tamil":"tam_Taml","Hindi":"hin_Deva","Telugu":"tel_Telu","Kannada":"kan_Knda","Malayalam":"mal_Mlym","French":"fra_Latn","Spanish":"spa_Latn","German":"deu_Latn","Chinese":"zho_Hans","Japanese":"jpn_Jpan","Arabic":"arb_Arab","Portuguese":"por_Latn","Russian":"rus_Cyrl","Bengali":"ben_Beng"}

@st.cache_resource
def load_model():
    name = "facebook/nllb-200-distilled-600M"
    token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")
    tok = AutoTokenizer.from_pretrained(name, token=token or None)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, token=token or None)
    return tok, model

def extract(f):
    n = f.name.lower()
    if n.endswith(".txt"): return f.getvalue().decode("utf-8", errors="replace")
    if n.endswith(".pdf"): return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
    if n.endswith(".docx"): return "\n".join(p.text for p in Document(f).paragraphs)
    if n.endswith(".json"): return json.dumps(json.load(f), ensure_ascii=False, indent=2)
    raise ValueError("Unsupported file format.")

def translate(text, src, dst, prompt, tok, model, bar):
    tok.src_lang = LANG[src]
    chunks = [x.strip() for x in re.split(r"\n+", text) if x.strip()]
    result = []
    if not chunks: return ""
    for i, chunk in enumerate(chunks):
        words = chunk.split()
        for j in range(0, len(words), 180):
            part = " ".join(words[j:j+180])
            instruction = (prompt.strip() + "\n\n") if prompt.strip() else ""
            inputs = tok(instruction + part, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                ids = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids(LANG[dst]), max_new_tokens=256)
            result.append(tok.batch_decode(ids, skip_special_tokens=True)[0])
        bar.progress((i+1)/len(chunks), text=f"Translating section {i+1}/{len(chunks)}")
    return "\n".join(result)

def make_docx(text):
    b = io.BytesIO()
    d = Document()
    for line in text.splitlines(): d.add_paragraph(line)
    d.save(b)
    return b.getvalue()

def make_pdf(text):
    b = io.BytesIO()
    doc = SimpleDocTemplate(b, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;") or " ", styles["Normal"]) for line in text.splitlines()]
    doc.build(story)
    return b.getvalue()

# Persistent history file: works on persistent local storage.
# For Streamlit Cloud, use a database if history must survive app restarts.
HISTORY_FILE = "translation_history.json"

def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except (OSError, json.JSONDecodeError): return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())[:8]
if "history" not in st.session_state: st.session_state.history = load_history()

st.subheader("Translation Settings")
thread_id = st.text_input("Thread ID", value=st.session_state.thread_id)
prompt = st.text_area("Prompt Template", value="Translate accurately into the target language. Preserve meaning, headings, names, numbers, and formatting.")
uploaded = st.file_uploader("Upload source file", type=["pdf", "txt", "docx", "json"])
c1, c2 = st.columns(2)
src = c1.selectbox("Source language", list(LANG))
dst = c2.selectbox("Target language", list(LANG), index=1)

if st.button("Translate Document", use_container_width=True):
    if not uploaded: st.warning("Please upload a file.")
    elif src == dst: st.warning("Choose different source and target languages.")
    elif not thread_id.strip(): st.warning("Enter a thread ID.")
    else:
        try:
            source_text = extract(uploaded)
            if not source_text.strip(): st.error("No readable text found.")
            else:
                with st.spinner("Loading translation model..."): tok, model = load_model()
                bar = st.progress(0, text="Starting translation...")
                result = translate(source_text, src, dst, prompt, tok, model, bar)
                if result:
                    record = {"thread_id":thread_id.strip(), "timestamp":datetime.now().isoformat(timespec="seconds"), "filename":uploaded.name, "source_language":src, "target_language":dst, "prompt":prompt, "source_text":source_text, "translation":result}
                    history = load_history()
                    history.append(record)
                    save_history(history)
                    st.session_state.history = history
                    st.session_state.translated = result
                    st.session_state.current_thread = thread_id.strip()
                    st.success("Translation completed and history saved.")
                else: st.warning("No text found to translate.")
        except Exception as e: st.error(f"Translation failed: {e}")

if st.session_state.get("translated"):
    result = st.session_state.translated
    st.subheader("Translated Output")
    st.text_area("Result", result, height=220)
    st.download_button("Download TXT", result, "translation.txt", "text/plain")
    st.download_button("Download PDF", make_pdf(result), "translation.pdf", "application/pdf")
    st.download_button("Download DOCX", make_docx(result), "translation.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    st.download_button("Download JSON", json.dumps({"thread_id":st.session_state.get("current_thread",""),"translation":result}, ensure_ascii=False, indent=2), "translation.json", "application/json")

st.subheader("Translation History")
history = load_history()
st.session_state.history = history
if history:
    ids = list(dict.fromkeys(r.get("thread_id", "") for r in history))
    selected = st.selectbox("Filter by Thread ID", ["All"] + ids)
    shown = history if selected == "All" else [r for r in history if r.get("thread_id") == selected]
    for i, record in enumerate(reversed(shown)):
        with st.expander(f'{record.get("timestamp","")} | {record.get("filename","")} | Thread: {record.get("thread_id","")}'):
            st.write("Languages:", record.get("source_language"), "→", record.get("target_language"))
            st.write("Prompt:", record.get("prompt", ""))
            st.text_area("Saved translation", record.get("translation", ""), height=150, key=f"hist_{i}_{record.get('thread_id','')}")
    st.download_button("Download History JSON", json.dumps(history, ensure_ascii=False, indent=2), "translation_history.json", "application/json")
else:
    st.info("No saved translation history yet.")
