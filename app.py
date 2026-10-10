
import io, os, re, json, uuid, requests, html
from datetime import datetime
import streamlit as st
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

st.set_page_config(page_title="AIT Translator", page_icon="🌐", layout="centered")
st.markdown("""
<style>
.stApp{background:#f3f7fc}
.main .block-container{max-width:900px;padding-top:1.5rem}
.hero{background:#123d78;padding:20px;border-radius:13px;text-align:center}
.hero h2{color:#ffd43b;margin:0;font-size:27px}
.hero p{color:white;margin:7px 0 0}
.stButton>button,.stDownloadButton>button{
background:#164e91;color:white;border-radius:9px;border:0}
</style>
<div class="hero"><h2>AIT GLOBAL TECHNOLOGIES</h2>
<p>TRANSLATOR</p></div>
""", unsafe_allow_html=True)

LANG = {
    "English":"eng_Latn", "Tamil":"tam_Taml", "Hindi":"hin_Deva",
    "Telugu":"tel_Telu", "Kannada":"kan_Knda",
    "Malayalam":"mal_Mlym", "French":"fra_Latn",
    "Spanish":"spa_Latn", "German":"deu_Latn",
    "Chinese":"zho_Hans", "Japanese":"jpn_Jpan",
    "Arabic":"arb_Arab", "Russian":"rus_Cyrl", "Bengali":"ben_Beng"
}

FONT_INFO = {
    "Tamil": ("notosanstamil", "NotoSansTamil[wdth,wght].ttf"),
    "Hindi": ("notosansdevanagari", "NotoSansDevanagari[wdth,wght].ttf"),
    "Telugu": ("notosanstelugu", "NotoSansTelugu[wdth,wght].ttf"),
    "Kannada": ("notosanskannada", "NotoSansKannada[wdth,wght].ttf"),
    "Malayalam": ("notosansmalayalam", "NotoSansMalayalam[wdth,wght].ttf"),
    "Bengali": ("notosansbengali", "NotoSansBengali[wdth,wght].ttf"),
    "Arabic": ("notosansarabic", "NotoSansArabic[wdth,wght].ttf"),
    "Chinese": ("notosanssc", "NotoSansSC[wght].ttf"),
    "Japanese": ("notosansjp", "NotoSansJP[wght].ttf"),
    "Korean": ("notosanskr", "NotoSansKR[wght].ttf"),
}

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
    token = os.getenv("HF_TOKEN", "")
    try:
        token = st.secrets.get("HF_TOKEN", "") or token
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(name, token=token or None)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, token=token or None)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tok, model

def read_file(f):
    name = f.name.lower()
    if name.endswith(".txt"):
        return f.getvalue().decode("utf-8-sig", errors="replace")
    if name.endswith(".pdf"):
        reader = PdfReader(f)
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    if name.endswith(".docx"):
        return "\n\n".join(p.text for p in Document(f).paragraphs)
    if name.endswith(".json"):
        return json.dumps(json.loads(f.getvalue().decode("utf-8-sig")),
                          ensure_ascii=False, indent=2)
    raise ValueError("Unsupported file type.")

def split_chunks(text, tok, max_tokens=300):
    # Split by sentences/paragraphs and enforce the model input limit.
    units = re.split(r"(?<=[.!?।])\s+|\n+", text)
    chunks, buf = [], ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        candidate = (buf + "\n" + unit).strip()
        if len(tok(candidate, add_special_tokens=False)["input_ids"]) <= max_tokens:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        ids = tok(unit, add_special_tokens=False)["input_ids"]
        for start in range(0, len(ids), max_tokens):
            part = tok.decode(ids[start:start + max_tokens],
                              skip_special_tokens=True)
            if part.strip():
                chunks.append(part)
    if buf:
        chunks.append(buf)
    return chunks

def translate_text(text, src, dst, tok, model, bar):
    tok.src_lang = LANG[src]
    chunks = split_chunks(text, tok)
    if not chunks:
        return ""
    results = []
    device = next(model.parameters()).device
    target_id = tok.convert_tokens_to_ids(LANG[dst])
    for i, chunk in enumerate(chunks):
        inputs = tok(chunk, return_tensors="pt",
                     truncation=True, max_length=384)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=192,
                num_beams=1,
                do_sample=False
            )
        results.append(tok.batch_decode(ids, skip_special_tokens=True)[0])
        bar.progress((i + 1) / len(chunks),
                     text=f"Translating {i+1}/{len(chunks)} chunks")
    return "\n\n".join(results)

@st.cache_data(show_spinner=False)
def get_font_path(language):
    if language not in FONT_INFO:
        return None
    folder, filename = FONT_INFO[language]
    path = f"/tmp/{filename}"
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    url = f"https://raw.githubusercontent.com/google/fonts/main/ofl/{folder}/{filename}"
    try:
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        if len(r.content) < 1000:
            return None
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception:
        return None

def make_pdf(text, language):
    b = io.BytesIO()
    styles = getSampleStyleSheet()
    font_path = get_font_path(language)
    font_name = "Helvetica"
    if font_path:
        try:
            try:
                pdfmetrics.registerFont(TTFont("TargetFont", font_path,
                                               shapable=True))
            except TypeError:
                pdfmetrics.registerFont(TTFont("TargetFont", font_path))
            font_name = "TargetFont"
        except Exception:
            pass
    style = ParagraphStyle(
        "Translated", parent=styles["Normal"], fontName=font_name,
        fontSize=10, leading=15, spaceAfter=7, wordWrap="CJK"
    )
    story = []
    for line in text.splitlines():
        story.append(Paragraph(html.escape(line) or " ", style))
        story.append(Spacer(1, 3))
    SimpleDocTemplate(b, pagesize=A4).build(story)
    return b.getvalue(), font_name

def make_docx(text, language):
    b = io.BytesIO()
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Noto Sans" if language == "English" else "Noto Sans"
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(b)
    return b.getvalue()

st.subheader("Translation Settings")
prompt = st.text_area(
    "Input Prompt Template",
    "Translate accurately from {source_language} to {target_language}. "
    "Preserve names, numbers, meaning and paragraph order.", height=80
)
uploaded = st.file_uploader("Upload PDF, DOCX, TXT or JSON",
                            type=["pdf", "docx", "txt", "json"])
c1, c2 = st.columns(2)
src = c1.selectbox("Source language", list(LANG), index=0)
dst = c2.selectbox("Target language", list(LANG), index=1)

if st.button("Translate Document", type="primary", use_container_width=True):
    if not uploaded:
        st.warning("Please upload a document.")
    elif src == dst:
        st.warning("Select different source and target languages.")
    else:
        try:
            source = read_file(uploaded)
            if not source.strip():
                st.error("No readable text found. Scanned PDFs need OCR.")
            else:
                with st.spinner("Loading model (first load may take time)..."):
                    tok, model = load_model()
                bar = st.progress(0, text="Preparing translation...")
                translated = translate_text(source, src, dst, tok, model, bar)
                if not translated.strip():
                    st.error("No translated text was produced.")
                else:
                    timestamp = datetime.now().isoformat(timespec="seconds")
                    record = {
                        "thread_id": st.session_state.thread_id,
                        "timestamp": timestamp, "filename": uploaded.name,
                        "source_language": src, "target_language": dst,
                        "prompt_template": prompt, "source_text": source,
                        "translation": translated
                    }
                    try:
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
                    except Exception as e:
                        st.warning(f"Translation succeeded; history save failed: {e}")
                    st.session_state.last_result = record
                    st.success("Translation completed!")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.exception(e)

if st.session_state.get("last_result"):
    r = st.session_state.last_result
    result = r["translation"]
    st.subheader("Translated Output")
    st.text_area("Result", result, height=220)
    st.download_button("Download TXT", result.encode("utf-8"),
                       "translation.txt", "text/plain", use_container_width=True)
    pdf, font_used = make_pdf(result, r["target_language"])
    st.download_button("Download PDF", pdf, "translation.pdf",
                       "application/pdf", use_container_width=True)
    if font_used == "Helvetica" and r["target_language"] != "English":
        st.warning("A Unicode font could not be loaded for the PDF. "
                   "The PDF may not display this script correctly.")
    st.download_button("Download DOCX",
                       make_docx(result, r["target_language"]),
                       "translation.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       use_container_width=True)
    st.download_button("Download JSON",
                       json.dumps(r, ensure_ascii=False, indent=2).encode("utf-8"),
                       "translation.json", "application/json",
                       use_container_width=True)

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
        ids = list(dict.fromkeys(x.get("thread_id", "") for x in entries))
        selected = st.selectbox("Filter by Thread ID", ["All threads"] + ids)
        shown = entries if selected == "All threads" else [
            x for x in entries if x.get("thread_id") == selected]
        for i, item in enumerate(shown):
            with st.expander(f'{item.get("timestamp","")} | {item.get("filename","")}'):
                st.caption("Thread ID: " + item.get("thread_id", ""))
                st.write(item.get("source_language", ""), "→",
                         item.get("target_language", ""))
                st.text_area("Saved translation", item.get("translation", ""),
                             height=120, key=f"hist_{i}_{item.get('thread_id','')}")
                st.download_button("Download record JSON",
                    json.dumps(item, ensure_ascii=False, indent=2),
                    f"history_{i}.json", "application/json",
                    key=f"dl_{i}_{item.get('thread_id','')}")
    else:
        st.info("No history records yet.")
except Exception as e:
    st.warning(f"History unavailable: {e}")

