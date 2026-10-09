
import io, re, json, uuid, html
from datetime import datetime
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

st.set_page_config(page_title="AIT GLOBAL Translator",
                   page_icon="🌍", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(140deg,#101010,#202020);
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.brand {
    text-align:center;
    font-size:13px;
    font-weight:800;
    letter-spacing:5px;
    color:#F5C542;
    margin-bottom:12px;
}
.hero {
    text-align:center;
    font-size:34px;
    font-weight:850;
    color:white;
    padding:12px 8px 5px 8px;
}
.hero span { color:#F5C542; }
.subtitle {
    text-align:center;
    color:#cccccc;
    font-size:14px;
    margin-bottom:26px;
}
div[data-testid="stFileUploader"] {
    border:1px dashed #F5C542;
    border-radius:14px;
    padding:12px;
    background:#181818;
}
div[data-testid="stTextArea"] textarea {
    background:#191919;
    color:white;
    border:1px solid #555;
    border-radius:10px;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background:#191919;
    border-radius:9px;
}
.stButton > button {
    background:#F5C542;
    color:#111111;
    font-weight:800;
    border:0;
    border-radius:10px;
    min-height:44px;
    transition:0.2s;
}
.stButton > button:hover {
    background:#ffffff;
    color:#111111;
    border:1px solid #F5C542;
}
.stDownloadButton > button {
    background:#202020;
    color:#F5C542;
    border:1px solid #F5C542;
    border-radius:9px;
    font-weight:700;
    width:100%;
}
.stDownloadButton > button:hover {
    background:#F5C542;
    color:#111111;
}
div[data-testid="stProgress"] > div > div > div {
    background:#F5C542;
}
div[data-testid="stExpander"] {
    border:1px solid #555;
    border-radius:10px;
}
hr { border-color:#444; }
</style>

<div class="brand">AIT GLOBAL</div>
<div class="hero">🌍 DOCUMENT <span>TRANSLATOR</span></div>
<div class="subtitle">
Professional multilingual document translation powered by NLLB-200
</div>
""", unsafe_allow_html=True)

MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {
    "English":"eng_Latn", "Tamil":"tam_Taml",
    "Hindi":"hin_Deva", "Telugu":"tel_Telu",
    "Malayalam":"mal_Mlym", "Kannada":"kan_Knda",
    "French":"fra_Latn", "German":"deu_Latn",
    "Spanish":"spa_Latn", "Arabic":"arb_Arab",
    "Chinese":"zho_Hans", "Japanese":"jpn_Jpan",
    "Portuguese":"por_Latn", "Russian":"rus_Cyrl",
    "Bengali":"ben_Beng", "Urdu":"urd_Arab"
}

if "thread" not in st.session_state:
    st.session_state.thread = str(uuid.uuid4())
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource(show_spinner="Loading NLLB translation model...")
def load_model():
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.eval()
    return tokenizer, model

def read_file(upload):
    name = upload.name.lower()
    raw = upload.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        if not text.strip():
            raise ValueError(
                "No selectable text found. Scanned PDFs require OCR."
            )
        return "text", text

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(raw))
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(c.text for c in row.cells))
        return "text", "\n".join(lines)

    raw_text = raw.decode("utf-8-sig")
    if name.endswith(".json"):
        return "json", json.loads(raw_text)
    return "text", raw_text

def split_chunks(text, size=300):
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    result, current = [], ""
    for sentence in sentences:
        while len(sentence) > size:
            if current:
                result.append(current)
                current = ""
            result.append(sentence[:size])
            sentence = sentence[size:]
        if current and len(current) + len(sentence) + 1 > size:
            result.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        result.append(current)
    return result

def pdf_bytes(text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=2*cm,
        rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    story = []
    for line in text.splitlines():
        if line.strip():
            story.append(Paragraph(html.escape(line), styles["Normal"]))
            story.append(Spacer(1, 6))
    doc.build(story or [Paragraph("No translated text", styles["Normal"])])
    return buffer.getvalue()

def docx_bytes(text):
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

def translate_text(text, tokenizer, model, target_code,
                   progress, status, live, offset=0, total=None):
    pieces = split_chunks(text)
    output = []
    total = total or len(pieces)

    for i, piece in enumerate(pieces):
        status.info(f"Translating chunk {offset+i+1} of {total}...")
        inputs = tokenizer(
            piece, return_tensors="pt",
            truncation=True, max_length=512
        )
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                    target_code
                ),
                max_new_tokens=256, num_beams=2,
                do_sample=False
            )
        output.append(
            tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        )
        progress.progress(min((offset+i+1)/total, 1.0))
        live.text_area(
            "Live translated chunks",
            "\n\n".join(output[-8:]),
            height=180,
            key=f"live_{st.session_state.thread}_{offset+i}"
        )
    return "\n\n".join(output)

def translate_json(value, tokenizer, model, target_code,
                   progress, status, live, counter, total):
    if isinstance(value, str):
        pieces = split_chunks(value)
        if not pieces:
            return value
        return translate_text(
            value, tokenizer, model, target_code,
            progress, status, live, counter[0], total
        ) if not counter.__setitem__(0, counter[0]+len(pieces)) else value
    if isinstance(value, list):
        return [translate_json(v, tokenizer, model, target_code,
                               progress, status, live, counter, total)
                for v in value]
    if isinstance(value, dict):
        return {k: translate_json(v, tokenizer, model, target_code,
                                  progress, status, live, counter, total)
                for k, v in value.items()}
    return value

left, right = st.columns(2)
with left:
    source = st.selectbox("📥 Source language", list(LANGS))
with right:
    target = st.selectbox("📤 Target language", list(LANGS), index=1)

st.markdown("### 📂 Upload your document")
upload = st.file_uploader(
    "PDF, DOCX, TXT or JSON",
    type=["pdf", "docx", "txt", "json"]
)
manual = st.text_area("✍️ Or enter text directly", height=120)

st.caption(f"Session ID: {st.session_state.thread}")

a, b = st.columns([1, 1])
with a:
    start = st.button("🚀 Start Translation",
                      type="primary", use_container_width=True)
with b:
    fresh = st.button("＋ New Session", use_container_width=True)

if fresh:
    st.session_state.thread = str(uuid.uuid4())
    st.session_state.result = None
    st.rerun()

if start:
    try:
        if source == target:
            st.warning("Select two different languages.")
            st.stop()

        if upload:
            kind, data = read_file(upload)
        elif manual.strip():
            kind, data = "text", manual.strip()
        else:
            st.warning("Please upload a file or enter text.")
            st.stop()

        tokenizer, model = load_model()
        tokenizer.src_lang = LANGS[source]
        progress = st.progress(0)
        status = st.empty()
        live = st.empty()

        if kind == "text":
            translated = translate_text(
                data, tokenizer, model, LANGS[target],
                progress, status, live
            )
            json_result = None
        else:
            def count_strings(v):
                if isinstance(v, str):
                    return max(1, len(split_chunks(v)))
                if isinstance(v, list):
                    return sum(count_strings(x) for x in v)
                if isinstance(v, dict):
                    return sum(count_strings(x) for x in v.values())
                return 0

            total = count_strings(data)
            counter = [0]

            def walk(v):
                if isinstance(v, str):
                    pieces = split_chunks(v)
                    if not pieces:
                        return v
                    out = []
                    for piece in pieces:
                        counter[0] += 1
                        status.info(
                            f"Translating JSON text {counter[0]}/{total}..."
                        )
                        inputs = tokenizer(
                            piece, return_tensors="pt",
                            truncation=True, max_length=512
                        )
                        with torch.inference_mode():
                            ids = model.generate(
                                **inputs,
                                forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                                    LANGS[target]
                                ),
                                max_new_tokens=256, num_beams=2,
                                do_sample=False
                            )
                        out.append(tokenizer.batch_decode(
                            ids, skip_special_tokens=True
                        )[0])
                        progress.progress(counter[0]/max(total, 1))
                        live.text_area(
                            "Live translated chunks",
                            "\n\n".join(out[-8:]),
                            height=180,
                            key=f"json_{st.session_state.thread}_{counter[0]}"
                        )
                    return " ".join(out)
                if isinstance(v, list):
                    return [walk(x) for x in v]
                if isinstance(v, dict):
                    return {k: walk(x) for k, x in v.items()}
                return v

            json_result = walk(data)
            translated = json.dumps(
                json_result, ensure_ascii=False, indent=2
            )

        st.session_state.result = {
            "text": translated, "json": json_result,
            "source": source, "target": target
        }
        st.session_state.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source, "target": target,
            "thread": st.session_state.thread
        })
        status.success("✅ Translation completed successfully!")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

if st.session_state.result:
    result = st.session_state.result
    st.markdown("---")
    st.markdown("### ✨ Translated Document")
    st.text_area("Final translation", result["text"], height=220)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇ Download PDF", pdf_bytes(result["text"]),
            "translated.pdf", "application/pdf",
            use_container_width=True
        )
        st.download_button(
            "⬇ Download TXT", result["text"],
            "translated.txt", "text/plain",
            use_container_width=True
        )
    with d2:
        st.download_button(
            "⬇ Download DOCX", docx_bytes(result["text"]),
            "translated.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        json_data = (
            result["json"] if result["json"] is not None else
            {"source_language": result["source"],
             "target_language": result["target"],
             "translation": result["text"]}
        )
        st.download_button(
            "⬇ Download JSON",
            json.dumps(json_data, ensure_ascii=False, indent=2),
            "translated.json", "application/json",
            use_container_width=True
        )

with st.expander("🕘 Translation History"):
    for item in reversed(st.session_state.history):
        st.write(
            f"{item['time']} | {item['source']} → "
            f"{item['target']} | {item['thread']}"
        )
