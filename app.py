
import io, uuid, re, time, streamlit as st, torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES", layout="wide")
st.title("🌐 AIT GLOBAL TECHNOLOGIES — Translator")

MODEL = "facebook/nllb-200-distilled-600M"
CHUNK_SIZE = 1200
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cpu":
    torch.set_num_threads( max(1, min(4, torch.get_num_threads())) )

LANGS = {
    "English":"eng_Latn", "Spanish":"spa_Latn",
    "French":"fra_Latn", "German":"deu_Latn",
    "Hindi":"hin_Deva", "Tamil":"tam_Taml",
    "Telugu":"tel_Telu", "Kannada":"kan_Knda",
    "Malayalam":"mal_Mlym", "Arabic":"arb_Arab",
    "Chinese":"zho_Hans", "Japanese":"jpn_Jpan",
    "Portuguese":"por_Latn"
}

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.to(DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model.half()
    return tok, model

def split_chunks(text, size=CHUNK_SIZE):
    # Split into 1200-character chunks, preferably at paragraph/sentence boundaries
    text = text.strip()
    chunks = []
    while len(text) > size:
        cut = max(
            text.rfind("\n", 0, size),
            text.rfind(". ", 0, size),
            text.rfind("? ", 0, size),
            text.rfind("! ", 0, size),
            text.rfind(" ", 0, size)
        )
        if cut < size // 2:
            cut = size
        else:
            cut += 1
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks

c1, c2 = st.columns(2)
src = c1.selectbox("Translate from", list(LANGS))
dst = c2.selectbox("Translate to", list(LANGS), index=1)

template = st.text_area(
    "Manual prompt template",
    "Translate the following text from {source_language} "
    "to {target_language}:\n\n{text}",
    height=80
)
text = st.text_area("Paste input text", height=150)

file = st.file_uploader(
    "Or upload TXT / PDF / DOCX",
    type=["txt", "pdf", "docx"]
)

if file:
    try:
        if file.name.lower().endswith(".txt"):
            text = file.getvalue().decode("utf-8-sig", "replace")
        elif file.name.lower().endswith(".docx"):
            text = "\n".join(p.text for p in Document(file).paragraphs)
        else:
            from pypdf import PdfReader
            text = "\n".join(
                p.extract_text() or "" for p in PdfReader(file).pages
            )
        st.info(f"Loaded: {file.name} ({len(text)} characters)")
    except Exception as e:
        st.error(f"File error: {e}")

if "tid" not in st.session_state:
    st.session_state.tid = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

st.caption(f"Thread ID: {st.session_state.tid} | Device: {DEVICE}")

if st.button("🆕 New thread"):
    st.session_state.tid = str(uuid.uuid4())
    st.rerun()

if st.button(
    "🚀 Translate and prepare PDF + DOCX",
    type="primary",
    use_container_width=True
):
    if not text.strip():
        st.warning("Enter text or upload a document.")
    elif src == dst:
        st.warning("Choose different languages.")
    elif "{text}" not in template:
        st.warning("Template must contain {text}.")
    else:
        try:
            tok, model = load_model()
            tok.src_lang = LANGS[src]
            chunks = split_chunks(text)
            total = len(chunks)
            output = []
            bar = st.progress(0)
            status = st.empty()
            live = st.empty()
            started = time.time()

            status.info(f"Starting translation: {total} chunks")

            # Process small batches to reduce model-call overhead
            for start in range(0, total, BATCH_SIZE):
                batch = chunks[start:start + BATCH_SIZE]

                inputs = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(DEVICE)

                with torch.inference_mode():
                    ids = model.generate(
                        **inputs,
                        forced_bos_token_id=tok.convert_tokens_to_ids(
                            LANGS[dst]
                        ),
                        max_new_tokens=512,
                        num_beams=1,
                        do_sample=False,
                        use_cache=True
                    )

                decoded = tok.batch_decode(
                    ids, skip_special_tokens=True
                )
                output.extend(decoded)

                done = min(start + len(batch), total)
                elapsed = time.time() - started
                bar.progress(
                    done / total,
                    text=f"Translated {done}/{total} chunks"
                )
                status.info(
                    f"Progress: {done}/{total} | "
                    f"Elapsed: {elapsed:.1f}s"
                )
                live.text_area(
                    "Live translation output",
                    "\n\n".join(output),
                    height=250,
                    key=f"live_{st.session_state.tid}_{done}"
                )

            result = "\n\n".join(output)
            st.session_state.output = result
            st.session_state.meta = {
                "from": src,
                "to": dst,
                "thread": st.session_state.tid,
                "time": datetime.now().isoformat(timespec="seconds")
            }
            st.session_state.history.insert(0, {
                **st.session_state.meta,
                "chunks": total,
                "characters": len(text)
            })
            st.success(
                f"Translation completed in {time.time()-started:.1f}s!"
            )

        except Exception as e:
            st.error(f"Translation failed: {e}")

if st.session_state.get("output"):
    result = st.session_state.output
    meta = st.session_state.get("meta", {})
    src_out = meta.get("from", src)
    dst_out = meta.get("to", dst)
    tid_out = meta.get("thread", st.session_state.tid)
    time_out = meta.get("time", "")

    pdf = io.BytesIO()
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AIT GLOBAL TECHNOLOGIES", styles["Title"]),
        Paragraph(
            f"{escape(src_out)} to {escape(dst_out)}",
            styles["Normal"]
        ),
        Paragraph(f"Thread: {escape(tid_out)}", styles["Normal"]),
        Paragraph(f"Date: {escape(time_out)}", styles["Normal"]),
        Spacer(1, 12)
    ]

    for paragraph in result.split("\n\n"):
        if paragraph.strip():
            safe = escape(paragraph).replace("\n", "<br/>")
            story.extend([
                Paragraph(safe, styles["BodyText"]),
                Spacer(1, 6)
            ])

    SimpleDocTemplate(pdf, pagesize=A4).build(story)

    doc = Document()
    doc.add_heading("AIT GLOBAL TECHNOLOGIES", 0)
    doc.add_paragraph(f"{src_out} to {dst_out}")
    doc.add_paragraph(f"Thread: {tid_out}")
    doc.add_paragraph(f"Date: {time_out}")
    for paragraph in result.split("\n\n"):
        if paragraph.strip():
            doc.add_paragraph(paragraph)

    word = io.BytesIO()
    doc.save(word)

    st.text_area("Final translation", result, height=250)
    a, b = st.columns(2)
    a.download_button(
        "⬇ Download PDF", pdf.getvalue(),
        "translation.pdf", "application/pdf",
        use_container_width=True
    )
    b.download_button(
        "⬇ Download DOCX", word.getvalue(),
        "translation.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True
    )

with st.expander("🕘 Translation history"):
    st.write(st.session_state.history)

