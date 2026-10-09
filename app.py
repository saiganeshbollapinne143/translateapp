
import io, uuid, time, re, streamlit as st, torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from xml.sax.saxutils import escape

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES", layout="wide")
st.title("🌐 AIT GLOBAL TECHNOLOGIES — Translator")

# Initialize ALL session state before accessing it
if "translated" not in st.session_state:
    st.session_state.translated = False
if "output" not in st.session_state:
    st.session_state.output = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "tid" not in st.session_state:
    st.session_state.tid = str(uuid.uuid4())
if "meta" not in st.session_state:
    st.session_state.meta = {}

MODEL = "facebook/nllb-200-distilled-600M"
CHUNK_SIZE = 1200
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cpu":
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

LANGS = {
    "English": "eng_Latn", "Spanish": "spa_Latn",
    "French": "fra_Latn", "German": "deu_Latn",
    "Hindi": "hin_Deva", "Tamil": "tam_Taml",
    "Telugu": "tel_Telu", "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym", "Arabic": "arb_Arab",
    "Chinese": "zho_Hans", "Japanese": "jpn_Jpan",
    "Portuguese": "por_Latn"
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def split_chunks(text, size=CHUNK_SIZE):
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
    height=90
)

text_input = st.text_area("Enter text to translate", height=160)
uploaded = st.file_uploader(
    "Or upload TXT / PDF / DOCX",
    type=["txt", "pdf", "docx"]
)

input_text = text_input

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".txt"):
            input_text = uploaded.getvalue().decode("utf-8-sig", "replace")
        elif uploaded.name.lower().endswith(".docx"):
            input_text = "\n".join(
                p.text for p in Document(uploaded).paragraphs
            )
        else:
            from pypdf import PdfReader
            reader = PdfReader(uploaded)
            input_text = "\n\n".join(
                p.extract_text() or "" for p in reader.pages
            )
        st.info(f"Loaded {uploaded.name}: {len(input_text)} characters")
    except Exception as e:
        st.error(f"File reading error: {e}")

st.caption(f"Thread ID: {st.session_state.tid} | Device: {DEVICE}")

if st.button("🆕 New thread"):
    st.session_state.tid = str(uuid.uuid4())
    st.session_state.output = ""
    st.session_state.translated = False
    st.session_state.meta = {}
    st.rerun()

if st.button(
    "🚀 Translate and prepare PDF + DOCX",
    type="primary",
    use_container_width=True
):
    if not input_text.strip():
        st.warning("Enter text or upload a document.")
    elif src == dst:
        st.warning("Choose different languages.")
    elif "{text}" not in template:
        st.warning("The prompt template must contain {text}.")
    else:
        try:
            tokenizer, model = load_model()
            tokenizer.src_lang = LANGS[src]
            target_id = tokenizer.convert_tokens_to_ids(LANGS[dst])
            chunks = split_chunks(input_text)
            total = len(chunks)
            translated_parts = []
            progress = st.progress(0)
            status = st.empty()
            live = st.empty()
            started = time.time()

            for start in range(0, total, BATCH_SIZE):
                batch = chunks[start:start + BATCH_SIZE]

                # NLLB is a translation model; actual direction is set by language IDs.
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(DEVICE)

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        forced_bos_token_id=target_id,
                        max_new_tokens=512,
                        num_beams=1,
                        do_sample=False,
                        use_cache=True
                    )

                decoded = tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                translated_parts.extend(decoded)

                done = min(start + len(batch), total)
                result_so_far = "\n\n".join(translated_parts)
                progress.progress(
                    done / total,
                    text=f"Translating chunk {done}/{total}"
                )
                status.info(
                    f"Completed {done}/{total} chunks | "
                    f"Elapsed: {time.time() - started:.1f} seconds"
                )
                live.text_area(
                    "Live translation output",
                    result_so_far,
                    height=220,
                    key=f"live_{st.session_state.tid}_{done}"
                )

            result = "\n\n".join(translated_parts)
            timestamp = datetime.now().isoformat(timespec="seconds")

            st.session_state.output = result
            st.session_state.translated = True
            st.session_state.meta = {
                "from": src,
                "to": dst,
                "thread": st.session_state.tid,
                "time": timestamp
            }
            st.session_state.history.insert(0, {
                **st.session_state.meta,
                "chunks": total,
                "characters": len(input_text),
                "seconds": round(time.time() - started, 2)
            })
            st.success(
                f"Translation completed in {time.time() - started:.1f} seconds!"
            )

        except Exception as e:
            st.session_state.translated = False
            st.error(f"Translation failed: {e}")

# Safe access: no missing-key errors
if st.session_state.get("translated", False) and st.session_state.get("output"):
    result = st.session_state.output
    meta = st.session_state.get("meta", {})
    source_lang = meta.get("from", src)
    target_lang = meta.get("to", dst)
    thread_id = meta.get("thread", st.session_state.tid)
    timestamp = meta.get("time", "")

    st.subheader("✅ Final translation")
    st.text_area("Translated text", result, height=250, key="final_translation")

    # Generate PDF
    pdf_buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AIT GLOBAL TECHNOLOGIES", styles["Title"]),
        Paragraph(
            f"{escape(source_lang)} → {escape(target_lang)}",
            styles["Normal"]
        ),
        Paragraph(f"Thread ID: {escape(thread_id)}", styles["Normal"]),
        Paragraph(f"Date: {escape(timestamp)}", styles["Normal"]),
        Spacer(1, 12)
    ]

    for paragraph in result.split("\n\n"):
        if paragraph.strip():
            safe_text = escape(paragraph).replace("\n", "<br/>")
            story.append(Paragraph(safe_text, styles["BodyText"]))
            story.append(Spacer(1, 6))

    SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch
    ).build(story)

    # Generate DOCX
    word_doc = Document()
    word_doc.add_heading("AIT GLOBAL TECHNOLOGIES", 0)
    word_doc.add_paragraph(f"{source_lang} → {target_lang}")
    word_doc.add_paragraph(f"Thread ID: {thread_id}")
    word_doc.add_paragraph(f"Date: {timestamp}")

    for paragraph in result.split("\n\n"):
        if paragraph.strip():
            word_doc.add_paragraph(paragraph)

    word_buffer = io.BytesIO()
    word_doc.save(word_buffer)

    a, b = st.columns(2)
    a.download_button(
        "⬇ Download PDF",
        data=pdf_buffer.getvalue(),
        file_name="translation.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    b.download_button(
        "⬇ Download DOCX",
        data=word_buffer.getvalue(),
        file_name="translation.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True
    )

with st.expander("🕘 Translation history"):
    st.write(st.session_state.history)


