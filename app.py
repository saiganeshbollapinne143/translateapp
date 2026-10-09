
import io, uuid, time, requests, streamlit as st
from datetime import datetime
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES", layout="wide")
st.title("🌐 AIT GLOBAL TECHNOLOGIES — Streaming Translator")
LANGS = ["English","Spanish","French","German","Hindi","Tamil","Telugu",
         "Kannada","Malayalam","Arabic","Chinese","Japanese","Portuguese"]
src = st.selectbox("Source language", LANGS)
dst = st.selectbox("Target language", LANGS, index=1)
key = st.text_input("Groq API key", type="password")
model = st.text_input("Groq model", "llama-3.3-70b-versatile")
prompt_tpl = st.text_area("Manual prompt template",
    "Translate from {source_language} to {target_language}. Preserve all meaning, headings and paragraphs. Return only the translation:\n\n{text}", height=120)
uploaded = st.file_uploader("Upload TXT / DOCX / PDF", type=["txt","docx","pdf"])
text = st.text_area("Or paste input text here", height=180)
if uploaded:
    try:
        if uploaded.name.endswith(".txt"): text = uploaded.getvalue().decode("utf-8", "replace")
        elif uploaded.name.endswith(".docx"):
            text = "\n".join(p.text for p in Document(uploaded).paragraphs)
        else:
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(uploaded).pages)
        st.info(f"Loaded: {uploaded.name}")
    except Exception as e: st.error(f"File read error: {e}")

if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state: st.session_state.history = []
st.caption("Thread ID: " + st.session_state.thread_id)
if st.button("🆕 New thread"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

if st.button("🚀 Translate & prepare PDF/DOCX", type="primary", use_container_width=True):
    if not key or not text.strip(): st.warning("Enter API key and input text.")
    elif "{text}" not in prompt_tpl: st.warning("Prompt template must contain {text}.")
    elif src == dst: st.warning("Choose different languages.")
    else:
        try:
            chunks = [text[i:i+1800] for i in range(0, len(text), 1800)]
            output, bar = "", st.progress(0)
            live = st.empty()
            for i, chunk in enumerate(chunks):
                prompt = prompt_tpl.replace("{source_language}", src).replace("{target_language}", dst).replace("{text}", chunk)
                r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "temperature": 0.1,
                          "messages": [{"role":"user","content":prompt}]}, timeout=180)
                r.raise_for_status()
                output += r.json()["choices"][0]["message"]["content"].strip() + "\n\n"
                live.text_area("Live translated output", output, height=300)
                bar.progress((i+1)/len(chunks), text=f"Translating chunk {i+1}/{len(chunks)}")
            st.session_state.translation = output.strip()
            st.session_state.history.insert(0, {"time": datetime.now().isoformat(timespec="seconds"),
                "thread": st.session_state.thread_id, "source": src, "target": dst})
            st.success("Translation complete.")
        except Exception as e: st.error(f"Translation failed: {e}")

if st.session_state.get("translation"):
    result = st.session_state.translation
    st.subheader("📄 Download translated documents")
    pdf = io.BytesIO()
    doc = SimpleDocTemplate(pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("AIT GLOBAL TECHNOLOGIES", styles["Title"]),
             Paragraph(f"{src} to {dst} | Thread: {st.session_state.thread_id}", styles["Normal"]),
             Spacer(1, 16)]
    story += [Paragraph(escape(p), styles["BodyText"]) for p in result.splitlines() if p.strip()]
    doc.build(story)
    d = Document()
    d.add_heading("AIT GLOBAL TECHNOLOGIES", 0)
    d.add_paragraph(f"{src} to {dst} | Thread: {st.session_state.thread_id}")
    for p in result.splitlines():
        if p.strip(): d.add_paragraph(p)
    word = io.BytesIO()
    d.save(word)
    a, b = st.columns(2)
    a.download_button("⬇ Download PDF", pdf.getvalue(), "translation.pdf", "application/pdf", use_container_width=True)
    b.download_button("⬇ Download DOCX", word.getvalue(), "translation.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)

with st.expander("🕘 Translation history"):
    st.write(st.session_state.history)



