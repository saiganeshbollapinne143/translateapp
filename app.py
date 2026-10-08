import csv
import html
import importlib.metadata as md
import io
import json
import re
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st
import torch
from docx import Document
from fpdf import FPDF
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(page_title="AI Translator", page_icon="🌐", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = str(BASE_DIR / "chroma_db")

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

LATIN_FONT = BASE_DIR / "NotoSans-Regular.ttf"
SCRIPT_FONTS = {
    "Tamil": "NotoSansTamil-Regular.ttf",
    "Hindi": "NotoSansDevanagari-Regular.ttf",
    "Telugu": "NotoSansTelugu-Regular.ttf",
    "Kannada": "NotoSansKannada-Regular.ttf",
    "Malayalam": "NotoSansMalayalam-Regular.ttf",
}
FONT_BASE_URLS = [
    "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts",
    "https://github.com/notofonts/notofonts.github.io/raw/main/fonts",
]


# =========================================================
# RESOURCES
# =========================================================

@st.cache_resource
def load_database():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name="translation_records")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


collection = load_database()
tokenizer, model = load_model()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# =========================================================
# INPUT: READ TEXT FROM UPLOADED FILES
# =========================================================

def decode_bytes(data):
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def join_cells(cells):
    return " | ".join(c.strip() for c in cells if c and c.strip())


class _TextCollector(HTMLParser):
    """Collects visible text from HTML / XML."""

    SKIP = {"script", "style", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "table"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def json_strings(node, out):
    """Collect every string value from any JSON structure."""
    if isinstance(node, str):
        if node.strip():
            out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            json_strings(value, out)
    elif isinstance(node, list):
        for item in node:
            json_strings(item, out)


def read_pdf(data):
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)


def read_docx(data):
    doc = Document(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            lines.append(join_cells(c.text for c in row.cells))
    return "\n".join(lines)


def read_delimited(data, delimiter):
    rows = csv.reader(io.StringIO(decode_bytes(data)), delimiter=delimiter)
    return "\n".join(line for line in (join_cells(row) for row in rows) if line)


def read_json(data):
    text = decode_bytes(data)
    try:
        strings = []
        json_strings(json.loads(text), strings)
        return "\n".join(strings)
    except json.JSONDecodeError:
        return text


def read_markup(data):
    parser = _TextCollector()
    parser.feed(decode_bytes(data))
    lines = (line.strip() for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def read_rtf(data):
    raw = decode_bytes(data)
    raw = re.sub(r"\\'([0-9a-fA-F]{2})",
                 lambda m: bytes.fromhex(m.group(1)).decode("cp1252", errors="ignore"), raw)
    raw = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), raw)
    raw = re.sub(r"\\par[d]?", "\n", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    return re.sub(r"[{}]", "", raw).strip()


def read_excel(data):
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)
    lines = []
    for frame in sheets.values():
        for row in frame.fillna("").values.tolist():
            line = join_cells(map(str, row))
            if line:
                lines.append(line)
    return "\n".join(lines)


def read_pptx(data):
    from pptx import Presentation

    lines = []
    for slide in Presentation(io.BytesIO(data)).slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(r.text for r in paragraph.runs)
                    if text.strip():
                        lines.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    lines.append(join_cells(c.text for c in row.cells))
    return "\n".join(lines)


def read_opendocument(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        raw = archive.read("content.xml").decode("utf-8", errors="ignore")
    raw = re.sub(r"</text:(p|h)>|<text:line-break/>", "\n", raw)
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def read_image(data):
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "Image input needs OCR. Install: pip install pytesseract pillow "
            "and install the Tesseract OCR program."
        )
    return pytesseract.image_to_string(Image.open(io.BytesIO(data)))


TEXT_LIKE = ["txt", "md", "markdown", "log", "rst", "ini", "cfg", "conf",
             "yaml", "yml", "toml", "srt", "vtt", "tex"]

READERS = {
    "pdf": read_pdf,
    "docx": read_docx,
    "csv": lambda d: read_delimited(d, ","),
    "tsv": lambda d: read_delimited(d, "\t"),
    "json": read_json,
    "html": read_markup, "htm": read_markup, "xml": read_markup,
    "rtf": read_rtf,
    "xlsx": read_excel, "xlsm": read_excel,
    "pptx": read_pptx,
    "odt": read_opendocument, "ods": read_opendocument, "odp": read_opendocument,
    **{ext: read_image for ext in ("png", "jpg", "jpeg", "bmp", "tiff", "webp")},
}

INPUT_EXTENSIONS = TEXT_LIKE + list(READERS)


def extract_text(file):
    ext = file.name.lower().rsplit(".", 1)[-1] if "." in file.name else ""
    return READERS.get(ext, decode_bytes)(file.getvalue())


# =========================================================
# TRANSLATION
# =========================================================

def split_text(text, size=700):
    chunks, current = [], ""
    for paragraph in text.split("\n"):
        for word in paragraph.split():
            if len(current) + len(word) + 1 <= size:
                current += (" " if current else "") + word
            else:
                if current:
                    chunks.append(current)
                current = word
    if current:
        chunks.append(current)
    return chunks


def translate(text, source_code, target_code):
    """Returns (translated_text, [(source_chunk, translated_chunk), ...])."""
    chunks = split_text(text)
    if not chunks:
        return "", []

    progress = st.progress(0)
    status = st.empty()
    tokenizer.src_lang = source_code
    target_id = tokenizer.convert_tokens_to_ids(target_code)
    results = []

    for i, chunk in enumerate(chunks, 1):
        status.write(f"Translating {i} / {len(chunks)}...")
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=2,
                do_sample=False,
            )
        results.append(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
        progress.progress(i / len(chunks))

    progress.empty()
    status.success("Translation completed!")
    return "\n\n".join(results), list(zip(chunks, results))


def save_record(source_text, translated_text, source_language, target_language):
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[translated_text],
        metadatas=[{
            "source_text": source_text[:4000],
            "source_language": source_language,
            "target_language": target_language,
            "thread_id": st.session_state.thread_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }],
    )


# =========================================================
# OUTPUT BUILDERS
# Each builder takes a context dict (text, pairs, source_text,
# source_language, target_language, timestamp) and returns bytes,
# or None when it cannot be built (the format is then not offered).
# =========================================================

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(value):
    return _CONTROL_CHARS.sub("", value or "")


def create_txt(ctx):
    return ctx["text"].encode("utf-8")


def create_md(ctx):
    return (
        f"# Translation: {ctx['source_language']} → {ctx['target_language']}\n\n"
        f"_Generated {ctx['timestamp']}_\n\n{ctx['text']}"
    ).encode("utf-8")


def create_html(ctx):
    src, tgt = html.escape(ctx["source_language"]), html.escape(ctx["target_language"])
    paragraphs = "\n".join(
        f"<p>{html.escape(p)}</p>" for p in ctx["text"].split("\n\n") if p.strip()
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Translation: {src} to {tgt}</title>
<style>
body {{ font-family: 'Noto Sans', 'Noto Sans Tamil', 'Noto Sans Devanagari',
       'Noto Sans Telugu', 'Noto Sans Kannada', 'Noto Sans Malayalam',
       Arial, sans-serif; max-width: 800px; margin: 2rem auto; line-height: 1.7; }}
.meta {{ color: #666; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>{src} → {tgt}</h1>
<p class="meta">Generated {html.escape(ctx['timestamp'])}</p>
{paragraphs}
</body>
</html>
""".encode("utf-8")


def create_csv(ctx):
    frame = pd.DataFrame(
        [(i, ctx["source_language"], ctx["target_language"], s, t)
         for i, (s, t) in enumerate(ctx["pairs"], 1)],
        columns=["segment", "source_language", "target_language",
                 "source_text", "translated_text"],
    )
    # utf-8-sig so Excel opens Indian scripts correctly
    return frame.to_csv(index=False).encode("utf-8-sig")


def create_json(ctx):
    payload = {
        "source_language": ctx["source_language"],
        "target_language": ctx["target_language"],
        "timestamp": ctx["timestamp"],
        "source_text": ctx["source_text"],
        "translated_text": ctx["text"],
        "segments": [{"index": i, "source": s, "translation": t}
                     for i, (s, t) in enumerate(ctx["pairs"], 1)],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def create_xml(ctx):
    root = ET.Element(
        "translation",
        source_language=ctx["source_language"],
        target_language=ctx["target_language"],
        timestamp=ctx["timestamp"],
    )
    for i, (src, tr) in enumerate(ctx["pairs"], 1):
        segment = ET.SubElement(root, "segment", index=str(i))
        ET.SubElement(segment, "source").text = src
        ET.SubElement(segment, "target").text = tr
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _rtf_escape(value):
    out = []
    for ch in value:
        code = ord(ch)
        if ch in "\\{}":
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\par\n")
        elif code < 128:
            out.append(ch)
        else:
            units = [code]
            if code > 0xFFFF:  # surrogate pair for characters outside the BMP
                code -= 0x10000
                units = [0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)]
            out.extend(f"\\u{u - 65536 if u > 32767 else u}?" for u in units)
    return "".join(out)


def create_rtf(ctx):
    return ("{\\rtf1\\ansi\\deff0\\uc1\n" + _rtf_escape(ctx["text"]) + "\n}").encode("ascii")


def create_docx(ctx):
    doc = Document()
    for paragraph in ctx["text"].split("\n"):
        doc.add_paragraph(paragraph)
    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()


def create_xlsx(ctx):
    frame = pd.DataFrame(
        [(i, ctx["source_language"], ctx["target_language"], s, t)
         for i, (s, t) in enumerate(ctx["pairs"], 1)],
        columns=["Segment", "Source Language", "Target Language",
                 "Source Text", "Translated Text"],
    )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Translation")
        sheet = writer.sheets["Translation"]
        for column, width in zip("ABCDE", (10, 18, 18, 60, 60)):
            sheet.column_dimensions[column].width = width
    return output.getvalue()


def create_pptx(ctx):
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    for _, translated in ctx["pairs"]:
        slide = prs.slides.add_slide(blank)
        box = slide.shapes.add_textbox(Inches(0.6), Inches(0.5), Inches(12.1), Inches(6.5))
        frame = box.text_frame
        frame.word_wrap = True
        frame.text = translated
        for paragraph in frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(18)

    output = io.BytesIO()
    prs.save(output)
    return output.getvalue()


def create_pdf(ctx):
    """Needs the Latin Noto font (plus the script font for Indian languages)."""
    if not LATIN_FONT.exists():
        return None

    script_file = SCRIPT_FONTS.get(ctx["target_language"])
    script_path = BASE_DIR / script_file if script_file else None
    if script_path and not script_path.exists():
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Latin", "", str(LATIN_FONT))
    primary = "Latin"

    if script_path:
        pdf.add_font("Script", "", str(script_path))
        pdf.set_fallback_fonts(["Latin"])
        primary = "Script"

    pdf.set_font(primary, size=12)

    try:
        pdf.set_text_shaping(True)
    except Exception:  # uharfbuzz not installed
        pass

    pdf.multi_cell(0, 8, ctx["text"], new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


# (label, extension, mime type, builder)
OUTPUT_FORMATS = [
    ("📄 TXT", "txt", "text/plain", create_txt),
    ("📕 PDF", "pdf", "application/pdf", create_pdf),
    ("📝 DOCX", "docx",
     "application/vnd.openxmlformats-officedocument.wordprocessingml.document", create_docx),
    ("📊 CSV", "csv", "text/csv", create_csv),
    ("🧾 JSON", "json", "application/json", create_json),
    ("📘 Markdown", "md", "text/markdown", create_md),
    ("🌐 HTML", "html", "text/html", create_html),
    ("🧬 XML", "xml", "application/xml", create_xml),
    ("📃 RTF", "rtf", "application/rtf", create_rtf),
    ("📗 XLSX", "xlsx",
     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", create_xlsx),
    ("📙 PPTX", "pptx",
     "application/vnd.openxmlformats-officedocument.presentationml.presentation", create_pptx),
]


def build_outputs(translated, pairs, source_text, source_name, target_name):
    """Returns {ext: (label, mime, data)} for every format that could be built."""
    ctx = {
        "text": clean(translated),
        "pairs": [(clean(s), clean(t)) for s, t in pairs],
        "source_text": clean(source_text),
        "source_language": source_name,
        "target_language": target_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    files = {}
    for label, ext, mime, builder in OUTPUT_FORMATS:
        try:
            data = builder(ctx)
        except Exception:
            data = None  # a failing format is simply not offered
        if data:
            files[ext] = (label, mime, data)
    return files


# =========================================================
# FONT DOWNLOAD
# =========================================================

def download_font(filename):
    """Returns (success, message) for one font file."""
    destination = BASE_DIR / filename
    if destination.exists():
        return True, "already available"

    family = filename.rsplit("-", 1)[0]
    error = ""

    for base in FONT_BASE_URLS:
        url = f"{base}/{family}/hinted/ttf/{filename}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        except Exception as exc:
            error = str(exc)
            continue

        if len(data) > 10000 and data[:4] in (b"\x00\x01\x00\x00", b"true"):
            destination.write_bytes(data)
            return True, "downloaded"
        error = "Downloaded file is not a valid TTF"

    return False, f"failed: {error}"


def download_fonts():
    names = [LATIN_FONT.name, *SCRIPT_FONTS.values()]
    return [(name, *download_font(name)) for name in names]


# =========================================================
# HISTORY
# =========================================================

def load_records(search=""):
    total = collection.count()
    if total == 0:
        return []

    if search:
        result = collection.query(query_texts=[search], n_results=min(20, total))
        ids, docs, metas = result["ids"][0], result["documents"][0], result["metadatas"][0]
    else:
        result = collection.get()
        ids, docs, metas = result["ids"], result["documents"], result["metadatas"]

    records = [
        {"id": rid, "translated": doc, **(meta or {})}
        for rid, doc, meta in zip(ids, docs, metas)
    ]
    if not search:
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records


HISTORY_COLUMNS = ["timestamp", "source_language", "target_language",
                   "source_text", "translated_text", "thread_id"]


def history_rows(records):
    return [
        {
            "timestamp": r.get("timestamp", ""),
            "source_language": r.get("source_language", ""),
            "target_language": r.get("target_language", ""),
            "source_text": r.get("source_text", ""),
            "translated_text": r["translated"],
            "thread_id": r.get("thread_id", ""),
        }
        for r in records
    ]


# =========================================================
# PAGES
# =========================================================

def page_translator():
    st.title("🌐 AI Translator")
    st.caption("Translate text and documents using AI")

    col1, col2 = st.columns(2)
    source_name = col1.selectbox("Source Language", list(LANGUAGES))
    target_name = col2.selectbox("Target Language", list(LANGUAGES), index=1)

    text = st.text_area("Enter text", height=180, placeholder="Type or paste your text here...")
    file = st.file_uploader(
        "Upload a file (" + ", ".join(e.upper() for e in INPUT_EXTENSIONS[:12])
        + ", ... see Guide for all types)",
        type=INPUT_EXTENSIONS,
    )

    if st.button("🔄 Translate", type="primary", use_container_width=True):
        if file:
            try:
                text = extract_text(file)
            except Exception as exc:
                st.error(f"Could not read {file.name}: {exc}")
                st.stop()

        if not text.strip():
            st.warning("Enter text or upload a document.")
            st.stop()

        if source_name == target_name:
            st.warning("Source and target languages are the same.")
            st.stop()

        translated, pairs = translate(text, LANGUAGES[source_name], LANGUAGES[target_name])
        save_record(text, translated, source_name, target_name)

        st.session_state.last_result = {
            "text": translated,
            "target": target_name,
            "files": build_outputs(translated, pairs, text, source_name, target_name),
        }

    result = st.session_state.get("last_result")
    if not result:
        return

    st.subheader("✅ Translated Text")
    st.text_area("Output", result["text"], height=300)

    st.subheader("⬇️ Download as")
    files = list(result["files"].items())

    for start in range(0, len(files), 4):
        for column, (ext, (label, mime, data)) in zip(st.columns(4), files[start:start + 4]):
            column.download_button(
                f"{label} (.{ext})",
                data,
                file_name=f"translated_{result['target']}.{ext}",
                mime=mime,
                key=f"dl_{ext}",
                use_container_width=True,
            )


def page_history():
    st.title("📚 Translation History")

    search = st.text_input("🔎 Search translation history", placeholder="Search saved translations...")
    records = load_records(search)

    col1, col2 = st.columns(2)
    col1.metric("Total Records", collection.count())
    col2.metric("Shown", len(records))
    st.divider()

    if not records:
        st.info("No translation records found.")

    for record in records:
        with st.container(border=True):
            head, stamp = st.columns([3, 1])
            head.markdown(
                f"### 🌐 {record.get('source_language', '?')} → {record.get('target_language', '?')}"
            )
            stamp.caption(record.get("timestamp", ""))

            left, right = st.columns(2)
            left.markdown("**Original Text**")
            left.write(record.get("source_text", ""))
            right.markdown("**Translated Text**")
            right.write(record["translated"])

            st.caption("🧵 Thread ID: " + record.get("thread_id", ""))

            with st.expander("✏️ Edit / 🗑️ Delete"):
                new_text = st.text_area(
                    "Edit translated text", record["translated"],
                    key=f"edit_{record['id']}", height=150,
                )
                c1, c2 = st.columns(2)

                if c1.button("💾 Save changes", key=f"save_{record['id']}"):
                    collection.update(ids=[record["id"]], documents=[new_text])
                    st.success("Record updated.")
                    st.rerun()

                if c2.button("🗑️ Delete record", key=f"delete_{record['id']}"):
                    collection.delete(ids=[record["id"]])
                    st.success("Record deleted.")
                    st.rerun()

    if collection.count() == 0:
        return

    st.divider()
    rows = history_rows(load_records())

    history_csv = pd.DataFrame(rows, columns=HISTORY_COLUMNS).to_csv(index=False).encode("utf-8-sig")
    history_json = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
    history_txt = "\n\n".join(
        f"[{r['timestamp']}] {r['source_language'] or '?'} → {r['target_language'] or '?'}\n"
        f"{r['source_text']}\n---\n{r['translated_text']}"
        for r in rows
    ).encode("utf-8")

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Export history (CSV)", history_csv,
                       file_name="translation_history.csv", mime="text/csv",
                       use_container_width=True)
    d2.download_button("⬇️ Export history (JSON)", history_json,
                       file_name="translation_history.json", mime="application/json",
                       use_container_width=True)
    d3.download_button("⬇️ Export history (TXT)", history_txt,
                       file_name="translation_history.txt", mime="text/plain",
                       use_container_width=True)

    confirm = st.checkbox("I understand this will permanently delete all records")
    if st.button("🗑️ Clear All History", disabled=not confirm):
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        st.success("Translation history cleared.")
        st.rerun()


def page_usage():
    st.title("📊 Usage Dashboard")
    st.caption("See visually how the translator is being used")

    records = load_records()
    if not records:
        st.info("No usage data yet. Translate something and it will appear here.")
        return

    df = pd.DataFrame(records)
    for column in ["timestamp", "source_text", "source_language", "target_language", "thread_id"]:
        if column not in df:
            df[column] = ""

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["characters"] = df["source_text"].astype(str).str.len()
    df["pair"] = df["source_language"] + " → " + df["target_language"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Translations", len(df))
    c2.metric("Characters translated", f"{int(df['characters'].sum()):,}")
    c3.metric("Conversations", df["thread_id"].nunique())
    c4.metric("Top target language", df["target_language"].mode().iat[0])
    st.divider()

    left, right = st.columns(2)
    left.subheader("Translations per day")
    left.bar_chart(df.groupby("date").size())
    right.subheader("Target languages")
    right.bar_chart(df["target_language"].value_counts())

    st.subheader("Language pairs")
    pairs = (
        df["pair"].value_counts()
        .rename_axis("Language pair")
        .reset_index(name="Translations")
    )
    pairs["Share"] = pairs["Translations"] / pairs["Translations"].sum() * 100
    st.dataframe(
        pairs, hide_index=True, use_container_width=True,
        column_config={
            "Share": st.column_config.ProgressColumn(
                "Share", format="%.0f%%", min_value=0, max_value=100
            )
        },
    )


def pkg_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def page_guide():
    st.title("ℹ️ Requirements & Guide")
    st.caption("Check application requirements")

    packages = [
        ("streamlit", "Web interface"),
        ("torch", "Runs translation model"),
        ("transformers", "Loads NLLB-200"),
        ("sentencepiece", "Tokenizer"),
        ("chromadb", "Translation history"),
        ("pypdf", "Reads PDF"),
        ("python-docx", "Reads and creates DOCX"),
        ("fpdf2", "Creates PDF"),
        ("uharfbuzz", "Shapes Indian scripts"),
        ("pandas", "Usage dashboard, reads Excel"),
        ("openpyxl", "Reads and creates XLSX"),
        ("python-pptx", "Reads and creates PPTX"),
    ]
    package_rows = []
    for name, purpose in packages:
        version = pkg_version(name)
        package_rows.append({
            "Package": name,
            "Used for": purpose,
            "Status": "✅ Installed" if version else "❌ Missing",
            "Version": version or "-",
        })

    font_rows = [
        {
            "Language": language,
            "Font file": path.name,
            "Status": "✅ Found" if path.exists() else "❌ Missing",
        }
        for language, path in [("Latin", LATIN_FONT)]
        + [(lang, BASE_DIR / f) for lang, f in SCRIPT_FONTS.items()]
    ]

    all_rows = package_rows + font_rows
    ready = sum(row["Status"].startswith("✅") for row in all_rows)

    st.subheader("Readiness")
    st.progress(ready / len(all_rows), text=f"{ready} of {len(all_rows)} requirements ready")

    c1, c2, c3 = st.columns(3)
    c1.metric("Device", "GPU (CUDA)" if DEVICE == "cuda" else "CPU")
    c2.metric("Model", MODEL.split("/")[-1])
    c3.metric("History records", collection.count())
    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Python packages")
        st.dataframe(pd.DataFrame(package_rows), hide_index=True, use_container_width=True)
        st.code(
            "pip install streamlit torch transformers sentencepiece chromadb pypdf "
            "python-docx fpdf2 uharfbuzz pandas openpyxl python-pptx",
            language="bash",
        )
        st.caption(
            "Optional (image input via OCR): pip install pytesseract pillow, "
            "plus the Tesseract OCR program."
        )

    with right:
        st.subheader("PDF Fonts")
        st.dataframe(pd.DataFrame(font_rows), hide_index=True, use_container_width=True)

        missing = any(row["Status"].startswith("❌") for row in font_rows)
        if st.button("⬇️ Download missing fonts", type="primary",
                     disabled=not missing, use_container_width=True):
            with st.spinner("Downloading fonts..."):
                st.session_state.font_results = download_fonts()
            st.rerun()

        for filename, success, message in st.session_state.pop("font_results", []):
            (st.success if success else st.error)(f"{filename}: {message}")

    st.divider()
    st.subheader("How to use")

    steps = [
        ("1️⃣ Choose languages", "Select source and target language."),
        ("2️⃣ Add content", "Type text or upload any supported file."),
        ("3️⃣ Translate", "Click Translate."),
        ("4️⃣ Download", "Pick any format: TXT, PDF, DOCX, CSV, JSON, MD, HTML, XML, RTF, XLSX, PPTX."),
    ]
    for column, (title, description) in zip(st.columns(4), steps):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    st.subheader("Supported languages")
    st.write(", ".join(LANGUAGES))

    st.subheader("Supported input files")
    st.write(", ".join(f".{ext}" for ext in INPUT_EXTENSIONS))

    st.subheader("Supported output files")
    st.write(", ".join(f".{ext}" for _, ext, _, _ in OUTPUT_FORMATS))

    st.info(
        "PDF files containing scanned images may require OCR because pypdf "
        "can only extract existing PDF text."
    )


# =========================================================
# SIDEBAR + ROUTING
# =========================================================

PAGES = {
    "🏠 Translator": page_translator,
    "📚 History": page_history,
    "📊 Usage": page_usage,
    "ℹ️ Requirements & Guide": page_guide,
}

st.sidebar.title("🌐 AI Translator")
page = st.sidebar.radio("Navigation", list(PAGES))

st.sidebar.divider()
st.sidebar.write("**Current Thread**")
st.sidebar.code(st.session_state.thread_id)

if st.sidebar.button("🆕 New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pop("last_result", None)
    st.rerun()

PAGES[page]()