import csv
import html
import io
import json
import re
import uuid
import zipfile
import importlib.metadata as md
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import streamlit as st
import torch
import chromadb
import pandas as pd
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Translator",
    page_icon="🌐",
    layout="wide"
)

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

# Every file extension the uploader accepts.
# Plain-text style files are all read the same way (see extract_text).
TEXT_LIKE = [
    "txt", "md", "markdown", "log", "rst", "ini", "cfg", "conf",
    "yaml", "yml", "toml", "srt", "vtt", "tex",
]
INPUT_EXTENSIONS = TEXT_LIKE + [
    "pdf", "docx", "csv", "tsv", "json", "html", "htm", "xml",
    "rtf", "xlsx", "xlsm", "pptx", "odt", "ods", "odp",
    "png", "jpg", "jpeg", "bmp", "tiff", "webp",
]


# =========================================================
# LOAD CHROMADB
# =========================================================

@st.cache_resource
def load_database():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name="translation_records"
    )


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


collection = load_database()
tokenizer, model = load_model()


# =========================================================
# SESSION
# =========================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# =========================================================
# FILE EXTRACTION (INPUT)
# =========================================================

def decode_bytes(data):

    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):

        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue

    return data.decode("utf-8", errors="ignore")


class _TextCollector(HTMLParser):
    """Collects visible text from HTML / XML."""

    SKIP = {"script", "style", "head"}
    BLOCK = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3",
        "h4", "h5", "h6", "section", "article", "table",
    }

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


def markup_to_text(raw):

    parser = _TextCollector()
    parser.feed(raw)

    lines = [
        line.strip()
        for line in "".join(parser.parts).splitlines()
    ]

    return "\n".join(line for line in lines if line)


def rtf_to_text(raw):

    raw = re.sub(
        r"\\'([0-9a-fA-F]{2})",
        lambda m: bytes.fromhex(m.group(1)).decode(
            "cp1252", errors="ignore"
        ),
        raw,
    )
    raw = re.sub(
        r"\\u(-?\d+)\??",
        lambda m: chr(int(m.group(1)) % 65536),
        raw,
    )
    raw = re.sub(r"\\par[d]?", "\n", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    raw = re.sub(r"[{}]", "", raw)

    return raw.strip()


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


def extract_text(file):

    name = file.name.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    data = file.getvalue()

    # ---- PDF ----
    if ext == "pdf":

        reader = PdfReader(io.BytesIO(data))

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    # ---- Word ----
    if ext == "docx":

        doc = Document(io.BytesIO(data))

        lines = [p.text for p in doc.paragraphs]

        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                lines.append(" | ".join(c for c in cells if c))

        return "\n".join(lines)

    # ---- CSV / TSV ----
    if ext in ("csv", "tsv"):

        text = decode_bytes(data)
        delimiter = "\t" if ext == "tsv" else ","

        rows = csv.reader(
            io.StringIO(text),
            delimiter=delimiter,
        )

        return "\n".join(
            " | ".join(cell.strip() for cell in row if cell.strip())
            for row in rows
            if any(cell.strip() for cell in row)
        )

    # ---- JSON ----
    if ext == "json":

        text = decode_bytes(data)

        try:
            strings = []
            json_strings(json.loads(text), strings)
            return "\n".join(strings)

        except json.JSONDecodeError:
            return text

    # ---- HTML / XML ----
    if ext in ("html", "htm", "xml"):
        return markup_to_text(decode_bytes(data))

    # ---- RTF ----
    if ext == "rtf":
        return rtf_to_text(decode_bytes(data))

    # ---- Excel ----
    if ext in ("xlsx", "xlsm"):

        sheets = pd.read_excel(
            io.BytesIO(data),
            sheet_name=None,
            header=None,
            dtype=str,
        )

        lines = []

        for frame in sheets.values():
            for row in frame.fillna("").values.tolist():
                cells = [str(c).strip() for c in row if str(c).strip()]
                if cells:
                    lines.append(" | ".join(cells))

        return "\n".join(lines)

    # ---- PowerPoint ----
    if ext == "pptx":

        from pptx import Presentation

        prs = Presentation(io.BytesIO(data))
        lines = []

        for slide in prs.slides:
            for shape in slide.shapes:

                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = "".join(r.text for r in paragraph.runs)
                        if text.strip():
                            lines.append(text)

                if getattr(shape, "has_table", False) and shape.has_table:
                    for row in shape.table.rows:
                        cells = [c.text.strip() for c in row.cells]
                        lines.append(" | ".join(c for c in cells if c))

        return "\n".join(lines)

    # ---- OpenDocument (odt / ods / odp) ----
    if ext in ("odt", "ods", "odp"):

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            raw = archive.read("content.xml").decode("utf-8", errors="ignore")

        raw = re.sub(r"</text:(p|h)>|<text:line-break/>", "\n", raw)
        raw = re.sub(r"<[^>]+>", "", raw)

        return html.unescape(raw).strip()

    # ---- Images (needs OCR) ----
    if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "webp"):

        try:
            import pytesseract
            from PIL import Image

        except ImportError:
            raise RuntimeError(
                "Image input needs OCR. Install: pip install pytesseract pillow "
                "and install the Tesseract OCR program."
            )

        return pytesseract.image_to_string(
            Image.open(io.BytesIO(data))
        )

    # ---- Anything else: plain text ----
    return decode_bytes(data)


# =========================================================
# SPLIT TEXT
# =========================================================

def split_text(text, size=700):

    chunks = []
    current = ""

    for paragraph in text.split("\n"):

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        words = paragraph.split()

        for word in words:

            if len(current) + len(word) + 1 <= size:

                current += (
                    " " if current else ""
                ) + word

            else:

                if current:
                    chunks.append(current)

                current = word

    if current:
        chunks.append(current)

    return chunks


# =========================================================
# TRANSLATION
# =========================================================

def translate(text, source_code, target_code):
    """Returns (translated_text, [(source_chunk, translated_chunk), ...])."""

    chunks = split_text(text)

    if not chunks:
        return "", []

    results = []

    progress = st.progress(0)
    status = st.empty()

    tokenizer.src_lang = source_code

    target_id = tokenizer.convert_tokens_to_ids(
        target_code
    )

    for i, chunk in enumerate(chunks):

        status.write(
            f"Translating {i + 1} / {len(chunks)}..."
        )

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(DEVICE)

        with torch.no_grad():

            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=2,
                do_sample=False,
            )

        translated = tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )[0]

        results.append(translated)

        progress.progress(
            (i + 1) / len(chunks)
        )

    progress.empty()
    status.success("Translation completed!")

    return "\n\n".join(results), list(zip(chunks, results))


# =========================================================
# SAVE HISTORY
# =========================================================

def save_record(
    source_text,
    translated_text,
    source_language,
    target_language,
):

    collection.add(

        ids=[str(uuid.uuid4())],

        documents=[translated_text],

        metadatas=[
            {
                "source_text": source_text[:4000],
                "source_language": source_language,
                "target_language": target_language,
                "thread_id": st.session_state.thread_id,
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        ],
    )


# =========================================================
# OUTPUT BUILDERS
# Each builder takes a context dict:
#   text, pairs, source_language, target_language,
#   source_text, timestamp
# and returns bytes.
# =========================================================

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(value):
    return _CONTROL_CHARS.sub("", value or "")


def create_txt(ctx):
    return ctx["text"].encode("utf-8")


def create_md(ctx):

    return (
        f"# Translation: {ctx['source_language']} → "
        f"{ctx['target_language']}\n\n"
        f"_Generated {ctx['timestamp']}_\n\n"
        + ctx["text"].replace("\n\n", "\n\n")
    ).encode("utf-8")


def create_html(ctx):

    paragraphs = "\n".join(
        f"<p>{html.escape(p)}</p>"
        for p in ctx["text"].split("\n\n")
        if p.strip()
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Translation: {html.escape(ctx['source_language'])} to {html.escape(ctx['target_language'])}</title>
<style>
body {{ font-family: 'Noto Sans', 'Noto Sans Tamil', 'Noto Sans Devanagari',
       'Noto Sans Telugu', 'Noto Sans Kannada', 'Noto Sans Malayalam',
       Arial, sans-serif; max-width: 800px; margin: 2rem auto; line-height: 1.7; }}
.meta {{ color: #666; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>{html.escape(ctx['source_language'])} → {html.escape(ctx['target_language'])}</h1>
<p class="meta">Generated {html.escape(ctx['timestamp'])}</p>
{paragraphs}
</body>
</html>
"""
    return page.encode("utf-8")


def create_csv(ctx):

    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(
        [
            "segment",
            "source_language",
            "target_language",
            "source_text",
            "translated_text",
        ]
    )

    for i, (src, tr) in enumerate(ctx["pairs"], 1):
        writer.writerow(
            [
                i,
                ctx["source_language"],
                ctx["target_language"],
                src,
                tr,
            ]
        )

    # utf-8-sig so Excel opens Indian scripts correctly
    return buffer.getvalue().encode("utf-8-sig")


def create_json(ctx):

    payload = {
        "source_language": ctx["source_language"],
        "target_language": ctx["target_language"],
        "timestamp": ctx["timestamp"],
        "source_text": ctx["source_text"],
        "translated_text": ctx["text"],
        "segments": [
            {
                "index": i,
                "source": src,
                "translation": tr,
            }
            for i, (src, tr) in enumerate(ctx["pairs"], 1)
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


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

    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
    )


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

            if code > 0xFFFF:
                code -= 0x10000
                units = [
                    0xD800 + (code >> 10),
                    0xDC00 + (code & 0x3FF),
                ]

            for unit in units:
                signed = unit - 65536 if unit > 32767 else unit
                out.append(f"\\u{signed}?")

    return "".join(out)


def create_rtf(ctx):

    body = _rtf_escape(ctx["text"])

    return (
        "{\\rtf1\\ansi\\deff0\\uc1\n" + body + "\n}"
    ).encode("ascii")


def create_xlsx(ctx):

    frame = pd.DataFrame(
        [
            {
                "Segment": i,
                "Source Language": ctx["source_language"],
                "Target Language": ctx["target_language"],
                "Source Text": src,
                "Translated Text": tr,
            }
            for i, (src, tr) in enumerate(ctx["pairs"], 1)
        ]
    )

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        frame.to_excel(
            writer,
            index=False,
            sheet_name="Translation",
        )

        sheet = writer.sheets["Translation"]

        for column, width in zip("ABCDE", (10, 18, 18, 60, 60)):
            sheet.column_dimensions[column].width = width

    return output.getvalue()


def create_pptx(ctx):

    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank = prs.slide_layouts[6]

    for _, translated in ctx["pairs"]:

        slide = prs.slides.add_slide(blank)

        box = slide.shapes.add_textbox(
            Inches(0.6),
            Inches(0.5),
            Inches(12.1),
            Inches(6.5),
        )

        frame = box.text_frame
        frame.word_wrap = True
        frame.text = translated

        for paragraph in frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(18)

    output = io.BytesIO()
    prs.save(output)

    return output.getvalue()


def create_docx(ctx):

    doc = Document()

    for paragraph in ctx["text"].split("\n"):
        doc.add_paragraph(paragraph)

    output = io.BytesIO()
    doc.save(output)

    return output.getvalue()


def create_pdf(ctx):

    language = ctx["target_language"]
    text = ctx["text"]

    script_font = SCRIPT_FONTS.get(language)

    script_path = (
        BASE_DIR / script_font
        if script_font
        else None
    )

    if script_path and not script_path.exists():
        return None

    pdf = FPDF()
    pdf.add_page()

    latin_name = None

    if LATIN_FONT.exists():

        pdf.add_font(
            "Latin",
            "",
            str(LATIN_FONT)
        )

        latin_name = "Latin"

    elif script_path:

        return None

    if script_path:

        pdf.add_font(
            "Script",
            "",
            str(script_path)
        )

        pdf.set_font(
            "Script",
            size=12
        )

        if latin_name:
            pdf.set_fallback_fonts(["Latin"])

    elif latin_name:

        pdf.set_font(
            "Latin",
            size=12
        )

    else:

        pdf.set_font(
            "Helvetica",
            size=12
        )

    try:

        pdf.set_text_shaping(True)

    except Exception:
        pass

    pdf.multi_cell(
        0,
        8,
        text,
        new_x="LMARGIN",
        new_y="NEXT",
    )

    return bytes(pdf.output())


# (label, extension, mime type, builder)
OUTPUT_FORMATS = [
    ("📄 TXT", "txt", "text/plain", create_txt),
    ("📕 PDF", "pdf", "application/pdf", create_pdf),
    (
        "📝 DOCX",
        "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        create_docx,
    ),
    ("📊 CSV", "csv", "text/csv", create_csv),
    ("🧾 JSON", "json", "application/json", create_json),
    ("📘 Markdown", "md", "text/markdown", create_md),
    ("🌐 HTML", "html", "text/html", create_html),
    ("🧬 XML", "xml", "application/xml", create_xml),
    ("📃 RTF", "rtf", "application/rtf", create_rtf),
    (
        "📗 XLSX",
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        create_xlsx,
    ),
    (
        "📙 PPTX",
        "pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        create_pptx,
    ),
]


def build_all_outputs(translated, pairs, source_text, source_name, target_name):
    """Build every output format. A failure in one never breaks the others."""

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
            files[ext] = {
                "label": label,
                "mime": mime,
                "data": data,
                "error": None if data else "required font missing",
            }

        except Exception as exc:
            files[ext] = {
                "label": label,
                "mime": mime,
                "data": None,
                "error": str(exc),
            }

    return files


# =========================================================
# DOWNLOAD FONTS
# =========================================================

def download_fonts():

    targets = [
        LATIN_FONT.name
    ] + list(SCRIPT_FONTS.values())

    results = []

    for fname in targets:

        destination = BASE_DIR / fname

        if destination.exists():

            results.append(
                (
                    fname,
                    True,
                    "already available"
                )
            )

            continue

        family = fname.rsplit(
            "-",
            1
        )[0]

        success = False
        error = ""

        for base in FONT_BASE_URLS:

            url = (
                f"{base}/{family}/"
                f"hinted/ttf/{fname}"
            )

            try:

                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    },
                )

                with urllib.request.urlopen(
                    request,
                    timeout=60
                ) as response:

                    data = response.read()

                if (
                    len(data) > 10000
                    and data[:4]
                    in (
                        b"\x00\x01\x00\x00",
                        b"true",
                    )
                ):

                    destination.write_bytes(data)

                    success = True
                    break

                error = (
                    "Downloaded file is not a valid TTF"
                )

            except Exception as exc:

                error = str(exc)

        results.append(
            (
                fname,
                success,
                "downloaded"
                if success
                else f"failed: {error}",
            )
        )

    return results


# =========================================================
# LOAD HISTORY
# =========================================================

def load_records(search=""):

    total = collection.count()

    if total == 0:
        return []

    if search:

        result = collection.query(
            query_texts=[search],
            n_results=min(20, total),
        )

        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]

    else:

        result = collection.get()

        ids = result["ids"]
        docs = result["documents"]
        metas = result["metadatas"]

    records = [

        {
            "id": record_id,
            "translated": document,
            **(metadata or {}),
        }

        for record_id, document, metadata
        in zip(ids, docs, metas)
    ]

    if not search:

        records.sort(
            key=lambda x: x.get(
                "timestamp",
                ""
            ),
            reverse=True,
        )

    return records


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌐 AI Translator")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Translator",
        "📚 History",
        "📊 Usage",
        "ℹ️ Requirements & Guide",
    ],
)

st.sidebar.divider()

st.sidebar.write("**Current Thread**")

st.sidebar.code(
    st.session_state.thread_id
)

if st.sidebar.button(
    "🆕 New Conversation"
):

    st.session_state.thread_id = str(
        uuid.uuid4()
    )

    st.session_state.pop(
        "last_result",
        None
    )

    st.rerun()


# =========================================================
# TRANSLATOR PAGE
# =========================================================

if page == "🏠 Translator":

    st.title("🌐 AI Translator")

    st.caption(
        "Translate text and documents using AI"
    )

    col1, col2 = st.columns(2)

    with col1:

        source_name = st.selectbox(
            "Source Language",
            list(LANGUAGES.keys()),
        )

    with col2:

        target_name = st.selectbox(
            "Target Language",
            list(LANGUAGES.keys()),
            index=1,
        )

    text = st.text_area(
        "Enter text",
        height=180,
        placeholder="Type or paste your text here...",
    )

    file = st.file_uploader(
        "Upload a file (" + ", ".join(
            ext.upper() for ext in INPUT_EXTENSIONS[:12]
        ) + ", ... see Guide for all types)",
        type=INPUT_EXTENSIONS,
    )

    if st.button(
        "🔄 Translate",
        type="primary",
        use_container_width=True,
    ):

        if file:

            try:
                text = extract_text(file)

            except Exception as exc:

                st.error(
                    f"Could not read {file.name}: {exc}"
                )

                st.stop()

        if not text.strip():

            st.warning(
                "Enter text or upload a document."
            )

            st.stop()

        if source_name == target_name:

            st.warning(
                "Source and target languages are the same."
            )

            st.stop()

        translated, pairs = translate(
            text,
            LANGUAGES[source_name],
            LANGUAGES[target_name],
        )

        save_record(
            text,
            translated,
            source_name,
            target_name,
        )

        st.session_state.last_result = {

            "text": translated,

            "target": target_name,

            "files": build_all_outputs(
                translated,
                pairs,
                text,
                source_name,
                target_name,
            ),
        }

    result = st.session_state.get(
        "last_result"
    )

    if result:

        st.subheader(
            "✅ Translated Text"
        )

        st.text_area(
            "Output",
            result["text"],
            height=300,
        )

        st.subheader(
            "⬇️ Download as"
        )

        files = list(result["files"].items())

        for start in range(0, len(files), 4):

            columns = st.columns(4)

            for column, (ext, info) in zip(
                columns,
                files[start:start + 4],
            ):

                with column:

                    if info["data"]:

                        st.download_button(
                            f"{info['label']} (.{ext})",
                            info["data"],
                            file_name=(
                                f"translated_"
                                f"{result['target']}.{ext}"
                            ),
                            mime=info["mime"],
                            use_container_width=True,
                        )

                    else:

                        st.button(
                            f"{info['label']} unavailable",
                            key=f"na_{ext}",
                            disabled=True,
                            help=info["error"],
                            use_container_width=True,
                        )

        if not result["files"]["pdf"]["data"]:

            st.warning(
                "PDF unavailable. "
                "Download the required Noto font first "
                "(Requirements & Guide page)."
            )


# =========================================================
# HISTORY
# =========================================================

if page == "📚 History":

    st.title(
        "📚 Translation History"
    )

    search = st.text_input(
        "🔎 Search translation history",
        placeholder="Search saved translations...",
    )

    records = load_records(search)

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Records",
        collection.count(),
    )

    col2.metric(
        "Shown",
        len(records),
    )

    st.divider()

    if not records:

        st.info(
            "No translation records found."
        )

    for record in records:

        with st.container(
            border=True
        ):

            head, stamp = st.columns(
                [3, 1]
            )

            head.markdown(
                f"### 🌐 "
                f"{record.get('source_language', '?')}"
                f" → "
                f"{record.get('target_language', '?')}"
            )

            stamp.caption(
                record.get(
                    "timestamp",
                    ""
                )
            )

            left, right = st.columns(2)

            with left:

                st.markdown(
                    "**Original Text**"
                )

                st.write(
                    record.get(
                        "source_text",
                        "",
                    )
                )

            with right:

                st.markdown(
                    "**Translated Text**"
                )

                st.write(
                    record["translated"]
                )

            st.caption(
                "🧵 Thread ID: "
                + record.get(
                    "thread_id",
                    "",
                )
            )

            with st.expander(
                "✏️ Edit / 🗑️ Delete"
            ):

                new_text = st.text_area(
                    "Edit translated text",
                    record["translated"],
                    key=f"edit_{record['id']}",
                    height=150,
                )

                c1, c2 = st.columns(2)

                if c1.button(
                    "💾 Save changes",
                    key=f"save_{record['id']}",
                ):

                    collection.update(
                        ids=[record["id"]],
                        documents=[new_text],
                    )

                    st.success(
                        "Record updated."
                    )

                    st.rerun()

                if c2.button(
                    "🗑️ Delete record",
                    key=f"delete_{record['id']}",
                ):

                    collection.delete(
                        ids=[record["id"]]
                    )

                    st.success(
                        "Record deleted."
                    )

                    st.rerun()

    if collection.count() > 0:

        st.divider()

        all_records = load_records()

        # ---- CSV export ----
        buffer = io.StringIO()

        writer = csv.writer(
            buffer
        )

        writer.writerow(
            [
                "timestamp",
                "source_language",
                "target_language",
                "source_text",
                "translated_text",
                "thread_id",
            ]
        )

        for record in all_records:

            writer.writerow(
                [
                    record.get("timestamp", ""),
                    record.get("source_language", ""),
                    record.get("target_language", ""),
                    record.get("source_text", ""),
                    record["translated"],
                    record.get("thread_id", ""),
                ]
            )

        # ---- JSON export ----
        history_json = json.dumps(
            [
                {
                    "timestamp": r.get("timestamp", ""),
                    "source_language": r.get("source_language", ""),
                    "target_language": r.get("target_language", ""),
                    "source_text": r.get("source_text", ""),
                    "translated_text": r["translated"],
                    "thread_id": r.get("thread_id", ""),
                }
                for r in all_records
            ],
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        # ---- TXT export ----
        history_txt = "\n\n".join(
            f"[{r.get('timestamp', '')}] "
            f"{r.get('source_language', '?')} → "
            f"{r.get('target_language', '?')}\n"
            f"{r.get('source_text', '')}\n---\n{r['translated']}"
            for r in all_records
        ).encode("utf-8")

        d1, d2, d3 = st.columns(3)

        d1.download_button(
            "⬇️ Export history (CSV)",
            buffer.getvalue().encode("utf-8-sig"),
            file_name="translation_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        d2.download_button(
            "⬇️ Export history (JSON)",
            history_json,
            file_name="translation_history.json",
            mime="application/json",
            use_container_width=True,
        )

        d3.download_button(
            "⬇️ Export history (TXT)",
            history_txt,
            file_name="translation_history.txt",
            mime="text/plain",
            use_container_width=True,
        )

        confirm = st.checkbox(
            "I understand this will permanently delete all records"
        )

        if st.button(
            "🗑️ Clear All History",
            disabled=not confirm,
        ):

            all_ids = collection.get()["ids"]

            if all_ids:
                collection.delete(
                    ids=all_ids
                )

            st.success(
                "Translation history cleared."
            )

            st.rerun()


# =========================================================
# USAGE DASHBOARD
# =========================================================

if page == "📊 Usage":

    st.title(
        "📊 Usage Dashboard"
    )

    st.caption(
        "See visually how the translator is being used"
    )

    records = load_records()

    if not records:

        st.info(
            "No usage data yet. "
            "Translate something and it will appear here."
        )

    else:

        df = pd.DataFrame(records)

        for column in [
            "timestamp",
            "source_text",
            "source_language",
            "target_language",
            "thread_id",
        ]:

            if column not in df:
                df[column] = ""

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df["date"] = df[
            "timestamp"
        ].dt.date

        df["characters"] = (
            df["source_text"]
            .astype(str)
            .str.len()
        )

        df["pair"] = (
            df["source_language"]
            + " → "
            + df["target_language"]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Translations",
            len(df)
        )

        c2.metric(
            "Characters translated",
            f"{int(df['characters'].sum()):,}",
        )

        c3.metric(
            "Conversations",
            df["thread_id"].nunique(),
        )

        c4.metric(
            "Top target language",
            df[
                "target_language"
            ].mode().iat[0],
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Translations per day"
            )

            st.bar_chart(
                df.groupby("date").size()
            )

        with right:

            st.subheader(
                "Target languages"
            )

            st.bar_chart(
                df[
                    "target_language"
                ].value_counts()
            )

        st.subheader(
            "Language pairs"
        )

        pairs = (
            df["pair"]
            .value_counts()
            .rename_axis("Language pair")
            .reset_index(
                name="Translations"
            )
        )

        pairs["Share"] = (
            pairs["Translations"]
            / pairs["Translations"].sum()
            * 100
        )

        st.dataframe(
            pairs,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Share":
                st.column_config.ProgressColumn(
                    "Share",
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                )
            },
        )


# =========================================================
# REQUIREMENTS & GUIDE
# =========================================================

if page == "ℹ️ Requirements & Guide":

    st.title(
        "ℹ️ Requirements & Guide"
    )

    st.caption(
        "Check application requirements"
    )

    def pkg_version(name):

        try:

            return md.version(name)

        except md.PackageNotFoundError:

            return None

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

        package_rows.append(
            {
                "Package": name,
                "Used for": purpose,
                "Status":
                    "✅ Installed"
                    if version
                    else "❌ Missing",
                "Version":
                    version or "-",
            }
        )

    font_files = [
        (
            "Latin",
            LATIN_FONT
        )
    ] + [
        (
            language,
            BASE_DIR / filename
        )
        for language, filename
        in SCRIPT_FONTS.items()
    ]

    font_rows = [

        {
            "Language": language,
            "Font file": path.name,
            "Status":
                "✅ Found"
                if path.exists()
                else "❌ Missing",
        }

        for language, path
        in font_files
    ]

    ready = sum(
        row["Status"].startswith("✅")
        for row
        in package_rows + font_rows
    )

    total = (
        len(package_rows)
        + len(font_rows)
    )

    st.subheader(
        "Readiness"
    )

    st.progress(
        ready / total,
        text=(
            f"{ready} of {total} "
            "requirements ready"
        ),
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Device",
        "GPU (CUDA)"
        if DEVICE == "cuda"
        else "CPU",
    )

    c2.metric(
        "Model",
        MODEL.split("/")[-1]
    )

    c3.metric(
        "History records",
        collection.count()
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Python packages"
        )

        st.dataframe(
            pd.DataFrame(package_rows),
            hide_index=True,
            use_container_width=True,
        )

        st.code(
            "pip install streamlit torch "
            "transformers sentencepiece "
            "chromadb pypdf python-docx "
            "fpdf2 uharfbuzz pandas "
            "openpyxl python-pptx",
            language="bash",
        )

        st.caption(
            "Optional (image input via OCR): "
            "pip install pytesseract pillow, plus the Tesseract OCR program."
        )

    with right:

        st.subheader(
            "PDF Fonts"
        )

        st.dataframe(
            pd.DataFrame(font_rows),
            hide_index=True,
            use_container_width=True,
        )

        missing_fonts = [
            row
            for row in font_rows
            if row["Status"].startswith("❌")
        ]

        if st.button(
            "⬇️ Download missing fonts",
            type="primary",
            disabled=not missing_fonts,
            use_container_width=True,
        ):

            with st.spinner(
                "Downloading fonts..."
            ):

                st.session_state.font_results = (
                    download_fonts()
                )

            st.rerun()

        for (
            filename,
            success,
            message,
        ) in st.session_state.pop(
            "font_results",
            [],
        ):

            if success:

                st.success(
                    f"{filename}: {message}"
                )

            else:

                st.error(
                    f"{filename}: {message}"
                )

    st.divider()

    st.subheader(
        "How to use"
    )

    steps = [

        (
            "1️⃣ Choose languages",
            "Select source and target language."
        ),

        (
            "2️⃣ Add content",
            "Type text or upload any supported file."
        ),

        (
            "3️⃣ Translate",
            "Click Translate."
        ),

        (
            "4️⃣ Download",
            "Pick any format: TXT, PDF, DOCX, CSV, JSON, MD, HTML, XML, RTF, XLSX, PPTX."
        ),

    ]

    columns = st.columns(4)

    for column, (title, description) in zip(
        columns,
        steps,
    ):

        with column:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{title}**"
                )

                st.write(
                    description
                )

    st.subheader(
        "Supported languages"
    )

    st.write(
        ", ".join(
            LANGUAGES.keys()
        )
    )

    st.subheader(
        "Supported input files"
    )

    st.write(
        ", ".join(
            f".{ext}" for ext in INPUT_EXTENSIONS
        )
    )

    st.subheader(
        "Supported output files"
    )

    st.write(
        ", ".join(
            f".{ext}" for _, ext, _, _ in OUTPUT_FORMATS
        )
    )

    st.info(
        "PDF files containing scanned images "
        "may require OCR because pypdf can only "
        "extract existing PDF text."
    )