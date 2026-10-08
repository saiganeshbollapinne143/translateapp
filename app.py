"""
AI Translator - fast single-file version (NLLB-200 + Streamlit)

Run:  streamlit run app.py

Speed-ups vs. the original:
  * Heavy libraries (torch, transformers, pandas, pptx, fpdf...) load lazily
  * Model loads only when you press Translate (not on every app start)
  * Batched, length-sorted, greedy decoding + inference_mode
  * int8 dynamic quantization on CPU / fp16 on GPU
  * SQLite history instead of ChromaDB (no embedding model download)
  * Only the output format you choose is built, on demand
"""
import csv
import html
import importlib.metadata as md
import io
import json
import os
import re
import sqlite3
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import streamlit as st

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="AI Translator", page_icon="🌐", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL = "facebook/nllb-200-distilled-600M"
DB_FILE = str(BASE_DIR / "translations.db")

LANGUAGES = {
    "English": "eng_Latn", "Hindi": "hin_Deva", "Telugu": "tel_Telu",
    "Tamil": "tam_Taml", "Kannada": "kan_Knda", "Malayalam": "mal_Mlym",
    "French": "fra_Latn", "German": "deu_Latn", "Spanish": "spa_Latn",
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

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# =========================================================
# DATABASE (SQLite - instant, no model downloads)
# =========================================================
@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY, timestamp TEXT, source_language TEXT,
            target_language TEXT, source_text TEXT, translated_text TEXT,
            thread_id TEXT)"""
    )
    conn.commit()
    return conn


def db_add(source_text, translated, src_lang, tgt_lang):
    db = get_db()
    db.execute(
        "INSERT INTO records VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         src_lang, tgt_lang, source_text[:4000], translated,
         st.session_state.thread_id),
    )
    db.commit()


def db_count():
    return get_db().execute("SELECT COUNT(*) FROM records").fetchone()[0]


def db_list(search="", limit=None):
    sql = ("SELECT id, timestamp, source_language, target_language, source_text, "
           "translated_text, thread_id FROM records")
    params = []
    if search:
        sql += " WHERE source_text LIKE ? OR translated_text LIKE ?"
        params = [f"%{search}%"] * 2
    sql += " ORDER BY timestamp DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    cols = ["id", "timestamp", "source_language", "target_language",
            "source_text", "translated_text", "thread_id"]
    return [dict(zip(cols, row)) for row in get_db().execute(sql, params)]


def db_update(rid, text):
    get_db().execute("UPDATE records SET translated_text=? WHERE id=?", (text, rid))
    get_db().commit()


def db_delete(rid):
    get_db().execute("DELETE FROM records WHERE id=?", (rid,))
    get_db().commit()


def db_clear():
    get_db().execute("DELETE FROM records")
    get_db().commit()


# =========================================================
# MODEL (lazy, cached, optimized)
# =========================================================
@st.cache_resource(show_spinner="Loading translation model (first run downloads ~2.5 GB)...")
def load_model(fast_cpu=True):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, low_cpu_mem_usage=True)
    model.eval()

    if device == "cuda":
        model = model.half().to(device)
    elif fast_cpu:
        try:  # int8 quantization: ~2x faster on CPU, tiny quality loss
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception:
            pass
    return tokenizer, model, device


# =========================================================
# INPUT: READ TEXT FROM UPLOADED FILES (libraries imported lazily)
# =========================================================
def decode_bytes(data):
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def join_cells(cells):
    return " | ".join(c.strip() for c in cells if c and c.strip())


class _TextCollector(HTMLParser):
    SKIP = {"script", "style", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "table"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def json_strings(node, out):
    if isinstance(node, str):
        if node.strip():
            out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            json_strings(v, out)
    elif isinstance(node, list):
        for v in node:
            json_strings(v, out)


def read_pdf(data):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)


def read_docx(data):
    from docx import Document
    doc = Document(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            lines.append(join_cells(c.text for c in row.cells))
    return "\n".join(lines)


def read_delimited(data, delimiter):
    rows = csv.reader(io.StringIO(decode_bytes(data)), delimiter=delimiter)
    return "\n".join(l for l in (join_cells(r) for r in rows) if l)


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
    lines = (l.strip() for l in "".join(parser.parts).splitlines())
    return "\n".join(l for l in lines if l)


def read_rtf(data):
    raw = decode_bytes(data)
    raw = re.sub(r"\\'([0-9a-fA-F]{2})",
                 lambda m: bytes.fromhex(m.group(1)).decode("cp1252", errors="ignore"), raw)
    raw = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), raw)
    raw = re.sub(r"\\par[d]?", "\n", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    return re.sub(r"[{}]", "", raw).strip()


def read_excel(data):
    import pandas as pd
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
                for para in shape.text_frame.paragraphs:
                    text = "".join(r.text for r in para.runs)
                    if text.strip():
                        lines.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    lines.append(join_cells(c.text for c in row.cells))
    return "\n".join(lines)


def read_opendocument(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        raw = z.read("content.xml").decode("utf-8", errors="ignore")
    raw = re.sub(r"</text:(p|h)>|<text:line-break/>", "\n", raw)
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def read_image(data):
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise RuntimeError("Image input needs OCR: pip install pytesseract pillow "
                           "and install the Tesseract OCR program.")
    return pytesseract.image_to_string(Image.open(io.BytesIO(data)))


TEXT_LIKE = ["txt", "md", "markdown", "log", "rst", "ini", "cfg", "conf",
             "yaml", "yml", "toml", "srt", "vtt", "tex"]
READERS = {
    "pdf": read_pdf, "docx": read_docx,
    "csv": lambda d: read_delimited(d, ","), "tsv": lambda d: read_delimited(d, "\t"),
    "json": read_json,
    "html": read_markup, "htm": read_markup, "xml": read_markup,
    "rtf": read_rtf, "xlsx": read_excel, "xlsm": read_excel, "pptx": read_pptx,
    "odt": read_opendocument, "ods": read_opendocument, "odp": read_opendocument,
    **{e: read_image for e in ("png", "jpg", "jpeg", "bmp", "tiff", "webp")},
}
INPUT_EXTENSIONS = TEXT_LIKE + list(READERS)


def extract_text(file):
    ext = file.name.lower().rsplit(".", 1)[-1] if "." in file.name else ""
    return READERS.get(ext, decode_bytes)(file.getvalue())


# =========================================================
# TRANSLATION
# =========================================================
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")


def split_text(text, size=400):
    """Paragraph- and sentence-aware chunking. Returns [(para_index, chunk)]."""
    chunks = []
    for p_idx, para in enumerate(text.split("\n")):
        para = para.strip()
        if not para:
            continue
        current = ""
        for sent in _SENT_SPLIT.split(para):
            if len(sent) > size:  # very long sentence: split on words
                if current:
                    chunks.append((p_idx, current))
                    current = ""
                piece = ""
                for word in sent.split():
                    if len(piece) + len(word) + 1 > size and piece:
                        chunks.append((p_idx, piece))
                        piece = word
                    else:
                        piece = f"{piece} {word}".strip()
                if piece:
                    chunks.append((p_idx, piece))
            elif len(current) + len(sent) + 1 <= size:
                current = f"{current} {sent}".strip()
            else:
                if current:
                    chunks.append((p_idx, current))
                current = sent
        if current:
            chunks.append((p_idx, current))
    return chunks


def translate(text, source_code, target_code, fast_cpu=True):
    """Returns (translated_text, [(source_chunk, translated_chunk), ...])."""
    import torch

    chunks = split_text(text)
    if not chunks:
        return "", []

    tokenizer, model, device = load_model(fast_cpu)
    tokenizer.src_lang = source_code
    target_id = tokenizer.convert_tokens_to_ids(target_code)
    batch_size = 16 if device == "cuda" else 8

    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i][1]))  # less padding
    results = [""] * len(chunks)
    progress = st.progress(0.0, text="Translating...")

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        inputs = tokenizer([chunks[i][1] for i in idx], return_tensors="pt",
                           padding=True, truncation=True, max_length=256).to(device)
        max_new = min(256, int(inputs["input_ids"].shape[1] * 1.6) + 16)
        with torch.inference_mode():
            out = model.generate(**inputs, forced_bos_token_id=target_id,
                                 max_new_tokens=max_new, num_beams=1, do_sample=False)
        for i, t in zip(idx, tokenizer.batch_decode(out, skip_special_tokens=True)):
            results[i] = t
        progress.progress(min(1.0, (start + batch_size) / len(order)),
                          text=f"Translating {min(start + batch_size, len(order))}/{len(order)}")
    progress.empty()

    paragraphs = {}
    for (p_idx, _), t in zip(chunks, results):
        paragraphs.setdefault(p_idx, []).append(t)
    translated = "\n".join(" ".join(v) for _, v in sorted(paragraphs.items()))
    return translated, [(c[1], t) for c, t in zip(chunks, results)]


# =========================================================
# OUTPUT BUILDERS (built only on demand)
# =========================================================
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(v):
    return _CONTROL.sub("", v or "")


def create_txt(c):
    return c["text"].encode("utf-8")


def create_md(c):
    return (f"# Translation: {c['source_language']} → {c['target_language']}\n\n"
            f"_Generated {c['timestamp']}_\n\n{c['text']}").encode("utf-8")


def create_html(c):
    s, t = html.escape(c["source_language"]), html.escape(c["target_language"])
    paras = "\n".join(f"<p>{html.escape(p)}</p>" for p in c["text"].split("\n") if p.strip())
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Translation: {s} to {t}</title>
<style>body {{ font-family:'Noto Sans','Noto Sans Tamil','Noto Sans Devanagari',
'Noto Sans Telugu','Noto Sans Kannada','Noto Sans Malayalam',Arial,sans-serif;
max-width:800px;margin:2rem auto;line-height:1.7; }}
.meta {{ color:#666;font-size:.9rem; }}</style></head>
<body><h1>{s} → {t}</h1><p class="meta">Generated {html.escape(c['timestamp'])}</p>
{paras}</body></html>""".encode("utf-8")


def create_csv(c):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["segment", "source_language", "target_language", "source_text", "translated_text"])
    for i, (s, t) in enumerate(c["pairs"], 1):
        w.writerow([i, c["source_language"], c["target_language"], s, t])
    return buf.getvalue().encode("utf-8-sig")


def create_json(c):
    return json.dumps({
        "source_language": c["source_language"], "target_language": c["target_language"],
        "timestamp": c["timestamp"], "source_text": c["source_text"],
        "translated_text": c["text"],
        "segments": [{"index": i, "source": s, "translation": t}
                     for i, (s, t) in enumerate(c["pairs"], 1)],
    }, ensure_ascii=False, indent=2).encode("utf-8")


def create_xml(c):
    root = ET.Element("translation", source_language=c["source_language"],
                      target_language=c["target_language"], timestamp=c["timestamp"])
    for i, (s, t) in enumerate(c["pairs"], 1):
        seg = ET.SubElement(root, "segment", index=str(i))
        ET.SubElement(seg, "source").text = s
        ET.SubElement(seg, "target").text = t
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
            if code > 0xFFFF:
                code -= 0x10000
                units = [0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)]
            out.extend(f"\\u{u - 65536 if u > 32767 else u}?" for u in units)
    return "".join(out)


def create_rtf(c):
    return ("{\\rtf1\\ansi\\deff0\\uc1\n" + _rtf_escape(c["text"]) + "\n}").encode("ascii")


def create_docx(c):
    from docx import Document
    doc = Document()
    for p in c["text"].split("\n"):
        doc.add_paragraph(p)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def create_xlsx(c):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Translation"
    ws.append(["Segment", "Source Language", "Target Language", "Source Text", "Translated Text"])
    for i, (s, t) in enumerate(c["pairs"], 1):
        ws.append([i, c["source_language"], c["target_language"], s, t])
    for col, width in zip("ABCDE", (10, 18, 18, 60, 60)):
        ws.column_dimensions[col].width = width
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def create_pptx(c):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    for _, translated in c["pairs"]:
        slide = prs.slides.add_slide(blank)
        box = slide.shapes.add_textbox(Inches(0.6), Inches(0.5), Inches(12.1), Inches(6.5))
        frame = box.text_frame
        frame.word_wrap = True
        frame.text = translated
        for para in frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(18)
    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()


def create_pdf(c):
    from fpdf import FPDF
    if not LATIN_FONT.exists():
        return None
    script_file = SCRIPT_FONTS.get(c["target_language"])
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
    except Exception:
        pass
    pdf.multi_cell(0, 8, c["text"], new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


OUTPUT_FORMATS = {
    "txt": ("📄 TXT", "text/plain", create_txt),
    "pdf": ("📕 PDF", "application/pdf", create_pdf),
    "docx": ("📝 DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", create_docx),
    "csv": ("📊 CSV", "text/csv", create_csv),
    "json": ("🧾 JSON", "application/json", create_json),
    "md": ("📘 Markdown", "text/markdown", create_md),
    "html": ("🌐 HTML", "text/html", create_html),
    "xml": ("🧬 XML", "application/xml", create_xml),
    "rtf": ("📃 RTF", "application/rtf", create_rtf),
    "xlsx": ("📗 XLSX", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", create_xlsx),
    "pptx": ("📙 PPTX", "application/vnd.openxmlformats-officedocument.presentationml.presentation", create_pptx),
}


@st.cache_data(show_spinner=False, max_entries=16)
def build_output(ext, text, pairs, source_text, source_name, target_name, stamp):
    ctx = {
        "text": clean(text), "pairs": [(clean(s), clean(t)) for s, t in pairs],
        "source_text": clean(source_text), "source_language": source_name,
        "target_language": target_name, "timestamp": stamp,
    }
    try:
        return OUTPUT_FORMATS[ext][2](ctx)
    except Exception:
        return None


# =========================================================
# FONT DOWNLOAD
# =========================================================
def download_font(filename):
    dest = BASE_DIR / filename
    if dest.exists():
        return True, "already available"
    family, error = filename.rsplit("-", 1)[0], ""
    for base in FONT_BASE_URLS:
        url = f"{base}/{family}/hinted/ttf/{filename}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
        except Exception as exc:
            error = str(exc)
            continue
        if len(data) > 10000 and data[:4] in (b"\x00\x01\x00\x00", b"true"):
            dest.write_bytes(data)
            return True, "downloaded"
        error = "Downloaded file is not a valid TTF"
    return False, f"failed: {error}"


def download_fonts():
    return [(n, *download_font(n)) for n in [LATIN_FONT.name, *SCRIPT_FONTS.values()]]


# =========================================================
# PAGES
# =========================================================
def page_translator(fast_cpu):
    st.title("🌐 AI Translator")
    st.caption("Translate text and documents using AI")

    c1, c2 = st.columns(2)
    source_name = c1.selectbox("Source Language", list(LANGUAGES))
    target_name = c2.selectbox("Target Language", list(LANGUAGES), index=1)

    text = st.text_area("Enter text", height=180, placeholder="Type or paste your text here...")
    file = st.file_uploader("Upload a file (any supported type, see Guide)", type=INPUT_EXTENSIONS)

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

        translated, pairs = translate(text, LANGUAGES[source_name], LANGUAGES[target_name], fast_cpu)
        db_add(text, translated, source_name, target_name)
        st.session_state.last_result = {
            "text": translated, "pairs": pairs, "source_text": text,
            "source": source_name, "target": target_name,
            "stamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    r = st.session_state.get("last_result")
    if not r:
        return

    st.subheader("✅ Translated Text")
    st.text_area("Output", r["text"], height=300)

    st.subheader("⬇️ Download as")
    ext = st.selectbox("Format", list(OUTPUT_FORMATS), format_func=lambda e: OUTPUT_FORMATS[e][0])
    data = build_output(ext, r["text"], tuple(r["pairs"]), r["source_text"],
                        r["source"], r["target"], r["stamp"])
    if data:
        label, mime, _ = OUTPUT_FORMATS[ext]
        st.download_button(f"{label} (.{ext})", data, file_name=f"translated_{r['target']}.{ext}",
                           mime=mime, type="primary", use_container_width=True)
    else:
        st.warning("This format couldn't be built (for PDF, download fonts on the Guide page).")


def page_history():
    import pandas as pd

    st.title("📚 Translation History")
    search = st.text_input("🔎 Search translation history", placeholder="Search saved translations...")
    records = db_list(search, limit=50)

    c1, c2 = st.columns(2)
    c1.metric("Total Records", db_count())
    c2.metric("Shown (latest 50)", len(records))
    st.divider()

    if not records:
        st.info("No translation records found.")

    for rec in records:
        with st.container(border=True):
            head, stamp = st.columns([3, 1])
            head.markdown(f"### 🌐 {rec['source_language']} → {rec['target_language']}")
            stamp.caption(rec["timestamp"])
            left, right = st.columns(2)
            left.markdown("**Original Text**")
            left.write(rec["source_text"])
            right.markdown("**Translated Text**")
            right.write(rec["translated_text"])
            st.caption("🧵 Thread ID: " + rec["thread_id"])

            with st.expander("✏️ Edit / 🗑️ Delete"):
                new_text = st.text_area("Edit translated text", rec["translated_text"],
                                        key=f"edit_{rec['id']}", height=150)
                b1, b2 = st.columns(2)
                if b1.button("💾 Save changes", key=f"save_{rec['id']}"):
                    db_update(rec["id"], new_text)
                    st.rerun()
                if b2.button("🗑️ Delete record", key=f"del_{rec['id']}"):
                    db_delete(rec["id"])
                    st.rerun()

    if db_count() == 0:
        return

    st.divider()
    rows = db_list()
    cols = ["timestamp", "source_language", "target_language", "source_text",
            "translated_text", "thread_id"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow([r[c] for c in cols])
    h_csv = buf.getvalue().encode("utf-8-sig")
    h_json = json.dumps([{c: r[c] for c in cols} for r in rows],
                        ensure_ascii=False, indent=2).encode("utf-8")
    h_txt = "\n\n".join(
        f"[{r['timestamp']}] {r['source_language']} → {r['target_language']}\n"
        f"{r['source_text']}\n---\n{r['translated_text']}" for r in rows).encode("utf-8")

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Export (CSV)", h_csv, "translation_history.csv", "text/csv",
                       use_container_width=True)
    d2.download_button("⬇️ Export (JSON)", h_json, "translation_history.json",
                       "application/json", use_container_width=True)
    d3.download_button("⬇️ Export (TXT)", h_txt, "translation_history.txt", "text/plain",
                       use_container_width=True)

    confirm = st.checkbox("I understand this will permanently delete all records")
    if st.button("🗑️ Clear All History", disabled=not confirm):
        db_clear()
        st.rerun()


def page_usage():
    import pandas as pd

    st.title("📊 Usage Dashboard")
    rows = db_list()
    if not rows:
        st.info("No usage data yet. Translate something and it will appear here.")
        return

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
    df["characters"] = df["source_text"].str.len()
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
    pairs = df["pair"].value_counts().rename_axis("Language pair").reset_index(name="Translations")
    pairs["Share"] = pairs["Translations"] / pairs["Translations"].sum() * 100
    st.dataframe(pairs, hide_index=True, use_container_width=True, column_config={
        "Share": st.column_config.ProgressColumn("Share", format="%.0f%%", min_value=0, max_value=100)})


def pkg_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def page_guide():
    import pandas as pd

    st.title("ℹ️ Requirements & Guide")
    packages = [
        ("streamlit", "Web interface"), ("torch", "Runs translation model"),
        ("transformers", "Loads NLLB-200"), ("sentencepiece", "Tokenizer"),
        ("pypdf", "Reads PDF"), ("python-docx", "Reads and creates DOCX"),
        ("fpdf2", "Creates PDF"), ("uharfbuzz", "Shapes Indian scripts"),
        ("pandas", "Reads Excel, usage dashboard"), ("openpyxl", "Reads and creates XLSX"),
        ("python-pptx", "Reads and creates PPTX"),
    ]
    pkg_rows = []
    for name, purpose in packages:
        v = pkg_version(name)
        pkg_rows.append({"Package": name, "Used for": purpose,
                         "Status": "✅ Installed" if v else "❌ Missing", "Version": v or "-"})
    font_rows = [{"Language": lang, "Font file": p.name,
                  "Status": "✅ Found" if p.exists() else "❌ Missing"}
                 for lang, p in [("Latin", LATIN_FONT)] + [(l, BASE_DIR / f) for l, f in SCRIPT_FONTS.items()]]

    all_rows = pkg_rows + font_rows
    ready = sum(r["Status"].startswith("✅") for r in all_rows)
    st.progress(ready / len(all_rows), text=f"{ready} of {len(all_rows)} requirements ready")
    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Python packages")
        st.dataframe(pd.DataFrame(pkg_rows), hide_index=True, use_container_width=True)
        st.code("pip install streamlit torch transformers sentencepiece pypdf python-docx "
                "fpdf2 uharfbuzz pandas openpyxl python-pptx", language="bash")
        st.caption("Optional OCR: pip install pytesseract pillow + Tesseract program.")
    with right:
        st.subheader("PDF Fonts")
        st.dataframe(pd.DataFrame(font_rows), hide_index=True, use_container_width=True)
        missing = any(r["Status"].startswith("❌") for r in font_rows)
        if st.button("⬇️ Download missing fonts", type="primary", disabled=not missing,
                     use_container_width=True):
            with st.spinner("Downloading fonts..."):
                st.session_state.font_results = download_fonts()
            st.rerun()
        for fn, ok, msg in st.session_state.pop("font_results", []):
            (st.success if ok else st.error)(f"{fn}: {msg}")

    st.divider()
    st.subheader("Supported languages")
    st.write(", ".join(LANGUAGES))
    st.subheader("Supported input files")
    st.write(", ".join(f".{e}" for e in INPUT_EXTENSIONS))
    st.subheader("Supported output files")
    st.write(", ".join(f".{e}" for e in OUTPUT_FORMATS))
    st.info("Scanned-image PDFs need OCR because pypdf only extracts existing text.")


# =========================================================
# SIDEBAR + ROUTING (only the selected page runs)
# =========================================================
st.sidebar.title("🌐 AI Translator")
page = st.sidebar.radio("Navigation", ["🏠 Translator", "📚 History", "📊 Usage", "ℹ️ Guide"])
fast_cpu = st.sidebar.toggle("⚡ Fast CPU mode (int8)", value=True,
                             help="~2x faster on CPU with a tiny quality loss. Ignored on GPU.")

st.sidebar.divider()
st.sidebar.write("**Current Thread**")
st.sidebar.code(st.session_state.thread_id)
if st.sidebar.button("🆕 New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pop("last_result", None)
    st.rerun()

if page.endswith("Translator"):
    page_translator(fast_cpu)
elif page.endswith("History"):
    page_history()
elif page.endswith("Usage"):
    page_usage()
else:
    page_guide()