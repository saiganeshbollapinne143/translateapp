import importlib, io, json, os, re, tempfile, time, urllib.request, uuid
from datetime import datetime
import streamlit as st
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF

# AIT GLOBAL TECHNOLOGIES — chunk-streaming translator for long documents
MODEL_NAME = "olob0/nllb-200-distilled-600M-ct2-int8_float16"
CHUNK_CHARS = 700                 # small chunks keep model input within limits
MAX_INPUT_TOKENS = 256
LANGUAGES = {"English":"eng_Latn", "Hindi":"hin_Deva", "Telugu":"tel_Telu", "Tamil":"tam_Taml", "Kannada":"kan_Knda", "Malayalam":"mal_Mlym", "Bengali":"ben_Beng", "Marathi":"mar_Deva", "Urdu":"urd_Arab", "French":"fra_Latn", "German":"deu_Latn", "Spanish":"spa_Latn", "Italian":"ita_Latn", "Arabic":"arb_Arab", "Chinese (Simplified)":"zho_Hans", "Japanese":"jpn_Jpan"}

st.set_page_config(page_title="AIT Global Technologies Translator", page_icon="🌍", layout="centered")
st.markdown("""<style>
.stApp{background:#fff;color:#202124}.block-container{max-width:900px;padding-top:2rem}
.brand{background:#0b2a5b;border-radius:12px;text-align:center;padding:18px;margin-bottom:18px}
.brand h1{font-size:27px;margin:0;color:#fff}.brand span{color:#ffc107}
.stButton>button,.stDownloadButton>button{background:#ffc107;color:#172033;border:0;border-radius:8px;font-weight:700}
.stProgress>div>div>div>div{background:#0b2a5b}
.livebox{background:#f3f7ff;border:1px solid #c8d8f4;border-left:5px solid #0b2a5b;border-radius:8px;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;min-height:100px}
</style><div class="brand"><h1><span>AIT</span> GLOBAL TECHNOLOGIES TRANSLATOR</h1></div>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Downloading/loading translation model (first run only)...")
def load_model():
    try:
        ctranslate2 = importlib.import_module("ctranslate2")
    except ImportError as exc:
        raise ImportError("CTranslate2 is required. Install it with: pip install ctranslate2") from exc
    path = snapshot_download(MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(path)
    model = ctranslate2.Translator(path, device="cpu", compute_type="int8", inter_threads=1, intra_threads=2)
    return tok, model

def read_file(file):
    data, name = file.getvalue(), file.name.lower()
    if name.endswith('.pdf'):
        return 'text', '\n\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith('.docx'):
        doc = Document(io.BytesIO(data)); return 'text', '\n'.join(p.text for p in doc.paragraphs)
    if name.endswith('.json'):
        return 'json', json.loads(data.decode('utf-8-sig'))
    return 'text', data.decode('utf-8-sig', errors='replace')

def split_text(text, limit=CHUNK_CHARS):
    # Split paragraphs/sentences first, then hard-split any unusually long sentence.
    out=[]
    for line in text.splitlines() or [text]:
        if not line.strip(): out.append(''); continue
        parts=re.split(r'(?<=[.!?।።؟])\s+', line.strip()); cur=''
        for part in parts:
            while len(part)>limit:
                if cur: out.append(cur); cur=''
                out.append(part[:limit]); part=part[limit:]
            if cur and len(cur)+1+len(part)>limit: out.append(cur); cur=part
            else: cur=(cur+' '+part).strip()
        if cur: out.append(cur)
    return out

def translate_one(text, src, tgt, on_partial):
    if not text.strip(): return text
    tok, model = load_model(); tok.src_lang=src
    ids=tok(text, truncation=True, max_length=MAX_INPUT_TOKENS).input_ids
    source=tok.convert_ids_to_tokens(ids); generated=[]
    def decode(tokens):
        tokens=[x for x in tokens if x != tgt]
        return tok.decode(tok.convert_tokens_to_ids(tokens), skip_special_tokens=True)
    def callback(step):
        if step.token != tgt: generated.append(step.token)
    try:
        future=model.translate_batch([source], target_prefix=[[tgt]], beam_size=1,
                    max_decoding_length=300, callback=callback, asynchronous=True)[0]
        last=-1
        while not future.done():
            if len(generated)!=last:
                on_partial(decode(generated)); last=len(generated)
            time.sleep(.05)
        result=decode(future.result().hypotheses[0])
        on_partial(result); return result
    except TypeError:  # Older CTranslate2 without callback support
        result=model.translate_batch([source], target_prefix=[[tgt]], beam_size=2,
                                     max_decoding_length=300)[0].hypotheses[0]
        if result and result[0]==tgt: result=result[1:]
        result=tok.decode(tok.convert_tokens_to_ids(result), skip_special_tokens=True)
        on_partial(result); return result

def translate_json(obj, fn):
    if isinstance(obj,str): return fn(obj)
    if isinstance(obj,list): return [translate_json(x,fn) for x in obj]
    if isinstance(obj,dict): return {k:translate_json(v,fn) for k,v in obj.items()}
    return obj

def make_docx(text):
    doc=Document()
    for line in text.splitlines(): doc.add_paragraph(line)
    b=io.BytesIO(); doc.save(b); return b.getvalue()

def make_pdf(text):
    pdf=FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    # Unicode font download supports most scripts; fallback avoids PDF generation failure.
    font=os.path.join(tempfile.gettempdir(),'NotoSans-Regular.ttf')
    if not os.path.exists(font):
        try:
            req=urllib.request.Request('https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf',headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=20) as r, open(font,'wb') as f: f.write(r.read())
        except Exception: pass
    unicode_ok=False
    if os.path.exists(font):
        try: pdf.add_font('Noto','',font); pdf.set_font('Noto',size=10); unicode_ok=True
        except Exception: pass
    if not unicode_ok: pdf.set_font('Helvetica',size=10)
    for line in text.splitlines():
        safe=line if unicode_ok else line.encode('latin-1','replace').decode('latin-1')
        pdf.multi_cell(pdf.epw,6,safe)
    return bytes(pdf.output())

# New thread gives a clean display and a separate session identifier.
if 'thread_id' not in st.session_state: st.session_state.thread_id='THR-'+uuid.uuid4().hex[:8].upper()
a,b=st.columns([3,1]); a.text_input('Thread ID',st.session_state.thread_id,disabled=True)
if b.button('New thread'):
    st.session_state.thread_id='THR-'+uuid.uuid4().hex[:8].upper(); st.session_state.pop('result',None); st.rerun()
c1,c2=st.columns(2); src_name=c1.selectbox('Translate from',list(LANGUAGES),index=0); tgt_name=c2.selectbox('Translate to',list(LANGUAGES),index=2)
file=st.file_uploader('Upload a document (PDF, DOCX, TXT, JSON — supports long documents, including ~100 pages)',type=['pdf','docx','txt','json'])
text_input=st.text_area('Or paste text',height=120)
st.subheader('Live chunk translation')
st.caption('Only the chunk currently being translated appears here. Earlier chunks are kept privately for the final downloadable file.')
live=st.empty(); live.markdown('<div class="livebox">Your current chunk will appear here while translating…</div>',unsafe_allow_html=True)
if st.button('Translate',type='primary',use_container_width=True):
    if src_name==tgt_name: st.warning('Choose two different languages.')
    elif not file and not text_input.strip(): st.warning('Upload a file or paste text first.')
    else:
        try:
            kind,content=read_file(file) if file else ('text',text_input)
            source_strings=[]
            if kind=='json':
                def gather(x):
                    if isinstance(x,str):
                        if x.strip(): source_strings.append(x)
                    elif isinstance(x,list):
                        for v in x: gather(v)
                    elif isinstance(x,dict):
                        for v in x.values(): gather(v)
                gather(content)
                # Keep JSON structure and translate each unique string once.
                chunks=list(dict.fromkeys(s for s in source_strings if s.strip()))
            else: chunks=[x for x in split_text(content) if x.strip()]
            if not chunks: st.error('No readable text found in the input.'); st.stop()
            total=len(chunks); bar=st.progress(0,text=f'Preparing {total:,} chunks…'); translated={}; all_lines=[]
            src,tgt=LANGUAGES[src_name],LANGUAGES[tgt_name]
            for i,chunk in enumerate(chunks,1):
                live.markdown(f'<div class="livebox"><b>Chunk {i:,} of {total:,}</b><br><br>Translating…</div>',unsafe_allow_html=True)
                def update(part, idx=i, total_chunks=total):
                    shown=part.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
                    live.markdown(f'<div class="livebox"><b>Chunk {idx:,} of {total_chunks:,}</b><hr>{shown or "Generating translation…"}</div>',unsafe_allow_html=True)
                result=translate_one(chunk,src,tgt,update); translated[chunk]=result; all_lines.append(result)
                bar.progress(i/total,text=f'Translated {i:,} / {total:,} chunks')
            if kind=='json': final_obj=translate_json(content,lambda s:translated.get(s,s)); final_text=json.dumps(final_obj,ensure_ascii=False,indent=2)
            else: final_text='\n'.join(all_lines)
            st.session_state.result={'text':final_text,'json':final_obj if kind=='json' else {'thread_id':st.session_state.thread_id,'source_language':src_name,'target_language':tgt_name,'translated_text':final_text},'target':tgt_name}
            bar.empty(); live.markdown(f'<div class="livebox"><b>Translation complete</b><br>{total:,} chunks processed. The complete result is ready to download; only the active chunk was shown during translation.</div>',unsafe_allow_html=True)
            st.success('Translation complete. Download the full translated document below.')
        except Exception as e: st.error(f'Translation failed: {e}')

result=st.session_state.get('result')
if result:
    st.subheader('Download complete translation')
    cols=st.columns(4)
    try: cols[0].download_button('📄 PDF',make_pdf(result['text']),'translation.pdf','application/pdf',use_container_width=True)
    except Exception as e: cols[0].button('PDF unavailable',disabled=True,use_container_width=True); st.caption(f'PDF note: {e}')
    cols[1].download_button('📝 TXT',result['text'].encode('utf-8'),'translation.txt','text/plain',use_container_width=True)
    cols[2].download_button('📑 DOCX',make_docx(result['text']),'translation.docx','application/vnd.openxmlformats-officedocument.wordprocessingml.document',use_container_width=True)
    cols[3].download_button('📋 JSON',json.dumps(result['json'],ensure_ascii=False,indent=2).encode('utf-8'),'translation.json','application/json',use_container_width=True)
