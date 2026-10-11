
import io, json, re, time, uuid, importlib
import streamlit as st
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF

MODEL = "olob0/nllb-200-distilled-600M-ct2-int8_float16"
LANG = {"English":"eng_Latn","Hindi":"hin_Deva","Telugu":"tel_Telu","Tamil":"tam_Taml","Kannada":"kan_Knda","Malayalam":"mal_Mlym","Bengali":"ben_Beng","Marathi":"mar_Deva","Urdu":"urd_Arab","French":"fra_Latn","German":"deu_Latn","Spanish":"spa_Latn","Italian":"ita_Latn","Arabic":"arb_Arab","Chinese (Simplified)":"zho_Hans","Japanese":"jpn_Jpan"}

st.set_page_config(page_title="Document Translator", page_icon="🌍", layout="centered")
st.markdown("""<style>
.stApp{background:#fff;color:#202124}.block-container{max-width:900px;padding-top:2rem}
.brand{background:#0b2a5b;border-radius:12px;text-align:center;padding:20px;margin-bottom:18px}
.brand h1{font-size:27px;font-weight:800;margin:0;letter-spacing:.5px}
.brand span:first-child{color:#ffc107}.brand span:last-child{color:#fff}
.stButton>button,.stDownloadButton>button{background:#ffc107;color:#172033;border:0;border-radius:8px;font-weight:700}
.stProgress>div>div>div>div{background:#0b2a5b}
.livebox{background:#f3f7ff;border:1px solid #c8d8f4;border-left:5px solid #0b2a5b;border-radius:8px;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;min-height:90px}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    ct=importlib.import_module("ctranslate2")
    path=snapshot_download(MODEL)
    tok=AutoTokenizer.from_pretrained(path)
    return tok,ct.Translator(path,device="cpu",compute_type="int8",inter_threads=1,intra_threads=2)

def read_file(f):
    b,n=f.getvalue(),f.name.lower()
    if n.endswith(".pdf"): return "text","\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(b)).pages)
    if n.endswith(".docx"): return "text","\n".join(p.text for p in Document(io.BytesIO(b)).paragraphs)
    if n.endswith(".json"): return "json",json.loads(b.decode("utf-8-sig"))
    return "text",b.decode("utf-8-sig",errors="replace")

def split_text(text,limit=700):
    out=[]
    for line in text.splitlines():
        if not line.strip(): continue
        parts=re.split(r'(?<=[.!?।؟])\s+',line.strip()); cur=""
        for p in parts:
            while len(p)>limit:
                if cur: out.append(cur); cur=""
                out.append(p[:limit]); p=p[limit:]
            if cur and len(cur)+len(p)+1>limit: out.append(cur); cur=p
            else: cur=(cur+" "+p).strip()
        if cur: out.append(cur)
    return out or ([text] if text.strip() else [])

def translate(text,src,tgt,update):
    if not text.strip(): return text
    tok,model=load_model(); tok.src_lang=src
    ids=tok(text,truncation=True,max_length=256).input_ids
    source=tok.convert_ids_to_tokens(ids); generated=[]
    def decode(ts):
        ts=[t for t in ts if t!=tgt]
        return tok.decode(tok.convert_tokens_to_ids(ts),skip_special_tokens=True)
    def callback(step):
        if step.token!=tgt: generated.append(step.token)
    try:
        future=model.translate_batch([source],target_prefix=[[tgt]],beam_size=1,max_decoding_length=300,callback=callback,asynchronous=True)[0]
        last=-1
        while not future.done():
            if len(generated)!=last: update(decode(generated)); last=len(generated)
            time.sleep(.05)
        result=decode(future.result().hypotheses[0])
    except TypeError:
        ts=model.translate_batch([source],target_prefix=[[tgt]],beam_size=2,max_decoding_length=300)[0].hypotheses[0]
        if ts and ts[0]==tgt: ts=ts[1:]
        result=tok.decode(tok.convert_tokens_to_ids(ts),skip_special_tokens=True)
    update(result); return result

def walk_json(x,fn):
    if isinstance(x,str): return fn(x) if x.strip() else x
    if isinstance(x,list): return [walk_json(v,fn) for v in x]
    if isinstance(x,dict): return {k:walk_json(v,fn) for k,v in x.items()}
    return x

def docx_bytes(text):
    d=Document()
    for line in text.splitlines(): d.add_paragraph(line)
    b=io.BytesIO(); d.save(b); return b.getvalue()

def pdf_bytes(text):
    p=FPDF(); p.set_auto_page_break(True,15); p.add_page(); p.set_font("Helvetica",size=10)
    for line in text.splitlines(): p.multi_cell(0,6,line.encode("latin-1","replace").decode("latin-1"))
    return bytes(p.output())

if "thread_id" not in st.session_state: st.session_state.thread_id="THR-"+uuid.uuid4().hex[:8].upper()
a,b=st.columns([3,1]); a.text_input("Thread ID",st.session_state.thread_id,disabled=True)
if b.button("New thread"):
    st.session_state.thread_id="THR-"+uuid.uuid4().hex[:8].upper()
    st.session_state.pop("result",None); st.rerun()

src_col,tgt_col=st.columns(2)
src_name=src_col.selectbox("Translate from",list(LANG),index=0)
tgt_name=tgt_col.selectbox("Translate to",list(LANG),index=2)
file=st.file_uploader("Upload PDF, DOCX, TXT or JSON",type=["pdf","docx","txt","json"])
text_input=st.text_area("Or paste text",height=100)
st.subheader("Live translation output")
live=st.empty()
live.markdown('<div class="livebox">Translation appears here…</div>',unsafe_allow_html=True)

if st.button("Translate",type="primary",use_container_width=True):
    if src_name==tgt_name: st.warning("Choose different languages.")
    elif not file and not text_input.strip(): st.warning("Upload a file or enter text.")
    else:
        try:
            kind,data=read_file(file) if file else ("text",text_input)
            chunks=[]
            if kind=="json":
                def gather(x):
                    if isinstance(x,str) and x.strip(): chunks.append(x)
                    elif isinstance(x,(list,dict)):
                        for v in (x.values() if isinstance(x,dict) else x): gather(v)
                gather(data); chunks=list(dict.fromkeys(chunks))
            else: chunks=[x for x in split_text(data) if x.strip()]
            if not chunks: st.error("No readable text found."); st.stop()
            bar=st.progress(0,text=f"Preparing {len(chunks)} chunks…"); done={}
            for i,chunk in enumerate(chunks,1):
                def update(part,idx=i):
                    safe=part.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    live.markdown(f'<div class="livebox"><b>Chunk {idx}/{len(chunks)}</b><hr>{safe or "Translating…"}</div>',unsafe_allow_html=True)
                done[chunk]=translate(chunk,LANG[src_name],LANG[tgt_name],update)
                bar.progress(i/len(chunks),text=f"Translated {i}/{len(chunks)} chunks")
            final=walk_json(data,lambda s:done.get(s,s)) if kind=="json" else "\n".join(done[c] for c in chunks)
            output=json.dumps(final,ensure_ascii=False,indent=2) if kind=="json" else final
            meta=final if kind=="json" else {"thread_id":st.session_state.thread_id,"source_language":src_name,"target_language":tgt_name,"translated_text":output}
            st.session_state.result={"text":output,"json":meta}
            bar.empty(); live.markdown('<div class="livebox"><b>Translation complete.</b></div>',unsafe_allow_html=True)
            st.success("Translation completed.")
        except Exception as e: st.error(f"Translation failed: {e}")

r=st.session_state.get("result")
if r:
    st.subheader("Download translation")
    c=st.columns(4)
    c[0].download_button("PDF",pdf_bytes(r["text"]),"translation.pdf","application/pdf",use_container_width=True)
    c[1].download_button("TXT",r["text"].encode(),"translation.txt","text/plain",use_container_width=True)
    c[2].download_button("DOCX",docx_bytes(r["text"]),"translation.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True)
    c[3].download_button("JSON",json.dumps(r["json"],ensure_ascii=False,indent=2).encode(),"translation.json","application/json",use_container_width=True)
