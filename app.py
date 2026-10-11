
import io,json,re,uuid,time,hashlib,importlib,html
import streamlit as st
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF

MODEL="olob0/nllb-200-distilled-600M-ct2-int8_float16"
LANG={"English":"eng_Latn","Hindi":"hin_Deva","Telugu":"tel_Telu","Tamil":"tam_Taml","Kannada":"kan_Knda","Malayalam":"mal_Mlym","Bengali":"ben_Beng","Marathi":"mar_Deva","Urdu":"urd_Arab","French":"fra_Latn","German":"deu_Latn","Spanish":"spa_Latn","Italian":"ita_Latn","Arabic":"arb_Arab","Chinese":"zho_Hans","Japanese":"jpn_Jpan"}

st.set_page_config(page_title="AIT GLOBAL TRANSLATOR",page_icon="🌐",layout="centered")
st.markdown("""<style>
.stApp{background:#f4f7fc;color:#18243b}.block-container{max-width:900px;padding:1rem}
.hero{background:linear-gradient(120deg,#071d43,#124b89);padding:22px;border-radius:16px;text-align:center}
.hero h1{color:white}.hero span{color:#ffc928}.hero p{color:#e0eaff}
.stButton>button,.stDownloadButton>button{background:#ffc928;color:#10213d;border:0;border-radius:9px;font-weight:bold}
.live{background:white;border:1px solid #d7e2f3;border-left:5px solid #124b89;padding:12px;border-radius:9px;white-space:pre-wrap;overflow-wrap:anywhere}
</style><div class="hero"><h1><span>AIT GLOBAL</span> TRANSLATOR</h1><p>Live Streaming · Incremental Chunk Cache · Document Downloads</p></div>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading translation model...")
def load():
    ct=importlib.import_module("ctranslate2")
    path=snapshot_download(MODEL)
    return AutoTokenizer.from_pretrained(path),ct.Translator(path,device="cpu",compute_type="int8",inter_threads=1,intra_threads=2)

def read_file(f):
    b,n=f.getvalue(),f.name.lower()
    if n.endswith(".pdf"):
        return "text","\n\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(b)).pages)
    if n.endswith(".docx"):
        return "text","\n".join(p.text for p in Document(io.BytesIO(b)).paragraphs)
    if n.endswith(".json"):
        return "json",json.loads(b.decode("utf-8-sig"))
    return "text",b.decode("utf-8-sig",errors="replace")

def split_text(text,tok,limit=300):
    chunks=[];current=""
    for sentence in re.split(r'(?<=[.!?।؟])\s+|\n+',text.strip()):
        for word in sentence.split():
            candidate=(current+" "+word).strip()
            if current and len(tok(candidate).input_ids)>limit:
                chunks.append(current);current=word
            else: current=candidate
    if current: chunks.append(current)
    return chunks

def translate_chunk(text,src,tgt,tok,model,update):
    tok.src_lang=src
    source=tok.convert_ids_to_tokens(tok(text).input_ids)
    generated=[]
    def callback(step):
        if step.token!=tgt:
            generated.append(step.token)
    def decode(tokens):
        tokens=[t for t in tokens if t!=tgt]
        return tok.decode(tok.convert_tokens_to_ids(tokens),skip_special_tokens=True)
    try:
        future=model.translate_batch([source],target_prefix=[[tgt]],beam_size=1,
            max_decoding_length=512,callback=callback,asynchronous=True)[0]
        last=-1
        while not future.done():
            if len(generated)!=last:
                update(decode(generated));last=len(generated)
            time.sleep(.05)
        result=decode(future.result().hypotheses[0])
    except (TypeError,AttributeError):
        result=decode(model.translate_batch([source],target_prefix=[[tgt]],
            beam_size=2,max_decoding_length=512)[0].hypotheses[0])
    return result.strip()

def walk_json(obj,fn):
    if isinstance(obj,str): return fn(obj) if obj.strip() else obj
    if isinstance(obj,list): return [walk_json(x,fn) for x in obj]
    if isinstance(obj,dict): return {k:walk_json(v,fn) for k,v in obj.items()}
    return obj

def docx_bytes(text):
    d=Document()
    for line in text.splitlines(): d.add_paragraph(line)
    b=io.BytesIO();d.save(b);return b.getvalue()

def pdf_bytes(text):
    p=FPDF();p.set_auto_page_break(True,15);p.add_page();p.set_font("Helvetica",size=10)
    for line in text.splitlines():
        p.multi_cell(0,6,line.encode("latin-1","replace").decode("latin-1"))
    return bytes(p.output())

if "thread" not in st.session_state: st.session_state.thread="THR-"+uuid.uuid4().hex[:8].upper()
if "cache" not in st.session_state: st.session_state.cache={}
if "result" not in st.session_state: st.session_state.result=None

a,b=st.columns([3,1])
a.text_input("Session ID",st.session_state.thread,disabled=True)
if b.button("New session",use_container_width=True):
    st.session_state.thread="THR-"+uuid.uuid4().hex[:8].upper()
    st.session_state.cache={};st.session_state.result=None;st.rerun()

a,b=st.columns(2)
src=a.selectbox("Source language",list(LANG))
tgt=b.selectbox("Target language",list(LANG),index=2)
file=st.file_uploader("Upload PDF, DOCX, TXT or JSON",type=["pdf","docx","txt","json"])
pasted=st.text_area("Or paste text",height=100)
live=st.empty();bar_slot=st.empty()

if st.button("Translate document",type="primary",use_container_width=True):
    if src==tgt: st.warning("Choose different languages.")
    elif not file and not pasted.strip(): st.warning("Upload a file or enter text.")
    else:
        try:
            kind,data=read_file(file) if file else ("text",pasted)
            tok,model=load()
            accumulated=[]

            def show_live(label,partial=""):
                # Preserve completed chunks; replace only the current chunk preview.
                content="\n\n".join(accumulated)
                if partial: content+=("\n\n" if content else "")+partial
                live.markdown('<div class="live"><b>'+html.escape(label)+
                    '</b><hr>'+html.escape(content)+'</div>',unsafe_allow_html=True)

            if kind=="json":
                strings=[]
                def gather(x):
                    if isinstance(x,str) and x.strip(): strings.append(x)
                    elif isinstance(x,dict):
                        for v in x.values(): gather(v)
                    elif isinstance(x,list):
                        for v in x: gather(v)
                gather(data);unique=list(dict.fromkeys(strings));mapped={}
                bar=bar_slot.progress(0,text="Preparing JSON...")
                for i,s in enumerate(unique,1):
                    key=hashlib.sha256((MODEL+src+tgt+s).encode()).hexdigest()
                    if key in st.session_state.cache: res=st.session_state.cache[key]
                    else:
                        res=translate_chunk(s,LANG[src],LANG[tgt],tok,model,
                            lambda part,idx=i: show_live(f"JSON field {idx}/{len(unique)}",part))
                        st.session_state.cache[key]=res
                    mapped[s]=res;accumulated.append(res)
                    show_live(f"JSON field {i}/{len(unique)}")
                    bar.progress(i/max(len(unique),1),text=f"Translated {i}/{len(unique)} fields")
                json_result=walk_json(data,lambda s:mapped.get(s,s))
                output=json.dumps(json_result,ensure_ascii=False,indent=2)
            else:
                pieces=split_text(data,tok)
                if not pieces: st.error("No readable text found.");st.stop()
                bar=bar_slot.progress(0,text=f"Translating {len(pieces)} chunks...")
                for i,piece in enumerate(pieces,1):
                    key=hashlib.sha256((MODEL+src+tgt+piece).encode()).hexdigest()
                    if key in st.session_state.cache: value=st.session_state.cache[key]
                    else:
                        value=translate_chunk(piece,LANG[src],LANG[tgt],tok,model,
                            lambda part,idx=i: show_live(f"Chunk {idx}/{len(pieces)}",part))
                        st.session_state.cache[key]=value
                    accumulated.append(value)
                    show_live(f"Chunk {i}/{len(pieces)} completed")
                    bar.progress(i/len(pieces),text=f"Translated {i}/{len(pieces)} chunks")
                output="\n\n".join(accumulated)
                json_result={"session_id":st.session_state.thread,"source_language":src,
                    "target_language":tgt,"translated_text":output}
            st.session_state.result={"text":output,"json":json_result}
            show_live("Translation complete")
            st.success("Translation completed.")
        except Exception as e: st.error(f"Translation failed: {e}")

r=st.session_state.result
if r:
    st.subheader("Download translated document")
    cols=st.columns(4)
    cols[0].download_button("📄 PDF",pdf_bytes(r["text"]),"translation.pdf","application/pdf",use_container_width=True)
    cols[1].download_button("📝 TXT",r["text"].encode("utf-8"),"translation.txt","text/plain",use_container_width=True)
    cols[2].download_button("📑 DOCX",docx_bytes(r["text"]),"translation.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True)
    cols[3].download_button("📋 JSON",json.dumps(r["json"],ensure_ascii=False,indent=2).encode("utf-8"),"translation.json","application/json",use_container_width=True)
