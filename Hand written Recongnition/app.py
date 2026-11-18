import io
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(page_title="InkSight AI | Handwriting Recognition", page_icon="✍️", layout="wide")
MODEL_DIR = Path("models/handwriting_trained_model")
MAX_PIXELS = 25_000_000

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:#0A0A0A;color:#F5F7FA}
[data-testid="stHeader"]{background:rgba(10,10,10,.9)}
.hero{padding:clamp(1.4rem,4vw,3rem);border:1px solid rgba(0,123,167,.5);border-radius:24px;margin:.5rem 0 1.5rem;background:radial-gradient(circle at 88% 15%,rgba(255,45,149,.22),transparent 28%),radial-gradient(circle at 4% 95%,rgba(0,123,167,.3),transparent 35%),linear-gradient(135deg,#15191D,#0A0A0A 75%)}
.eyebrow{color:#FF2D95;text-transform:uppercase;letter-spacing:.15em;font-size:.76rem;font-weight:800}
.hero h1{color:white;font-size:clamp(2rem,5vw,3.6rem);line-height:1.08;font-weight:900;margin:.7rem 0}
.hero p{color:#C7D1D8;max-width:780px;line-height:1.7}
.panel,.metric{background:#12191E;border:1px solid rgba(0,123,167,.4);border-radius:16px;padding:1rem 1.1rem;margin:.5rem 0}
.metric{border-color:rgba(255,45,149,.42);min-height:90px}.metric-label{color:#9EABB4;font-size:.8rem}.metric-value{color:white;font-size:1.55rem;font-weight:800}
.accent{color:#FF2D95}.blue{color:#45B9E4}.small{color:#9EABB4;font-size:.82rem}
div.stButton>button,div.stDownloadButton>button{background:#FF2D95;color:white;border:0;border-radius:10px;font-weight:800;min-height:2.8rem}
div.stButton>button:hover,div.stDownloadButton>button:hover{background:#E91E83;color:white;border:0}
[data-testid="stFileUploader"]{background:#11171B;border:1px dashed #007BA7;border-radius:15px;padding:1rem}
</style>
<div class="hero"><div class="eyebrow">AI-powered optical character recognition</div>
<h1>AI-Based Handwritten<br><span class="accent">English Recognition</span></h1>
<p>Convert handwritten alphabets, words, sentences and paragraphs into editable text. Upload an image and let the fine-tuned TrOCR model read the handwriting line by line.</p></div>
""", unsafe_allow_html=True)

c1,c2,c3=st.columns(3)
with c1: st.markdown('<div class="panel"><span class="blue">01</span><br><b>Alphabets & words</b><br><span class="small">Read individual letters and words.</span></div>',unsafe_allow_html=True)
with c2: st.markdown('<div class="panel"><span class="blue">02</span><br><b>Sentences</b><br><span class="small">Recognize handwritten English sentences.</span></div>',unsafe_allow_html=True)
with c3: st.markdown('<div class="panel"><span class="accent">03</span><br><b>Paragraphs</b><br><span class="small">Extract multi-line handwritten text.</span></div>',unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained TrOCR model...")
def load_model(path):
    processor=TrOCRProcessor.from_pretrained(path)
    model=VisionEncoderDecoderModel.from_pretrained(path)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return processor,model,device

def preprocess(pil):
    pil=ImageOps.exif_transpose(pil).convert("RGB")
    gray=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2GRAY)
    gray=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    return pil,cv2.bilateralFilter(gray,5,35,35)

def detect_lines(gray):
    h,w=gray.shape
    if h<20 or w<20:return [gray]
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    joined=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(15,3)))
    active=np.where(np.sum(joined>0,axis=1)>max(2,int(w*.002)))[0]
    if not len(active):return [gray]
    gap=max(8,int(h*.006)); groups=[]; start=prev=int(active[0])
    for y in active[1:]:
        y=int(y)
        if y-prev>gap:groups.append((start,prev));start=y
        prev=y
    groups.append((start,prev)); merged=[]
    for top,bottom in groups:
        if merged and top-merged[-1][1]<max(10,int(h*.008)):merged[-1]=(merged[-1][0],bottom)
        else:merged.append((top,bottom))
    lines=[]
    for top,bottom in merged:
        if bottom-top<5:continue
        py=max(10,int((bottom-top)*.22)); y1=max(0,top-py); y2=min(h,bottom+py+1)
        crop=gray[y1:y2,:]; ink=np.where(crop<245)
        if len(ink[1])==0:continue
        px=max(12,int(w*.01)); x1=max(0,int(ink[1].min())-px); x2=min(w,int(ink[1].max())+px+1)
        line=crop[:,x1:x2]
        if line.shape[0]>=8 and line.shape[1]>=20:lines.append(line)
    if not lines or (len(lines)>1 and sum(x.shape[0] for x in lines)<h*.12):return [gray]
    return lines

def word_confidence(model,processor,out,text):
    words=text.split()
    if not words or not out.scores:return []
    seq=out.sequences
    try:
        trans=model.compute_transition_scores(seq,out.scores,getattr(out,"beam_indices",None),normalize_logits=True)[0]
        token_probs=trans.float().exp().detach().cpu().tolist()
    except Exception:
        token_probs=[]
        n=min(len(out.scores),max(0,seq.shape[1]-1))
        for i in range(n):
            tid=int(seq[0,i+1]); token_probs.append(float(torch.softmax(out.scores[i][0].float(),-1)[tid]))
    ids=seq[0,1:1+len(token_probs)].tolist(); groups=[]; current=[]; tok=processor.tokenizer
    for tid,prob in zip(ids,token_probs):
        if tid in tok.all_special_ids:continue
        piece=tok.convert_ids_to_tokens(int(tid))
        if piece.startswith(("Ġ","▁")) and current:
            groups.append(float(np.mean(current))*100);current=[]
        current.append(prob)
    if current:groups.append(float(np.mean(current))*100)
    # Avoid incorrect word-to-score alignment if tokenizer pieces do not match whitespace words.
    if len(groups)!=len(words):
        fallback=float(np.mean(token_probs))*100 if token_probs else None
        return [round(fallback,2) if fallback is not None else None for _ in words]
    return [round(float(np.clip(x,0,100)),2) for x in groups]

def recognize(line,model,processor,device):
    rgb=cv2.cvtColor(line,cv2.COLOR_GRAY2RGB)
    pixels=processor(images=Image.fromarray(rgb),return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        out=model.generate(pixels,max_new_tokens=128,num_beams=4,return_dict_in_generate=True,output_scores=True)
    text=processor.batch_decode(out.sequences,skip_special_tokens=True)[0].strip()
    return text,word_confidence(model,processor,out,text)

if not MODEL_DIR.exists() or not (MODEL_DIR/"config.json").exists():
    st.error("Trained model not found. Run **python train.py** in the project terminal first. The model will be saved in models/handwriting_trained_model.")
    st.stop()
try: processor,model,device=load_model(str(MODEL_DIR))
except Exception as e: st.error(f"Could not load model: {e}");st.stop()
st.caption(f"Trained model loaded • Device: {device.type.upper()}")
st.markdown("## Upload your handwriting")
st.write("Supported formats: PNG, JPG, JPEG and WEBP. Use a clear, well-lit image with dark writing on a light background.")
up=st.file_uploader("Choose a handwritten image",type=["png","jpg","jpeg","webp"])
if up:
    try:
        raw=up.getvalue()
        if len(raw)>20*1024*1024:st.error("Image exceeds 20 MB.");st.stop()
        pil=Image.open(io.BytesIO(raw))
        if pil.width*pil.height>MAX_PIXELS:st.error("Image dimensions are too large. Resize and upload again.");st.stop()
        original,enhanced=preprocess(pil)
        a,b=st.columns(2)
        with a:st.markdown("### Original image");st.image(original,use_container_width=True)
        with b:st.markdown("### Enhanced image");st.image(enhanced,use_container_width=True,clamp=True)
        if st.button("✦ Recognize handwriting",use_container_width=True):
            lines=detect_lines(enhanced);texts=[];conf_lines=[];bar=st.progress(0,text="Recognizing handwriting...")
            for i,line in enumerate(lines):
                text,conf=recognize(line,model,processor,device);texts.append(text);conf_lines.append(conf)
                bar.progress((i+1)/len(lines),text=f"Recognizing line {i+1} of {len(lines)}")
            bar.empty(); final="\n".join(x for x in texts if x).strip(); rows=[]
            for ln,(line,conf) in enumerate(zip(texts,conf_lines),1):
                for wi,word in enumerate(line.split()):
                    value=conf[wi] if wi<len(conf) else None
                    rows.append({"Line":ln,"Word":word,"Estimated confidence":f"{value:.2f}%" if value is not None else "N/A"})
            df=pd.DataFrame(rows,columns=["Line","Word","Estimated confidence"])
            vals=[v for line in conf_lines for v in line if v is not None and np.isfinite(v)]
            st.session_state.ocr={"text":final,"table":df,"confidence":float(np.mean(vals)) if vals else None,"lines":len(lines)}
    except Exception as e:st.error(f"Image processing failed: {e}");st.exception(e)

if "ocr" in st.session_state:
    result=st.session_state.ocr
    st.markdown("---\n## Recognition results")
    x,y,z=st.columns(3)
    with x:st.markdown(f'<div class="metric"><div class="metric-label">Detected lines</div><div class="metric-value">{result["lines"]}</div></div>',unsafe_allow_html=True)
    with y:st.markdown(f'<div class="metric"><div class="metric-label">Recognized words</div><div class="metric-value">{len(result["text"].split())}</div></div>',unsafe_allow_html=True)
    with z:
        cv=f'{result["confidence"]:.2f}%' if result["confidence"] is not None else 'N/A'
        st.markdown(f'<div class="metric"><div class="metric-label">Estimated confidence</div><div class="metric-value">{cv}</div></div>',unsafe_allow_html=True)
    st.markdown("### Extracted handwritten text")
    edited=st.text_area("Recognized text — editable",value=result["text"],height=220)
    st.markdown("### Word-by-word confidence")
    st.dataframe(result["table"],use_container_width=True,hide_index=True)
    st.download_button("⬇ Download recognized text (.txt)",edited.encode("utf-8"),file_name="recognized_handwriting.txt",mime="text/plain",use_container_width=True)
    st.download_button("⬇ Download confidence report (.csv)",result["table"].to_csv(index=False).encode("utf-8"),file_name="word_confidence_report.csv",mime="text/csv",use_container_width=True)
    st.info("Confidence is estimated from generated-token probabilities; it is not verified accuracy. True accuracy requires comparing the OCR output with the correct transcription.")

st.markdown('<hr><div style="text-align:center;padding:1rem;color:#9EABB4"><b style="color:#45B9E4">INKSIGHT AI</b> <span style="color:#FF2D95">•</span> Handwritten English Recognition<br><span style="font-size:.78rem">Powered by TrOCR fine-tuned on IAM handwriting data</span></div>',unsafe_allow_html=True)
