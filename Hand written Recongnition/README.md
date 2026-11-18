# InkSight AI — VS Code project

A Streamlit website for handwritten English OCR, powered by Microsoft's TrOCR fine-tuned on the IAM line handwriting dataset.

## Project files
- `train.py` — fine-tunes and evaluates the model; run once.
- `app.py` — responsive upload and recognition website.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — dark theme configuration.
- `models/handwriting_trained_model/` — created automatically after training; do not delete it.

## Setup (Windows PowerShell)
Use Python 3.10 or 3.11 recommended. Open this folder in VS Code and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install a CUDA-enabled PyTorch build if you have an NVIDIA GPU and the default installation does not detect it. Use https://pytorch.org/get-started/locally/ for the matching command.

## Train
Quick smoke test (500 examples, one epoch):
```powershell
$env:TRAIN_LIMIT="500"
$env:EPOCHS="1"
python train.py
```
Full fine-tuning (complete IAM training split, three epochs):
```powershell
Remove-Item Env:TRAIN_LIMIT -ErrorAction SilentlyContinue
$env:EPOCHS="3"
python train.py
```
If GPU memory is limited:
```powershell
$env:BATCH_SIZE="1"
$env:GRADIENT_ACCUMULATION_STEPS="8"
python train.py
```
The first run downloads the pretrained model and IAM dataset. Full training on CPU can be very slow; a CUDA GPU is strongly recommended. The trained model is saved to `models/handwriting_trained_model/`.

## Run the website
After training:
```powershell
streamlit run app.py
```
Open the local URL printed by Streamlit, normally `http://localhost:8501`. Upload a PNG/JPG/JPEG/WEBP image and click **Recognize handwriting**. You do not need to train again for each upload.

## Important accuracy notes
- TrOCR is a line-level handwriting recognition model. The app segments page images into horizontal line crops and recognizes each crop.
- IAM mainly supplies handwritten text-line examples, not a balanced isolated A–Z alphabet dataset. Isolated single-letter images may therefore be less reliable than words and text lines. For strong standalone alphabet classification, add a dedicated letter dataset and train/evaluate for that task.
- The per-word confidence shown in the app is an estimate derived from token probabilities, not a calibrated probability that the word is correct.
- Character/word accuracy is measured on the held-out IAM test set by comparing model predictions with its reference transcriptions. Accuracy for a newly uploaded image cannot be known without its correct transcription.
- Clear, straight, well-lit images with dark writing on a light background work best. Shadows, blur, ruled lines, skew and cut-off text can reduce performance.
