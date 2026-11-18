"""Fine-tune TrOCR on IAM line handwriting data. Run: python train.py"""
import os, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from jiwer import cer, wer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-base-handwritten"
OUTPUT_DIR = Path("models/handwriting_trained_model")
EPOCHS = int(os.getenv("EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
ACCUM_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
MAX_TEXT_LENGTH = 128
TRAIN_LIMIT = int(os.environ["TRAIN_LIMIT"]) if os.getenv("TRAIN_LIMIT") else None
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "1000"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

class IAMLineDataset(Dataset):
    def __init__(self, split, processor): self.split, self.processor = split, processor
    def __len__(self): return len(self.split)
    def __getitem__(self, idx):
        item = self.split[idx]
        image = item["image"].convert("RGB")
        text = str(item["text"]).strip()
        pixels = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(text, max_length=MAX_TEXT_LENGTH, padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixels, "labels": labels}

def collate(items):
    return {"pixel_values": torch.stack([x["pixel_values"] for x in items]), "labels": torch.stack([x["labels"] for x in items])}

def save_checkpoint(model, processor):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Generation options are stored in generation_config, not model.config.
    model.config.max_length = None
    model.config.num_beams = None
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    model.generation_config.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

def main():
    print("Device:", DEVICE)
    if DEVICE.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    else: print("WARNING: CPU training can be very slow; CUDA GPU is recommended.")

    print("Loading pretrained TrOCR...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.to(DEVICE)

    print("Loading IAM-line dataset...")
    ds = load_dataset("Teklia/IAM-line")
    train, val, test = ds["train"], ds["validation"], ds["test"]
    if TRAIN_LIMIT is not None: train = train.select(range(min(TRAIN_LIMIT, len(train))))
    if EVAL_LIMIT:
        val = val.select(range(min(EVAL_LIMIT, len(val))))
        test = test.select(range(min(EVAL_LIMIT, len(test))))
    print(f"Train: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}")

    train_loader = DataLoader(IAMLineDataset(train, processor), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate, pin_memory=USE_AMP)
    val_loader = DataLoader(IAMLineDataset(val, processor), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    history = {"train_loss": [], "validation_loss": []}

    for epoch in range(EPOCHS):
        model.train(); total = 0.0; optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(bar):
            pixels = batch["pixel_values"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                loss = model(pixel_values=pixels, labels=labels).loss
                scaled_loss = loss / ACCUM_STEPS
            scaler.scale(scaled_loss).backward()
            if (step + 1) % ACCUM_STEPS == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total += loss.item(); bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = total / max(1, len(train_loader))

        model.eval(); val_total = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixels = batch["pixel_values"].to(DEVICE); labels = batch["labels"].to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                    val_total += model(pixel_values=pixels, labels=labels).loss.item()
        val_loss = val_total / max(1, len(val_loader))
        history["train_loss"].append(train_loss); history["validation_loss"].append(val_loss)
        print(f"Epoch {epoch+1}: Training Loss={train_loss:.4f} | Validation Loss={val_loss:.4f}")
        save_checkpoint(model, processor)
        print("Checkpoint saved:", OUTPUT_DIR.resolve())

    print("Evaluating on IAM test split...")
    test_loader = DataLoader(IAMLineDataset(test, processor), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)
    refs, preds = [], []; model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            pixels = batch["pixel_values"].to(DEVICE)
            out = model.generate(pixels, max_new_tokens=MAX_TEXT_LENGTH, num_beams=4)
            preds.extend(x.strip() for x in processor.batch_decode(out, skip_special_tokens=True))
            labels = batch["labels"].clone(); labels[labels == -100] = processor.tokenizer.pad_token_id
            refs.extend(x.strip() for x in processor.batch_decode(labels, skip_special_tokens=True))
    c_err, w_err = float(cer(refs, preds)), float(wer(refs, preds))
    metrics = {"character_error_rate": c_err, "character_accuracy_estimate_percent": max(0, (1-c_err)*100), "word_error_rate": w_err, "word_accuracy_estimate_percent": max(0, (1-w_err)*100), "train_samples": len(train), "validation_samples": len(val), "test_samples": len(test), "epochs": EPOCHS}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("\nTRAINING COMPLETE")
    print(f"Character error rate: {c_err*100:.2f}% | Character accuracy estimate: {metrics['character_accuracy_estimate_percent']:.2f}%")
    print(f"Word error rate: {w_err*100:.2f}% | Word accuracy estimate: {metrics['word_accuracy_estimate_percent']:.2f}%")
    print("Model folder:", OUTPUT_DIR.resolve())
    print("Start the website with: streamlit run app.py")

if __name__ == "__main__": main()
