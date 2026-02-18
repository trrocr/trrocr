# ============================================================
# TrOCR Fine-tuning: English + Bengali (+ Mixed)
# Features:
#   - Auto-downloads Noto Sans Bengali font (Google Fonts)
#   - Visualizes dataset samples before training (no warnings)
#   - Trains on English, Bengali, and MIXED images
# ============================================================
# pip install transformers datasets pillow torch torchvision
#             evaluate accelerate jiwer sentencepiece requests
# ============================================================

import os
import random
import warnings
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from datasets import load_dataset
import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Suppress matplotlib glyph warnings (Bengali can't render in DejaVu Sans)
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
warnings.filterwarnings("ignore", message="Matplotlib currently does not support")

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
MODEL_CHECKPOINT  = "microsoft/trocr-small-handwritten"
OUTPUT_DIR        = "./trocr-finetuned"
FONT_DIR          = "./fonts"
FONT_PATH         = os.path.join(FONT_DIR, "NotoSansBengali.ttf")
VIZ_PATH          = "./dataset_preview.png"
MAX_TARGET_LENGTH = 64
BATCH_SIZE        = 4
GRAD_ACCUM        = 4        # effective batch = 16
NUM_EPOCHS        = 8
LR                = 4e-5
SEED              = 42
MAX_EN_SAMPLES    = 2000
MAX_BN_SAMPLES    = 2000
MAX_MIX_SAMPLES   = 1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# 1. AUTO-DOWNLOAD BENGALI FONT
#    Noto Sans Bengali from Google Fonts GitHub
#    (always available, perfect Bengali support)
# ─────────────────────────────────────────────
FONT_URLS = [
    # Primary: Google Fonts GitHub repo (static TTF)
    "https://github.com/google/fonts/raw/main/ofl/notosansbengali/static/NotoSansBengali-Regular.ttf",
    # Mirror 1: jsDelivr CDN
    "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/notosansbengali/static/NotoSansBengali-Regular.ttf",
    # Mirror 2: googlefonts/noto-fonts repo
    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf",
]

def download_font():
    os.makedirs(FONT_DIR, exist_ok=True)

    if os.path.exists(FONT_PATH) and os.path.getsize(FONT_PATH) > 50_000:
        print(f"  ✓ Font already exists: {FONT_PATH}")
        return True

    print("  Downloading Noto Sans Bengali font (Google Fonts)...")
    for url in FONT_URLS:
        try:
            print(f"    Trying: {url}")
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200 and len(resp.content) > 50_000:
                with open(FONT_PATH, "wb") as f:
                    f.write(resp.content)
                print(f"  ✓ Font downloaded ({len(resp.content) // 1024} KB)")
                return True
            else:
                print(f"    ✗ status={resp.status_code}, size={len(resp.content)}")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    print("\n  ⚠ Auto-download failed. Manual steps:")
    print("  1. Go to: https://fonts.google.com/noto/specimen/Noto+Sans+Bengali")
    print("  2. Click 'Download family'")
    print(f"  3. Copy 'NotoSansBengali-Regular.ttf' → {os.path.abspath(FONT_DIR)}")
    print(f"  4. Rename it to: NotoSansBengali.ttf")
    return False


# ─────────────────────────────────────────────
# 2. IMAGE RENDERER
# ─────────────────────────────────────────────
def get_font(size=32, use_bengali=False):
    if use_bengali and os.path.exists(FONT_PATH):
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def text_to_image(
    text: str,
    img_width: int = 640,
    img_height: int = 64,
    font_size: int = 28,
    use_bengali: bool = False,
    add_noise: bool = True,
) -> Image.Image:
    img  = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)
    font = get_font(size=font_size, use_bengali=use_bengali)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    x    = max(4, (img_width  - tw) // 2)
    y    = max(4, (img_height - th) // 2)
    draw.text((x, y), text, fill="black", font=font)

    if add_noise:
        arr   = np.array(img, dtype=np.int16)
        noise = np.random.randint(-8, 8, arr.shape, dtype=np.int16)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)

    return img


# ─────────────────────────────────────────────
# 3. SAMPLE GENERATORS
# ─────────────────────────────────────────────
EN_WORDS = [
    "Hello", "World", "Invoice", "Total", "Date", "Amount", "Python",
    "Network", "OCR", "Receipt", "Name", "Address", "Phone", "Email",
    "Payment", "Order", "January", "February", "March", "April",
    "Description", "Quantity", "Price", "Subtotal", "Tax", "Balance",
    "Customer", "Product", "Reference", "Code", "Number", "Report",
]

EN_NUMBERS = [
    "12345", "1,250.00", "$99.99", "2024", "#4821",
    "INV-0042", "REF:8821", "ZIP:10001", "007", "3.14",
]

BN_WORDS = [
    "আমি", "তুমি", "সে", "আমরা", "তারা", "বাংলা", "ভাষা",
    "দেশ", "মানুষ", "জীবন", "পানি", "আলো", "বায়ু", "মাটি",
    "আকাশ", "ঢাকা", "বাড়ি", "স্কুল", "বই", "কাজ", "খাবার",
    "রাস্তা", "শহর", "গ্রাম", "নদী", "পাহাড়", "ফুল", "ফল",
    "গাছ", "পাখি", "মাছ", "সময়", "বাজার", "দোকান",
    "চালান", "মোট", "তারিখ", "পরিমাণ", "মূল্য", "নাম",
]

BN_NUMBERS = ["১২৩৪", "৫৬৭৮", "৯০১২", "৩,৪৫০", "৭৮৯.০০", "২০২৪"]


def make_english_sample():
    parts = random.choices(EN_WORDS, k=random.randint(2, 5))
    if random.random() > 0.4:
        parts.append(random.choice(EN_NUMBERS))
    text = " ".join(parts)
    return text_to_image(text, use_bengali=False), text


def make_bengali_sample():
    parts = random.choices(BN_WORDS, k=random.randint(1, 4))
    if random.random() > 0.5:
        parts.append(random.choice(BN_NUMBERS))
    text = " ".join(parts)
    return text_to_image(text, use_bengali=True, img_width=700, font_size=30), text


def make_mixed_sample():
    """English + Bengali + numbers in one image — key for real invoices."""
    en_parts  = random.choices(EN_WORDS,  k=random.randint(1, 3))
    bn_parts  = random.choices(BN_WORDS,  k=random.randint(1, 3))
    num       = random.choice(EN_NUMBERS + BN_NUMBERS)
    all_parts = en_parts + bn_parts + [num]
    random.shuffle(all_parts)
    text = " ".join(all_parts)
    return text_to_image(text, use_bengali=True, img_width=750, font_size=26), text


# ─────────────────────────────────────────────
# 4. LOAD / BUILD DATASETS
# ─────────────────────────────────────────────
def load_english_samples(max_samples=2000):
    print("Loading English IAM dataset...")
    try:
        ds = load_dataset("Teklia/IAM-line", split="train")
        ds = ds.shuffle(seed=SEED).select(range(min(max_samples, len(ds))))
        samples = [(item["image"], item["text"].strip())
                   for item in ds if item["text"].strip()]
        print(f"  ✓ Loaded {len(samples)} real English samples from IAM.")
        return samples
    except Exception as e:
        print(f"  IAM failed ({e}) → generating synthetic English samples.")
    samples = [make_english_sample() for _ in range(max_samples)]
    print(f"  ✓ Generated {len(samples)} synthetic English samples.")
    return samples


def load_bengali_samples(max_samples=2000):
    print("Generating Bengali samples...")
    samples = [make_bengali_sample() for _ in range(max_samples)]
    print(f"  ✓ Generated {len(samples)} Bengali samples.")
    return samples


def load_mixed_samples(max_samples=1000):
    print("Generating MIXED (English + Bengali) samples...")
    samples = [make_mixed_sample() for _ in range(max_samples)]
    print(f"  ✓ Generated {len(samples)} mixed samples.")
    return samples


def build_splits(samples, val_ratio=0.1):
    random.shuffle(samples)
    split = int(len(samples) * (1 - val_ratio))
    return samples[:split], samples[split:]


# ─────────────────────────────────────────────
# 5. VISUALIZATION
#    PIL images show correct Bengali glyphs.
#    Matplotlib xlabel labels are ASCII-safe only
#    (avoids glyph warnings entirely).
# ─────────────────────────────────────────────
def safe_label(text: str, max_len: int = 38) -> str:
    """Make a matplotlib-safe ASCII label from any text."""
    has_bengali = any(ord(c) > 127 for c in text)
    if not has_bengali:
        return (text[:max_len] + "…") if len(text) > max_len else text
    # Keep ASCII tokens, count Bengali words separately
    ascii_parts = [w for w in text.split() if all(ord(c) < 128 for c in w)]
    bn_count    = sum(1 for w in text.split() if any(ord(c) > 127 for c in w))
    label = " ".join(ascii_parts)
    if bn_count:
        label += f" [+{bn_count} Bengali]"
    return label


def visualize_samples(en_samples, bn_samples, mix_samples, save_path=VIZ_PATH, n=6):
    """
    Save a PNG grid (dataset_preview.png).
    PIL images render correctly — Bengali glyphs visible.
    xlabels are ASCII only — zero matplotlib glyph warnings.
    """
    categories = [
        ("English / Numerical", en_samples,  "#2196F3"),
        ("Bengali",             bn_samples,  "#4CAF50"),
        ("Mixed  EN + BN",      mix_samples, "#FF9800"),
    ]

    cols = n
    rows = len(categories)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.4))
    fig.patch.set_facecolor("#1a1a2e")

    for r, (cat_name, samples, color) in enumerate(categories):
        chosen = random.sample(samples, min(n, len(samples)))
        for c in range(cols):
            ax = axes[r][c]
            ax.set_facecolor("#16213e")
            if c < len(chosen):
                img, text = chosen[c]
                if hasattr(img, "mode") and img.mode != "RGB":
                    img = img.convert("RGB")
                ax.imshow(np.array(img), aspect="auto")
                ax.set_xlabel(safe_label(text), fontsize=7, color="white", labelpad=3)
            else:
                ax.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.8)

        axes[r][0].set_ylabel(
            cat_name, fontsize=10, color=color,
            fontweight="bold", rotation=90, labelpad=8,
        )

    patches = [mpatches.Patch(color=c, label=n) for n, _, c in categories]
    fig.legend(handles=patches, loc="upper center", ncol=3, fontsize=9,
               framealpha=0.3, facecolor="#0f3460", edgecolor="white",
               labelcolor="white", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("TrOCR Training Data — Sample Preview",
                 fontsize=13, color="white", fontweight="bold", y=1.04)

    plt.tight_layout(pad=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Preview saved → {os.path.abspath(save_path)}")
    print(f"    Open it now to verify Bengali glyphs look correct!\n")


# ─────────────────────────────────────────────
# 6. TORCH DATASET
# ─────────────────────────────────────────────
class OCRDataset(Dataset):
    def __init__(self, samples, processor, max_target_length):
        self.samples           = samples
        self.processor         = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, text = self.samples[idx]
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# ─────────────────────────────────────────────
# 7. METRICS
# ─────────────────────────────────────────────
def make_compute_metrics(processor):
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        label_ids = pred.label_ids
        pred_ids  = pred.predictions
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": round(cer, 4), "wer": round(wer, 4)}

    return compute_metrics


# ─────────────────────────────────────────────
# 8. INFERENCE HELPER
# ─────────────────────────────────────────────
def run_inference(model, processor, device, img, label=""):
    model.eval()
    if hasattr(img, "mode") and img.mode != "RGB":
        img = img.convert("RGB")
    px = processor(images=img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(
            px,
            max_length           = MAX_TARGET_LENGTH,
            num_beams            = 2,
            repetition_penalty   = 2.5,
            length_penalty       = 1.0,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
        )
    pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
    if label:
        print(f"    GT  : {label}")
    print(f"    PRED: {pred}")
    return pred


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── STEP 0: Font ─────────────────────────
    print("=" * 60)
    print("STEP 0 — Font Setup")
    print("=" * 60)
    font_ok = download_font()
    if font_ok:
        print(f"  Bengali font ready: {os.path.abspath(FONT_PATH)}")
    else:
        print("  ⚠ No font — Bengali will render as boxes. Please install manually.")

    # ── STEP 1: Model ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — Load Model & Processor")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    print(f"  Model  : {MODEL_CHECKPOINT}")

    processor = TrOCRProcessor.from_pretrained(MODEL_CHECKPOINT)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)

    # Structural token settings → model.config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.eos_token_id           = processor.tokenizer.sep_token_id

    # Generation settings → model.generation_config (new API)
    model.generation_config.max_length             = MAX_TARGET_LENGTH
    model.generation_config.early_stopping         = True
    model.generation_config.no_repeat_ngram_size   = 3
    model.generation_config.length_penalty         = 2.0
    model.generation_config.num_beams              = 4
    model.generation_config.repetition_penalty     = 2.5
    model.generation_config.pad_token_id           = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id           = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id

    model.gradient_checkpointing_enable()   # saves ~30% VRAM
    model.to(device)

    # ── STEP 2: Datasets ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Build Datasets")
    print("=" * 60)
    en_samples  = load_english_samples(MAX_EN_SAMPLES)
    bn_samples  = load_bengali_samples(MAX_BN_SAMPLES)
    mix_samples = load_mixed_samples(MAX_MIX_SAMPLES)

    # ── STEP 3: Visualize ────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Visualize Dataset Samples")
    print("=" * 60)
    visualize_samples(en_samples, bn_samples, mix_samples)

    # ── STEP 4: Splits ───────────────────────
    print("=" * 60)
    print("STEP 4 — Train / Val Splits")
    print("=" * 60)
    en_train,  en_val  = build_splits(en_samples)
    bn_train,  bn_val  = build_splits(bn_samples)
    mix_train, mix_val = build_splits(mix_samples)

    all_train = en_train + bn_train + mix_train
    all_val   = en_val   + bn_val   + mix_val
    random.shuffle(all_train)

    print(f"  English  train={len(en_train):>4}  val={len(en_val):>3}")
    print(f"  Bengali  train={len(bn_train):>4}  val={len(bn_val):>3}")
    print(f"  Mixed    train={len(mix_train):>4}  val={len(mix_val):>3}")
    print(f"  {'─'*33}")
    print(f"  TOTAL    train={len(all_train):>4}  val={len(all_val):>3}")

    train_dataset = OCRDataset(all_train, processor, MAX_TARGET_LENGTH)
    val_dataset   = OCRDataset(all_val,   processor, MAX_TARGET_LENGTH)

    # ── STEP 5: Train ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Training")
    print("=" * 60)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_steps                = 150,
        weight_decay                = 0.01,
        logging_steps               = 50,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,
        predict_with_generate       = True,
        generation_max_length       = MAX_TARGET_LENGTH,
        fp16                        = torch.cuda.is_available(),
        report_to                   = "none",
        seed                        = SEED,
        dataloader_num_workers      = 0,      # must be 0 on Windows
        remove_unused_columns       = False,
        dataloader_pin_memory       = True,
    )

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        processing_class = processor,
        data_collator    = default_data_collator,
        compute_metrics  = make_compute_metrics(processor),
    )

    trainer.train()

    # ── STEP 6: Save ─────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Save Model")
    print("=" * 60)
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"  ✓ Saved to: {os.path.abspath(OUTPUT_DIR)}")

    # ── STEP 7: Inference Test ───────────────
    print("\n" + "=" * 60)
    print("STEP 7 — Inference Test")
    print("=" * 60)

    test_cases = [
        ("English only",
         text_to_image("Invoice Total 1,250.00", use_bengali=False),
         "Invoice Total 1,250.00"),
        ("English + numbers",
         text_to_image("Order REF-4821 $99.99", use_bengali=False),
         "Order REF-4821 $99.99"),
        ("Bengali only",
         text_to_image("বাংলা ভাষা আমাদের মাতৃভাষা", use_bengali=True, img_width=700, font_size=28),
         "বাংলা ভাষা আমাদের মাতৃভাষা"),
        ("Bengali + numbers",
         text_to_image("মোট ৳১,২৫০ তারিখ ২০২৪", use_bengali=True, img_width=700, font_size=28),
         "মোট ৳১,২৫০ তারিখ ২০২৪"),
        ("Mixed EN + BN",
         text_to_image("Invoice বাংলা Total ৳1,500 মোট", use_bengali=True, img_width=760, font_size=26),
         "Invoice বাংলা Total ৳1,500 মোট"),
        ("Mixed with numbers",
         text_to_image("REF-007 আমি 2024 বাংলা $99", use_bengali=True, img_width=760, font_size=26),
         "REF-007 আমি 2024 বাংলা $99"),
    ]

    for desc, img, gt in test_cases:
        print(f"\n  [{desc}]")
        run_inference(model, processor, device, img, label=gt)

    print("\n" + "=" * 60)
    print("  DONE!")
    print(f"  Model   : {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Preview : {os.path.abspath(VIZ_PATH)}")

    print("=" * 60)
