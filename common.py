import io
from pathlib import Path

import jiwer
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class IAMLineDataset(Dataset):
    def __init__(self, parquet_path, start=0, end=None):
        import pandas as pd
        frame = pd.read_parquet(parquet_path).iloc[start:end].reset_index(drop=True)
        self.images = frame["image"].tolist()
        self.texts = frame["text"].astype(str).tolist()

    def __len__(self): return len(self.texts)

    def __getitem__(self, index):
        value = self.images[index]
        if isinstance(value, dict):
            image = Image.open(io.BytesIO(value["bytes"])) if value.get("bytes") is not None else Image.open(value["path"])
        else:
            image = Image.open(value)
        return image.convert("RGB"), self.texts[index]


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trocr(model_name_or_path):
    target = device()
    processor = TrOCRProcessor.from_pretrained(model_name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    return processor, model.to(target), target


def collate(processor, batch):
    images, texts = zip(*batch)
    pixels = processor(images=list(images), return_tensors="pt").pixel_values
    labels = processor.tokenizer(list(texts), padding=True, return_tensors="pt").input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return pixels, labels, list(texts)


def predict_dataset(model, processor, dataset, target_device):
    model.eval(); refs, preds = [], []
    with torch.inference_mode():
        for image, text in dataset:
            pixels = processor(images=image, return_tensors="pt").pixel_values.to(target_device)
            ids = model.generate(pixels, max_new_tokens=128)
            refs.append(text); preds.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
    return refs, preds


def cer_wer(refs, preds):
    return {"cer": float(jiwer.cer(refs, preds)), "wer": float(jiwer.wer(refs, preds))}
