import io
import json
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
        self.identifiers = frame.get("source_index", frame.index).tolist()

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
    processor_kwargs = {}
    tokenizer_config = Path(model_name_or_path) / "tokenizer_config.json"
    if tokenizer_config.is_file():
        with tokenizer_config.open() as file:
            extra_special_tokens = json.load(file).get("extra_special_tokens")
        if isinstance(extra_special_tokens, list):
            processor_kwargs["extra_special_tokens"] = {}
    processor = TrOCRProcessor.from_pretrained(model_name_or_path, **processor_kwargs)
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    generation_config = model.generation_config
    token_ids = {
        "decoder_start_token_id": generation_config.decoder_start_token_id,
        "eos_token_id": generation_config.eos_token_id,
        "pad_token_id": generation_config.pad_token_id,
    }
    if any(value is None for value in token_ids.values()):
        raise ValueError(f"Incomplete generation token configuration: {token_ids}")
    if token_ids["eos_token_id"] != processor.tokenizer.eos_token_id:
        raise ValueError(f"EOS token mismatch: {token_ids['eos_token_id']} != {processor.tokenizer.eos_token_id}")
    if token_ids["pad_token_id"] != processor.tokenizer.pad_token_id:
        raise ValueError(f"PAD token mismatch: {token_ids['pad_token_id']} != {processor.tokenizer.pad_token_id}")
    model.config.decoder_start_token_id = token_ids["decoder_start_token_id"]
    model.config.eos_token_id = token_ids["eos_token_id"]
    model.config.pad_token_id = token_ids["pad_token_id"]
    model.generation_config.decoder_start_token_id = token_ids["decoder_start_token_id"]
    model.generation_config.eos_token_id = token_ids["eos_token_id"]
    model.generation_config.pad_token_id = token_ids["pad_token_id"]
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


def pair_metrics(reference, prediction):
    return {
        "cer": float(jiwer.cer(reference, prediction)),
        "wer": float(jiwer.wer(reference, prediction)),
    }
