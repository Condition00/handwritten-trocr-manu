# Handwritten line recognition experiment

This project uses `microsoft/trocr-small-handwritten` to import pretrained
visual encoder and decoder components. It fine-tunes image-to-text recognition
on genuine IAM line images. It does not train a vision encoder-decoder from
scratch.

Run inference:

```bash
HF_HOME=/tmp/dl-project-hf uv run python infer_trocr.py path/to/line.png
```

Offline evaluation works after the model is cached:

```bash
HF_HOME=/tmp/dl-project-hf HF_HUB_OFFLINE=1 uv run python evaluate_trocr.py \
  --model microsoft/trocr-small-handwritten \
  --data data/iam_line/official_subset/test.parquet \
  --count 100 \
  --predictions outputs/pretrained_recheck.csv \
  --results outputs/pretrained_recheck.json
```

Controlled fine-tuning uses `outputs/fine_tune_lr1e5/` and keeps the pretrained
model as a validation candidate.

See `ACHIEVABILITY.md` for measured results, limitations, and commands.
