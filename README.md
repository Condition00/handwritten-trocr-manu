# Handwritten line recognition experiment

This project uses `microsoft/trocr-small-handwritten` to import pretrained
visual encoder and decoder components. It fine-tunes image-to-text recognition
on genuine IAM line images. It does not train a vision encoder-decoder from
scratch.

Run inference:

```bash
HF_HOME=/tmp/dl-project-hf uv run python infer_trocr.py path/to/line.png
```

See `ACHIEVABILITY.md` for measured results, limitations, and commands.
