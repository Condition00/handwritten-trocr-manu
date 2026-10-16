# Current Achievability

Project uses `microsoft/trocr-small-handwritten` as pretrained visual encoder
and decoder components. It does not train vision encoder-decoder from scratch.
This is one model choice, not the only possible model.

## Completed experiment

- Dataset: genuine `Teklia/IAM-line` rows from Hugging Face datasets-server.
- Official split boundaries: 500 train, 50 validation, 100 test.
- Public schema: `image`, `text`; no writer IDs.
- Writer-disjoint split: unavailable from public mirror.
- Hardware: CPU only. CUDA unavailable. `nvidia-smi` could not communicate with NVIDIA driver.
- Batch size: 1.
- Optimizer: AdamW.
- Learning rate: `5e-5`.
- Seed: 42.
- Epochs: 3.
- Training steps: 1,500.
- Validation evaluation: after every epoch.
- Selected checkpoint: `outputs/official_trocr_3epoch/epoch-3`.

Validation history:

| Checkpoint | CER | WER |
|---|---:|---:|
| Epoch 1 | 1.0000 | 1.0000 |
| Epoch 2 | 0.9801 | 0.9951 |
| Epoch 3 | 0.7576 | 0.9534 |

Fixed test results:

| Model state | CER | WER |
|---|---:|---:|
| Pretrained model | 0.0645 | 0.1673 |
| Selected fine-tuned model | 0.7585 | 0.9565 |

Fine-tuning degraded performance. Results do not show improvement.

## Own handwritten images

Single-line images work for inference:

```bash
uv run python infer_trocr.py my_line.png --model microsoft/trocr-small-handwritten
```

Fine-tuning needs image/transcription pairs. Use parquet columns:

- `image`: image path or image bytes.
- `text`: exact transcription.

Keep train, validation, and test images separate. Do not evaluate on training
images.

## Artifacts

- `data/iam_line/official_subset/`
- `outputs/official_trocr_3epoch/`
- `outputs/official_base_test_metrics.json`
- `outputs/official_finetuned_test_metrics.json`
- `outputs/official_base_test_predictions.csv`
- `outputs/official_finetuned_test_predictions.csv`
- `outputs/official_trocr_3epoch/training_log.jsonl`

## Limits

- Dataset subset smaller than full IAM-line.
- No writer-disjoint evaluation.
- No GPU run.
- No learning-rate sweep or multi-seed study.
- No production accuracy claim.
- Current pipeline accepts single-line images only.
