import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from common import IAMLineDataset, cer_wer, collate, load_trocr, predict_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True); parser.add_argument("--validation-data")
parser.add_argument("--model", default="microsoft/trocr-small-handwritten"); parser.add_argument("--output", required=True)
parser.add_argument("--start", type=int, default=0); parser.add_argument("--count", type=int, default=8)
parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=5e-5); parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
if args.gradient_accumulation_steps < 1: raise ValueError("--gradient-accumulation-steps must be positive")
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
processor, model, target = load_trocr(args.model)
dataset = IAMLineDataset(args.data, args.start, args.start + (1 if args.smoke else args.count))
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(processor, b))
validation = IAMLineDataset(args.validation_data) if args.validation_data else None
output = Path(args.output)
if output.exists() and any(output.iterdir()): raise FileExistsError(f"Output directory is not empty: {output}")
output.mkdir(parents=True, exist_ok=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
best = float("inf"); best_fine_tuned = None; steps = 0; optimizer_steps = 0
base_refs, base_preds = predict_dataset(model, processor, validation, target) if validation else ([], [])
if validation:
    base_metrics = cer_wer(base_refs, base_preds)
    best = base_metrics["cer"]; selected = Path(args.model)
    (output / "training_log.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (output / "training_log.jsonl").open("w") as handle:
        handle.write(json.dumps({"epoch": 0, "steps": 0, "optimizer_steps": 0, "checkpoint": str(args.model), **base_metrics}) + "\n")
optimizer.zero_grad(set_to_none=True)
for epoch in range(1, args.epochs + 1):
    model.train()
    for batch_index, (pixels, labels, _) in enumerate(loader, start=1):
        loss = model(pixel_values=pixels.to(target), labels=labels.to(target)).loss
        (loss / args.gradient_accumulation_steps).backward(); steps += 1
        should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(loader)
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
        print(f"step={steps} optimizer_step={optimizer_steps} loss={loss.item():.5f}", flush=True)
        if args.smoke: break
    checkpoint = output / f"epoch-{epoch}"; model.save_pretrained(checkpoint); processor.save_pretrained(checkpoint)
    record = {"epoch": epoch, "steps": steps, "optimizer_steps": optimizer_steps, "checkpoint": str(checkpoint)}
    if validation:
        refs, preds = predict_dataset(model, processor, validation, target); record.update(cer_wer(refs, preds))
        if record["cer"] < best:
            best = record["cer"]; best_fine_tuned = checkpoint
            if record["cer"] < base_metrics["cer"]: selected = checkpoint
    with (output / "training_log.jsonl").open("a") as handle: handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    if args.smoke: break
selected = locals().get("selected", output / f"epoch-{1 if args.smoke else args.epochs}")
(output / "selected_checkpoint.txt").write_text(str(selected) + "\n")
(output / "best_fine_tuned_checkpoint.txt").write_text(str(best_fine_tuned) + "\n")
(output / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
print(f"selected={selected} steps={steps}")
