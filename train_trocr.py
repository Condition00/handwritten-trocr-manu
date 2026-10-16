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
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
processor, model, target = load_trocr(args.model)
dataset = IAMLineDataset(args.data, args.start, args.start + (1 if args.smoke else args.count))
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(processor, b))
validation = IAMLineDataset(args.validation_data) if args.validation_data else None
output = Path(args.output)
if output.exists() and any(output.iterdir()): raise FileExistsError(f"Output directory is not empty: {output}")
output.mkdir(parents=True, exist_ok=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
best = float("inf"); steps = 0
for epoch in range(1, args.epochs + 1):
    model.train()
    for pixels, labels, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = model(pixel_values=pixels.to(target), labels=labels.to(target)).loss
        loss.backward(); optimizer.step(); steps += 1
        print(f"step={steps} loss={loss.item():.5f}", flush=True)
        if args.smoke: break
    checkpoint = output / f"epoch-{epoch}"; model.save_pretrained(checkpoint); processor.save_pretrained(checkpoint)
    record = {"epoch": epoch, "steps": steps, "checkpoint": str(checkpoint)}
    if validation:
        refs, preds = predict_dataset(model, processor, validation, target); record.update(cer_wer(refs, preds))
        if record["cer"] < best: best = record["cer"]; selected = checkpoint
    with (output / "training_log.jsonl").open("a") as handle: handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    if args.smoke: break
selected = locals().get("selected", output / f"epoch-{1 if args.smoke else args.epochs}")
(output / "selected_checkpoint.txt").write_text(str(selected) + "\n")
print(f"selected={selected} steps={steps}")
