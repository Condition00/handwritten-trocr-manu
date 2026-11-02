import argparse, csv, json
from pathlib import Path
import transformers
from common import IAMLineDataset, cer_wer, load_trocr, pair_metrics, predict_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True); parser.add_argument("--data", required=True)
parser.add_argument("--start", type=int, default=0); parser.add_argument("--count", type=int, default=8)
parser.add_argument("--predictions", required=True); parser.add_argument("--results", required=True)
parser.add_argument("--revision")
args = parser.parse_args()
processor, model, target = load_trocr(args.model)
dataset = IAMLineDataset(args.data, args.start, args.start + args.count)
refs, preds = predict_dataset(model, processor, dataset, target)
metrics = {
    "model": args.model, "data": args.data, "start": args.start, "count": len(refs),
    "generation": {"max_new_tokens": 128},
    "transformers_version": transformers.__version__,
    "revision": args.revision,
    **cer_wer(refs, preds),
}
Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
with open(args.predictions, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_id", "reference", "prediction", "cer", "wer"]); writer.writeheader()
    writer.writerows(
        {"image_id": image_id, "reference": r, "prediction": p, **pair_metrics(r, p)}
        for image_id, r, p in zip(dataset.identifiers, refs, preds)
    )
Path(args.results).write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
