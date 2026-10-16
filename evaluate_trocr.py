import argparse, csv, json
from pathlib import Path
from common import IAMLineDataset, cer_wer, load_trocr, predict_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True); parser.add_argument("--data", required=True)
parser.add_argument("--start", type=int, default=0); parser.add_argument("--count", type=int, default=8)
parser.add_argument("--predictions", required=True); parser.add_argument("--results", required=True)
args = parser.parse_args()
processor, model, target = load_trocr(args.model)
refs, preds = predict_dataset(model, processor, IAMLineDataset(args.data, args.start, args.start + args.count), target)
metrics = {"model": args.model, "data": args.data, "start": args.start, "count": len(refs), **cer_wer(refs, preds)}
Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
with open(args.predictions, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["index", "reference", "prediction"]); writer.writeheader()
    writer.writerows({"index": i, "reference": r, "prediction": p} for i, (r, p) in enumerate(zip(refs, preds)))
Path(args.results).write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
