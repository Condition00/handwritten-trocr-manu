import argparse, json, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

API = "https://datasets-server.huggingface.co/rows"

def rows(dataset, config, split, count):
    result = []
    while len(result) < count:
        n = min(100, count - len(result)); query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": len(result), "length": n})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"{API}?{query}", timeout=120) as response: result.extend(x["row"] for x in json.load(response)["rows"])
                break
            except Exception:
                if attempt == 4: raise
                time.sleep(2 ** attempt)
    return result

def fetch(row):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(row["image"]["src"], timeout=120) as response: image = response.read()
            return {"bytes": image, "path": None}, row["text"]
        except Exception:
            if attempt == 4: raise
            time.sleep(2 ** attempt)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", default="Teklia/IAM-line"); p.add_argument("--config", default="default"); p.add_argument("--output-dir", default="data/iam_line/official_subset"); p.add_argument("--train-count", type=int, default=500); p.add_argument("--validation-count", type=int, default=50); p.add_argument("--test-count", type=int, default=100); p.add_argument("--workers", type=int, default=16); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", a.train_count), ("validation", a.validation_count), ("test", a.test_count)):
        path = out / f"{split}.parquet"
        if path.exists() and len(pd.read_parquet(path)) == count: print(f"reuse {path}"); continue
        source = rows(a.dataset, a.config, split, count)
        with ThreadPoolExecutor(max_workers=a.workers) as pool: values = list(pool.map(fetch, source))
        pd.DataFrame([{"image": i, "text": t, "source_split": split, "source_index": n} for n, (i, t) in enumerate(values)]).to_parquet(path, index=False)
        print(f"saved {path}: {count} rows")
    print("writer_disjoint: unavailable; official split boundaries used")

if __name__ == "__main__": main()
