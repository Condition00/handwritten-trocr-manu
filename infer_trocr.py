import argparse
from PIL import Image
from common import load_trocr

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--model", default="microsoft/trocr-small-handwritten")
args = parser.parse_args()
processor, model, target = load_trocr(args.model)
image = Image.open(args.image).convert("RGB")
pixels = processor(images=image, return_tensors="pt").pixel_values.to(target)
with __import__("torch").inference_mode():
    ids = model.generate(pixels, max_new_tokens=128)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
