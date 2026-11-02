import math
import unittest
from pathlib import Path

import torch

from common import IAMLineDataset, collate, load_trocr


ROOT = Path(__file__).parents[1]
CHECKPOINTS = (
    ROOT / "outputs/cpu_trocr_3epoch/epoch-3",
    ROOT / "outputs/fine_tune_lr1e5/epoch-2",
)
DATA = ROOT / "data/iam_line/official_subset/test.parquet"


class TrOCRPipelineTest(unittest.TestCase):
    @unittest.skipUnless(any(path.is_dir() for path in CHECKPOINTS) and DATA.is_file(), "local experiment artifacts unavailable")
    def test_checkpoint_configuration_and_forward_pass(self):
        dataset = IAMLineDataset(str(DATA), 0, 1)
        for checkpoint in CHECKPOINTS:
            if not checkpoint.is_dir():
                continue
            with self.subTest(checkpoint=checkpoint):
                processor, model, target = load_trocr(str(checkpoint))
                self.assertEqual(model.config.decoder_start_token_id, 2)
                self.assertEqual(model.generation_config.decoder_start_token_id, 2)
                self.assertEqual(model.config.eos_token_id, processor.tokenizer.eos_token_id)
                self.assertEqual(model.config.pad_token_id, processor.tokenizer.pad_token_id)
                pixels, labels, _ = collate(processor, [dataset[0]])
                with torch.inference_mode():
                    loss = model(pixel_values=pixels.to(target), labels=labels.to(target)).loss
                self.assertTrue(math.isfinite(float(loss)))


if __name__ == "__main__":
    unittest.main()
