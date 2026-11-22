"""Evaluate paired recordings using next-token A/B forced choice."""
import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from scipy.signal import resample_poly
from math import gcd
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


def waveform(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    factor = gcd(rate, 16000)
    return resample_poly(samples, 16000 // factor, rate // factor).astype(np.float32)


@torch.inference_mode()
def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained("model")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "model", torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    ids = [processor.tokenizer.encode(letter, add_special_tokens=False) for letter in "AB"]
    assert all(len(x) == 1 for x in ids)
    labels = [x[0] for x in ids]

    def scores(prompt, clip=None):
        content = [{"type": "audio", "audio_url": clip}] if clip else []
        content.append({"type": "text", "text": prompt})
        template = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        kwargs = {"audios": [waveform(clip)], "sampling_rate": 16000} if clip else {}
        inputs = processor(text=template, return_tensors="pt", **kwargs).to(model.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
        logits = model(**inputs, use_cache=False).logits[0, -1, labels].float().tolist()
        return logits[1] - logits[0]

    output = Path(os.environ.get("RESULT_DIR", "results"))
    output.mkdir(exist_ok=True)
    pairs = json.loads(Path(os.environ.get("PAIR_FILE", "data/pairs.json")).read_text())
    rows = []
    load_seconds = time.monotonic() - start
    with (output / "predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["pair", "task", "clips", "condition", "clip_index", "target", "prediction", "logit_B_minus_A"])
        for i, pair in enumerate(pairs):
            a, b = pair["options"]
            prompt = f"{pair['question']}\nA. {a}\nB. {b}\nAnswer with only A or B."
            audio = [scores(prompt, p) for p in pair["clips"]]
            text = scores(prompt)
            for condition, margins in [("audio", audio), ("text", [text, text]), ("swapped", audio[::-1])]:
                hits = []
                for target, margin in enumerate(margins):
                    pred = int(margin > 0)
                    hits.append(pred == target)
                    writer.writerow([i, pair["task"], "|".join(pair["clips"]), condition, target,
                                     "AB"[target], "AB"[pred], margin])
                rows.append((pair["task"], condition, hits))
            file.flush()
            if i % 10 == 0:
                print(f"{i+1}/{len(pairs)} pairs", flush=True)
    summary = {}
    for task, condition in sorted({(t, c) for t, c, _ in rows}):
        hits = np.asarray([h for t, c, h in rows if (t, c) == (task, condition)])
        summary[f"{task}/{condition}"] = dict(pairs=len(hits), correct=int(hits.sum()),
            both_correct=int(hits.all(axis=1).sum()), accuracy=float(hits.mean()))
    stats = dict(results=summary, gpu=torch.cuda.get_device_name(), gpu_index=os.getenv("CUDA_VISIBLE_DEVICES"),
                 peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                 python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                 load_seconds=load_seconds, eval_seconds=time.monotonic()-start-load_seconds)
    (output / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
