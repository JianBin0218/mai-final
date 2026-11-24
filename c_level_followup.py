"""Compare a frozen reader, threshold calibration, and the Track C adapter on relative level."""

import csv
import numpy as np
import torch
from torch.nn import functional as F

from c_readout import Adapter
from level_contrast import (append_scores, calibration, digest, estimate, order_prompts, pair_metrics,
                            read_json, score_pairs, summarize, write_json)


def measure(args, prototype, features, decode):
    plan = read_json(args.c_level_plan)
    pairs = read_json(args.pair_manifest)
    if digest(args.pair_manifest) != plan["manifest_sha256"] or digest(args.compressor) != plan["checkpoint_sha256"]:
        raise ValueError("Expected the frozen A stimuli manifest and K=16 checkpoint")
    development, evaluation = pairs["development"], pairs["evaluation"]
    full_model = type(prototype)(prototype.keys.in_features, 16).cuda().eval().requires_grad_(False)
    full_model.load_state_dict(torch.load(args.compressor, weights_only=True))
    examples = [(p[k], j) for p in development for j, k in enumerate(("first", "second"))]
    prompts = order_prompts(plan["prompt"])
    summary = {"plan_sha256": digest(args.c_level_plan), "models": {}}
    path = args.out / "native.csv"
    with path.open("x", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "db", "prompt", "original_id", "source", "AB_first_clip", "AB_second_clip",
                         "BA_first_clip", "BA_second_clip"])
        for name in ("full", "pool16", "K16"):
            @torch.no_grad()
            def latent(path):
                audio = features(path)
                if name == "full":
                    return audio
                if name == "pool16":
                    return F.adaptive_avg_pool1d(audio.T[None], 16)[0].T
                return full_model(audio)

            torch.manual_seed(plan["seed"])
            rng = np.random.default_rng(plan["seed"])
            adapter = Adapter(prototype.keys.in_features, plan["hidden"]).cuda()
            with torch.no_grad():
                dev_base = score_pairs(development, latent, decode, plan["prompt"], args.batch_size)
                base = score_pairs(evaluation, latent, decode, plan["prompt"], args.batch_size)
                initial = score_pairs(development[:2], lambda path: adapter(latent(path)), decode, plan["prompt"], args.batch_size)
                original = score_pairs(development[:2], latent, decode, plan["prompt"], args.batch_size)
                if not np.array_equal(initial, original):
                    raise RuntimeError(f"The zero-initialized adapter changed {name} predictions")
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=plan["lr"])
            for step in range(plan["steps"]):
                order = step % 2
                indices = rng.choice(len(examples), size=plan["train_batch"], replace=False)
                batch = [(latent(examples[i][0]), examples[i][1] ^ order) for i in indices]
                logits = decode(adapter(torch.stack([x for x, _ in batch])), prompts[order])
                loss = F.cross_entropy(logits, torch.tensor([y for _, y in batch], device="cuda"))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                tuned = score_pairs(evaluation, lambda path: adapter(latent(path)), decode, plan["prompt"], args.batch_size)
                append_scores(writer, f"{name}-base", plan["prompt"], evaluation, base)
                append_scores(writer, f"{name}-adapter", plan["prompt"], evaluation, tuned)
                calibrated = base - np.median(dev_base, axis=(0, 2))[None, :, None]
                summary["models"][name] = {"base": summarize(base, evaluation, plan),
                                             "adapter": summarize(tuned, evaluation, plan),
                                             "calibration": calibration(dev_base, base, evaluation, plan),
                                             "adapter_minus_base_ranking": estimate(
                                                 (pair_metrics(tuned)["ranking"]-pair_metrics(base)["ranking"]).mean(1),
                                                 evaluation, plan),
                                             "adapter_minus_calibrated_both_correct": estimate(
                                                 pair_metrics(tuned)["both_correct"].mean(1)
                                                 -pair_metrics(calibrated)["both_correct"].mean(1), evaluation, plan)}
                if name == "K16" and args.extra_manifest:
                    rows = []
                    for item in read_json(args.extra_manifest)["val"]:
                        x = latent(item["path"])
                        q = item["question"] + "\nA. yes\nB. no\nAnswer with only A or B."
                        logits = decode(torch.stack((x, adapter(x))), q)
                        base_hit, adapter_hit = (logits.argmax(1) == item["label"]).tolist()
                        rows.append([item["path"], item["question"], item["label"], int(base_hit), int(adapter_hit)])
                    with (args.out / "natural.csv").open("x", encoding="utf-8", newline="") as out:
                        cw = csv.writer(out)
                        cw.writerow(["recording", "question", "label", "base_correct", "adapter_correct"])
                        cw.writerows(rows)
                    summary["natural_K16"] = {"n": len(rows), "base_accuracy": float(np.mean([r[-2] for r in rows])),
                                               "adapter_accuracy": float(np.mean([r[-1] for r in rows]))}
            torch.save(adapter.state_dict(), args.out / f"{name}-adapter.pt")
            print("Scored fixed readers for", name, flush=True)
    write_json(args.out / "summary.json", summary)
