"""Track B2: train and score question-guided compressors on synthetic mixtures or Clotho multi-question data.

    python b2_train.py train --data synthetic --arm route --k 4 --seed 17 --out runs/b2/syn-k4-s17-route
    python b2_train.py train --data clotho --arm gatedc --k 4 --seed 17 --steps 2250 --out runs/b2/clo-k4-s17-gatedc
    python b2_train.py score --data synthetic --split confirm --out runs/b2/syn-k4-s17-route   # rescore a checkpoint

Arms warm-start from the B1 audio-only parents (runs/guidance/k{K}-s{seed}-parent). The optimizer, batch size,
learning rates and decoder prompt follow bottleneck.py; bottleneck.py itself is not modified because track A's
locked final evaluation pins its hash. See phase2/b2-v2/plan.json.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from b2_mixtures import SUFFIX, Backbone
from bottleneck import GENERIC_STEM, Compressor


class Router(Compressor):
    """Each learned query attends over the question's token embeddings and gets its own bounded offset."""

    def __init__(self, width, k):
        super().__init__(width, k, "add")
        self.question_keys = nn.Linear(width, 64, bias=False)
        self.question_values = nn.Linear(width, 64, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, audio, tokens=None):
        queries = self.queries
        if tokens is not None:
            t = F.layer_norm(tokens, tokens.shape[-1:])
            attention = (queries @ self.question_keys(t).T / 8).softmax(-1)
            queries = queries + torch.tanh(self.gate) * F.normalize(attention @ self.question_values(t), dim=-1)
        weights = (queries @ self.keys(F.layer_norm(audio, audio.shape[-1:])).T / 8).softmax(-1)
        pooled = F.adaptive_avg_pool1d(audio.T.unsqueeze(0), len(queries))[0].T
        return pooled + weights @ audio - audio.mean(0)


def clotho_train():
    """All unanimous yes/no questions on locally available official-train recordings, with a donor per question."""
    local = {f for f in os.listdir("data/clotho") if f.endswith(".wav")}
    groups = collections.defaultdict(list)
    with open("data/clotho/clotho_aqa_train.csv", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            groups[r["file_name"], r["QuestionText"].strip()].append(r["answer"].strip().lower())
    per_clip = collections.defaultdict(list)
    for (name, question), answers in sorted(groups.items()):
        if name in local and len(answers) >= 3 and len(set(answers)) == 1 and answers[0] in ("yes", "no"):
            per_clip[name].append((question, 0 if answers[0] == "yes" else 1))
    items = []
    for name, questions in sorted(per_clip.items()):
        for j, (question, label) in enumerate(questions):
            donor = questions[(j + 1) % len(questions)][0] if len(questions) > 1 else GENERIC_STEM
            items.append({"path": f"data/clotho/{name}", "question": question, "donor": donor, "label": label})
    return items


def load_data(kind, split):
    if kind == "synthetic":
        manifest = {"dev": "data/b2mix/manifest.json", "confirm": "data/b2mix-confirm/manifest.json"}[split]
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        train = json.loads(Path("data/b2mix/manifest.json").read_text(encoding="utf-8"))["train"]
        return train, data["val"]
    fresh = json.loads(Path("data/clotho/fresh_val.json").read_text(encoding="utf-8"))["val"]
    train = clotho_train()
    assert not {x["path"] for x in train} & {x["path"] for x in fresh}
    return train, fresh


class Arm:
    def __init__(self, model, arm, k, train_stems):
        self.model, self.arm = model, arm
        width = model.embed.embedding_dim
        self.comp = (Router(width, k) if arm == "route" else Compressor(width, k, "gated" if arm == "gatedc" else "add")).cuda()
        self.center = None
        if arm == "gatedc":
            with torch.no_grad():
                self.center = torch.cat([self.pooled(s) for s in sorted(train_stems)]).mean(0, keepdim=True)

    def tokens(self, stem):
        ids = self.model.proc.tokenizer(stem, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        return self.model.embed(ids)[0].float()

    def pooled(self, stem):
        return self.tokens(stem).mean(0, keepdim=True)

    def latent(self, audio, stem):
        if self.arm == "audio" or stem is None:
            return self.comp(audio)
        if self.arm == "route":
            return self.comp(audio, self.tokens(stem))
        vector = F.layer_norm(self.pooled(stem) - self.center, (audio.shape[-1],))
        return self.comp(audio, vector)


def score(arm, items, out, cache):
    conditions = {"correct": lambda x: x["question"]}
    if arm.arm != "audio":
        conditions.update(wrong=lambda x: x["donor"], generic=lambda x: GENERIC_STEM, none=lambda x: None)
    partial = out.with_suffix(".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle, torch.no_grad():
        writer = csv.writer(handle)
        writer.writerow(["item", "path", "n", "mixture", "label", "condition", "order", "yes_minus_no"])
        for i, x in enumerate(items):
            audio = cache(x["path"])
            for name, stem in conditions.items():
                latent = arm.latent(audio, stem(x))
                for order, suffix in SUFFIX.items():
                    logits = arm.model.decode(latent, x["question"] + suffix)
                    margin = float(logits[0] - logits[1]) * (1 if order == 0 else -1)
                    writer.writerow([i, x["path"], x.get("n", ""), x.get("mixture", ""), x["label"], name, order, f"{margin:.5f}"])
    partial.replace(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "score"])
    parser.add_argument("--data", choices=["synthetic", "clotho"], required=True)
    parser.add_argument("--arm", choices=["audio", "gatedc", "route"])
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--split", default="dev", help="synthetic scoring split: dev or confirm")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=Path("cache/features"))
    args = parser.parse_args()
    if args.action == "score":
        config = json.loads((args.out / "train.json").read_text())["config"]
        args.arm, args.k, args.seed = config["arm"], config["k"], config["seed"]
    elif (args.out / "compressor.pt").exists():
        parser.error("Choose a fresh output directory")
    model = Backbone(args.feature_cache)
    memory = {}

    def cache(path):
        if path not in memory:
            memory[path] = model.features(path).cpu()
        return memory[path].cuda()

    train, evaluation = load_data(args.data, args.split if args.data == "synthetic" else "dev")
    arm = Arm(model, args.arm, args.k, {x["question"] for x in train})
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.action == "score":
        arm.comp.load_state_dict(torch.load(args.out / "compressor.pt", weights_only=True))
        arm.comp.eval()
        score(arm, evaluation, args.out / f"scores-{args.split}.csv", cache)
        return

    parent = Path(f"runs/guidance/k{args.k}-s{args.seed}-parent/compressor.pt")
    missing, unexpected = arm.comp.load_state_dict(torch.load(parent, weights_only=True), strict=False)
    assert not unexpected and set(missing) <= {"gate", "question_keys.weight", "question_values.weight"}, (missing, unexpected)
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"config": dict(arm=args.arm, k=args.k, seed=args.seed, steps=args.steps, data=args.data,
                             train_questions=len(train), parent=str(parent),
                             parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())}
    if args.arm != "audio":
        with torch.no_grad():
            for x in evaluation:
                audio = cache(x["path"])
                assert torch.equal(arm.latent(audio, x["question"]), arm.latent(audio, None))
        x = train[0]
        loss = F.cross_entropy(model.decode(arm.latent(cache(x["path"]), x["question"]), x["question"] + SUFFIX[0])[None],
                               torch.tensor([x["label"]], device="cuda"))
        gate_grad = torch.autograd.grad(loss, arm.comp.gate)[0].item()
        assert gate_grad != 0
        result["start_check"] = dict(identical_latents=len(evaluation), gate_grad=gate_grad)
    groups = [dict(params=[p for n, p in arm.comp.named_parameters() if n != "gate"])]
    if args.arm != "audio":
        groups.append(dict(params=[arm.comp.gate], lr=0.01, weight_decay=0.0))
    opt = torch.optim.AdamW(groups, lr=0.001)
    order = list(train)
    losses, gates, start = [], [], time.time()
    arm.comp.train()
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        x = order[step % len(order)]
        scores = model.decode(arm.latent(cache(x["path"]), x["question"]), x["question"] + SUFFIX[0])
        loss = F.cross_entropy(scores[None], torch.tensor([x["label"]], device="cuda"))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(arm.comp.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(loss.item())
        if args.arm != "audio":
            gates.append(torch.tanh(arm.comp.gate).item())
        if (step + 1) % 200 == 0:
            print(f"step {step+1} loss {sum(losses[-200:])/200:.4f}" + (f" gate {gates[-1]:+.3f}" if gates else "")
                  + f" {time.time()-start:.0f}s", flush=True)
    arm.comp.eval()
    torch.save(arm.comp.state_dict(), args.out / "compressor.pt")
    result.update(loss_by_200=[sum(losses[i:i+200]) / len(losses[i:i+200]) for i in range(0, len(losses), 200)],
                  gate=gates[-1] if gates else None, train_seconds=time.time() - start,
                  query_norm=arm.comp.queries.norm(dim=-1).mean().item(),
                  peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30)
    (args.out / "train.json").write_text(json.dumps(result, indent=1))
    score(arm, evaluation, args.out / "scores-dev.csv", cache)
    print("done", args.out, flush=True)


if __name__ == "__main__":
    main()
