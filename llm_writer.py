"""Compare a Qwen latent writer with a learned audio resampler under the same frozen reader."""

import csv
import hashlib
import json
import random
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LatentWriter(nn.Module):
    def __init__(self, slots, model, rank):
        super().__init__()
        self.slots = nn.Parameter(slots)
        self.updates = nn.ModuleDict()
        self._targets = {}
        if rank:
            for i, layer in enumerate(model.layers[-4:]):
                for name in ("q_proj", "v_proj"):
                    target = getattr(layer.self_attn, name)
                    key = f"layer{i}_{name}"
                    down = nn.Linear(target.in_features, rank, bias=False, device=slots.device)
                    up = nn.Linear(rank, target.out_features, bias=False, device=slots.device)
                    nn.init.zeros_(up.weight)
                    self.updates[key] = nn.Sequential(down, up)
                    self._targets[key] = target

    def encode(self, trunk, prefix, target_norm):
        sequence = torch.cat((prefix, self.slots[None].to(prefix.dtype)), dim=1)
        with ExitStack() as stack:
            for key, adapter in self.updates.items():
                def add_update(_module, inputs, output, adapter=adapter):
                    return output + adapter(inputs[0].float()).to(output.dtype)

                handle = self._targets[key].register_forward_hook(add_update)
                stack.callback(handle.remove)
            hidden = trunk(inputs_embeds=sequence, use_cache=False).last_hidden_state[0, -len(self.slots):]
        return F.normalize(hidden.float(), dim=-1) * target_norm


def yes_scores(logits, reverse):
    return (logits[..., 0] - logits[..., 1]) * (-1 if reverse else 1)


def summarize(rows):
    by_split = defaultdict(list)
    for row in rows:
        by_split[row[0], row[4], row[5]].append(row)
    result = {}
    for (split, order, arm), items in sorted(by_split.items()):
        qids = defaultdict(dict)
        hits = []
        for _, clip, _, label, _, _, score in items:
            predicted_yes = score >= 0 if order == "AB" else score > 0
            hits.append(predicted_yes == (label == 0))
            qids[clip][label] = score
        gaps = np.array([scores[0] - scores[1] for scores in qids.values() if len(scores) == 2])
        result.setdefault(split, {}).setdefault(order, {})[arm] = {
            "n": len(hits), "accuracy": float(np.mean(hits)), "n_pairs": len(gaps),
            "ranking": float(np.mean((gaps > 0) + .5 * (gaps == 0))) if len(gaps) else None,
        }
    return result


def measure(args, prototype, features, decode, decode_text, lm, embed, proc, audio_token_index,
            train, test, extra, donor_stems):
    comparator = args.writer_reference or Path(f"capacity-k{args.k}-s{args.seed}-audio/compressor.pt")
    reference = type(prototype)(prototype.keys.in_features, args.k).cuda().eval().requires_grad_(False)
    reference.load_state_dict(torch.load(comparator, weights_only=True))
    rng = random.Random(args.seed)
    trunk = lm.model
    tokens_by_stem = {}

    def writer_prefix(audio, stem):
        if stem not in tokens_by_stem:
            content = [{"type": "text", "text": stem}] if stem else []
            content.append({"type": "audio", "audio_url": "placeholder"})
            prompt = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                              add_generation_prompt=True)
            ids = proc.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            index = (ids[0] == audio_token_index).nonzero().item()
            tokens_by_stem[stem] = embed(ids), index
        text, index = tokens_by_stem[stem]
        return torch.cat((text[:, :index], audio[None].to(text.dtype), text[:, index + 1:]), dim=1)

    with torch.no_grad():
        paths = sorted({path for path, _, _ in train})[:12]
        scale = torch.cat([reference(features(path)).norm(dim=-1) for path in paths]).median().item()
        slots = torch.randn(args.k, embed.embedding_dim, device="cuda") * embed.weight.float().std()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training reference did not provide a usable latent scale")
    model = LatentWriter(slots, trunk, args.writer_rank)
    checkpoint_path = args.out / "writer.pt"
    if checkpoint_path.exists() or (args.out / "summary.json").exists():
        raise FileExistsError("Use a fresh output directory for the LLM writer")

    def encode(audio, stem):
        return model.encode(trunk, writer_prefix(audio, stem), scale)

    if args.writer_pair_plan:
        plan = json.loads(args.writer_pair_plan.read_text(encoding="utf-8"))
        if plan.get("event_probe"):
            from writer_event_probe import measure as measure_pairs
        elif plan.get("position_study"):
            from writer_position import measure as measure_pairs
        else:
            from writer_pairs import measure as measure_pairs
        if hashlib.sha256(comparator.read_bytes()).hexdigest() != plan["checkpoints"]["external"]:
            raise ValueError("External checkpoint differs from the paired evaluation plan")
        if (args.k, args.seed, args.writer_rank) != (plan["k"], plan["writer_seed"], plan["writer_rank"]):
            raise ValueError("Writer configuration differs from the paired evaluation plan")
        measure_pairs(args, reference, model, encode, features, decode, decode_text)
        return

    def encode_stem(question):
        return question.split("\n", 1)[0] if args.writer_conditioned else None

    qa = test[0][1]
    with torch.no_grad():
        audio = features(test[0][0])
        before = decode(reference(audio), qa)
        z = encode(audio, encode_stem(qa))
        after = decode(reference(audio), qa)
        if z.shape != (args.k, embed.embedding_dim) or not torch.isfinite(z).all() or not torch.equal(before, after):
            raise RuntimeError("Invalid latent shape, or writer adapters changed the frozen reader")
        print("writer smoke", {"latent_shape": list(z.shape), "target_norm": scale,
                               "frozen_reader_max_logit_change": (after - before).abs().max().item()}, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.writer_lr)
    first_grad = None
    for step in range(args.steps):
        if step % len(train) == 0:
            rng.shuffle(train)
        path, question, label = train[step % len(train)]
        optimizer.zero_grad(set_to_none=True)
        latent = encode(features(path), encode_stem(question))
        logits = decode(latent, question)
        loss = F.cross_entropy(logits.view(1, -1), torch.tensor([label], device="cuda"))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item()
        if step == 0:
            first_grad = norm
            if norm <= 0:
                raise RuntimeError("Writer did not receive a gradient from the frozen reader")
        optimizer.step()
        if step == 0 or (step + 1) % 16 == 0 or step + 1 == args.steps:
            print("writer train", {"step": step + 1, "loss": loss.item(), "gradient_norm": norm,
                                   "peak_gib": torch.cuda.max_memory_allocated() / 2**30}, flush=True)
    torch.save(model.state_dict(), checkpoint_path)

    rows = []
    with torch.no_grad():
        for split, samples in (("pilot", test), ("fresh", extra)):
            latents = {}
            for i, (path, question, label) in enumerate(samples):
                audio = features(path)
                stems = {"writer": encode_stem(question)}
                if args.writer_conditioned:
                    stems.update(wrong_stem=donor_stems[path, question], no_stem=None,
                                 generic_stem="What can be heard in this recording?")
                arms = {"external": reference(audio)}
                for name, stem in stems.items():
                    if (path, stem) not in latents:
                        latents[path, stem] = encode(audio, stem).cpu()
                    arms[name] = latents[path, stem].cuda()
                for reverse in (False, True):
                    prompt = question.replace("\nA. yes\nB. no", "\nA. no\nB. yes") if reverse else question
                    order = "BA" if reverse else "AB"
                    scores = dict(zip(arms, yes_scores(decode(torch.stack(list(arms.values())), prompt), reverse).cpu().tolist()))
                    scores.update(full=yes_scores(decode(audio, prompt), reverse).item(),
                                  text=yes_scores(decode_text(prompt), reverse).item())
                    rows.extend([split, path, question.split("\n", 1)[0], label, order, name, score]
                                for name, score in scores.items())
                if i % 50 == 49:
                    print(f"writer evaluation {split}: {i+1}/{len(samples)}", flush=True)

    with (args.out / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "clip", "stem", "label", "order", "arm", "margin_yes"])
        writer.writerows(rows)
    summary = {"k": args.k, "seed": args.seed, "steps": args.steps, "writer_rank": args.writer_rank,
               "writer_conditioned": args.writer_conditioned, "target_norm": scale, "first_grad_norm": first_grad,
               "reference_sha256": hashlib.sha256(comparator.read_bytes()).hexdigest(),
               "train_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "extra_manifest_sha256": hashlib.sha256(args.extra_manifest.read_bytes()).hexdigest() if args.extra_manifest else None,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "results": summarize(rows)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("writer results", summary, flush=True)
