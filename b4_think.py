"""Track B4: does Qwen2-Audio use latent thoughts about the audio? One writer, three readers.

Writer (pass 1): [audio][question + choices] <assistant> [start] [K learned queries]; the last-layer hidden states at
the K query positions, rescaled per dimension, are the latent thoughts (Coconut-style continuous states).
Reader (pass 2): [audio?][question + choices] <assistant> [start] [K latents] [end] -> letter logits.

  --reader soft  the reader also sees the audio (the usual latent-reasoning setup; the latents can be bypassed)
  --reader hard  the reader sees no audio; every bit of audio evidence must pass through the K latents
  --reader mid   training drops the reader's audio with probability --drop (default 0.5); both are scored

--recon W adds a CoAT-style auxiliary loss: a linear head must reconstruct the K chunk means of the audio tokens from
the K latents (MSE relative to the target's power, weight W), so the latents carry audio content by construction.

Controls: --k 0 keeps the start/end tokens and the same LoRA but no latents (the boundary-token-only effect);
--no-think removes the think block entirely (plain LoRA fine-tuning); --no-lora --no-think --steps 0 is the base model;
--blind-writer hides the question from the writer.

Scoring applies interventions to the latents of every item: normal, swap (another item's latents), mean (the
evaluation-set mean latent), zero, noise (Gaussian at the latents' RMS added), and for readers that can see audio,
the same without audio. Choices are scored in the original and the reversed order, with the writer and reader
using the same order. --writer-choice-order fixed-original keeps the writer's original order as a separate control.

    python b4_think.py train --reader soft --k 4 --seed 17 --steps 3000 --out runs/b4/soft-k4-s17
    python b4_think.py score --eval mmau --out runs/b4/soft-k4-s17
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from b2_mixtures import Backbone
from b3_latent import add_lora

LETTERS = "ABCDEFGH"
SETS = {"avqa-train": "data/reasoning/avqa/train.json", "avqa-dev": "data/reasoning/avqa/dev.json", "avqa-valid": "data/reasoning/avqa/valid.json",
        "mmau": "data/reasoning/mmau/test.json", "mmar": "data/reasoning/mmar/test.json"}


def prompt(question, choices):
    return question + "\n" + "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, choices)) + "\nAnswer with only the letter."


class Think(nn.Module):
    """New-token embeddings start at the mean token embedding plus small noise (in-distribution for the frozen LLM)."""

    def __init__(self, center, k, std):
        super().__init__()
        width = len(center)
        self.start = nn.Parameter(center + torch.randn(width, device="cuda") * 0.1 * std)
        self.end = nn.Parameter(center + torch.randn(width, device="cuda") * 0.1 * std)
        self.queries = nn.Parameter(center + torch.randn(k, width, device="cuda") * 0.1 * std)
        self.scale = nn.Parameter(torch.ones(width, device="cuda"))
        self.bias = nn.Parameter(torch.zeros(width, device="cuda"))


class Reasoner:
    def __init__(self, model, config):
        self.m, self.c = model, config
        weight = model.embed.weight.float()
        self.think = Think(weight.mean(0), config["k"], weight.std().item()) if config["think"] else None
        self.lora = add_lora(model.lm) if config["lora"] else None
        width = model.embed.embedding_dim
        self.head = nn.Linear(width, width, device="cuda") if config.get("recon") else None
        self.letters = [model.proc.tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS]

    def user(self, audio, text):
        content = ([{"type": "audio", "audio_url": "placeholder"}] if audio is not None else []) + \
                  ([{"type": "text", "text": text}] if text is not None else [])
        template = self.m.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        tokens = self.m.proc.tokenizer(template, return_tensors="pt").input_ids.cuda()
        embeds = self.m.embed(tokens)
        if audio is None:
            return embeds
        i = (tokens[0] == self.m.base.config.audio_token_index).nonzero().item()
        return torch.cat((embeds[:, :i], audio[None].to(embeds.dtype), embeds[:, i+1:]), dim=1)

    def hidden(self, audio, text):
        prefix = self.user(audio, None if self.c["blind_writer"] else text)
        sequence = torch.cat((prefix, self.think.start[None, None].to(prefix.dtype), self.think.queries[None].to(prefix.dtype)), 1)
        return self.m.lm.model(inputs_embeds=sequence, use_cache=False).last_hidden_state[0, -self.c["k"]:].float()

    def write(self, audio, text):
        if self.think is None or not self.c["k"]:
            return None
        return self.hidden(audio, text) * self.think.scale + self.think.bias

    def read(self, audio, text, latents, n):
        parts = [self.user(audio, text)]
        if self.think is not None:
            dtype = parts[0].dtype
            parts.append(self.think.start[None, None].to(dtype))
            if latents is not None:
                parts.append(latents[None].to(dtype))
            parts.append(self.think.end[None, None].to(dtype))
        return self.m.lm(inputs_embeds=torch.cat(parts, 1), use_cache=False).logits[0, -1, self.letters[:n]].float()

    def params(self):
        groups = []
        if self.think is not None:
            groups.append(dict(params=list(self.think.parameters()), lr=1e-3, weight_decay=0.0))
        if self.lora is not None:
            groups.append(dict(params=list(self.lora.parameters()), lr=1e-4, weight_decay=0.0))
        if self.head is not None:
            groups.append(dict(params=list(self.head.parameters()), lr=1e-3, weight_decay=0.0))
        return groups

    def reconstruction(self, latents, audio):
        """Relative MSE between head(latents) and the K contiguous chunk means of the audio tokens."""
        target = torch.stack([chunk.mean(0) for chunk in audio.float().chunk(len(latents))])
        return F.mse_loss(self.head(latents), target) / target.pow(2).mean()

    def state(self):
        return {"think": self.think.state_dict() if self.think else None, "lora": self.lora.state_dict() if self.lora else None,
                "head": self.head.state_dict() if self.head else None}

    def load(self, state):
        if self.think is not None:
            self.think.load_state_dict(state["think"])
        if self.lora is not None:
            self.lora.load_state_dict(state["lora"])
        if self.head is not None:
            self.head.load_state_dict(state["head"])


def load(name, limit=0):
    items = json.loads(Path(SETS[name]).read_text())
    return items[:limit] if limit else items


def score(r, items, out, cache, seed, writer_choice_order="matched"):
    """Per item, order and condition: the predicted choice (mapped back to the original order) and all logits."""
    reader_audio = r.c["reader"] in ("soft", "mid")
    n = len(items)
    orders = []
    for step in (1, -1):
        indices = [list(range(len(x["choices"])))[::step] for x in items]
        texts = [prompt(x["question"], [x["choices"][j] for j in index]) for x, index in zip(items, indices)]
        orders.append((indices, texts))
    writer_orders = (0, 1) if writer_choice_order == "matched" and not r.c["blind_writer"] else (0, 0)
    with torch.no_grad():
        latents = {order: [r.write(cache(x["path"]), text) for x, text in zip(items, orders[order][1])]
                   for order in sorted(set(writer_orders))}
        has_latents = latents[0][0] is not None
        stacks = {order: torch.stack(values) for order, values in latents.items()} if has_latents else {}
        means = {order: s.mean(0) for order, s in stacks.items()}
        scales = {order: s.pow(2).mean().sqrt() for order, s in stacks.items()}
    generator = torch.Generator(device="cuda").manual_seed(seed)
    is_base = not has_latents and r.think is None and r.lora is None
    audio_conditions = (True, False) if reader_audio or is_base else (False,)
    partial = out.with_suffix(".partial")
    with partial.open("w", newline="") as handle, torch.no_grad():
        writer = csv.writer(handle)
        writer.writerow(["item", "id", "task", "category", "order", "condition", "audio", "answer", "pred", "logits", "writer_order"])
        for i, x in enumerate(items):
            audio = cache(x["path"])
            for order, (indices, texts) in enumerate(orders):
                index = indices[i]
                writer_order = writer_orders[order]
                values = latents[writer_order]
                conditions = {"normal": values[i]}
                if has_latents:
                    mean, rms = means[writer_order], scales[writer_order]
                    noise = torch.randn(values[i].shape, device="cuda", generator=generator)
                    conditions.update(swap=values[(i + n // 2) % n], mean=mean, zero=torch.zeros_like(mean),
                                      noise=values[i] + rms * noise)
                for name, latent in conditions.items():
                    for with_audio in audio_conditions:
                        logits = r.read(audio if with_audio else None, texts[i], latent, len(index))
                        pred = index[int(logits.argmax())]
                        back = [0.0] * len(index)
                        for slot, j in enumerate(index):
                            back[j] = float(logits[slot])
                        writer.writerow([i, x["id"], x["task"], x["category"], order, name, int(with_audio), x["answer"], pred,
                                         ";".join(f"{v:.3f}" for v in back), writer_order if has_latents else ""])
    partial.replace(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "score"])
    parser.add_argument("--reader", choices=["soft", "hard", "mid"], default="soft")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--blind-writer", action="store_true")
    parser.add_argument("--drop", type=float, default=0.5)
    parser.add_argument("--recon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--train", default="avqa-train")
    parser.add_argument("--eval", nargs="+", default=["avqa-dev"])
    parser.add_argument("--limit", type=int, default=0, help="score only the first N items of each set (smoke tests)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=Path("cache/features"))
    parser.add_argument("--writer-choice-order", choices=["matched", "fixed-original"], default="matched",
                        help="Use matching writer and reader choice orders, or always write from the original order")
    args = parser.parse_args()
    if args.action == "score":
        config = json.loads((args.out / "train.json").read_text())["config"]
    else:
        if (args.out / "state.pt").exists():
            parser.error("Choose a fresh output directory")
        config = dict(reader=args.reader, k=0 if args.no_think else args.k, think=not args.no_think, lora=not args.no_lora,
                      blind_writer=args.blind_writer, drop=args.drop if args.reader == "mid" else None, seed=args.seed,
                      steps=args.steps, train=args.train, recon=args.recon, lora_spec=dict(rank=8, alpha=16, modules="q_proj,v_proj"))
    model = Backbone(args.feature_cache)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    memory = {}

    def cache(path):
        if path not in memory:
            memory[path] = model.features(path).cpu()
        return memory[path].cuda()

    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    r = Reasoner(model, config)
    if args.action == "score":
        r.load(torch.load(args.out / "state.pt", weights_only=True))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        train = load(config["train"]) if config["steps"] else []
        result = {"config": config, "train_items": len(train)}
        if r.think is not None and config["k"]:
            with torch.no_grad():  # start the latents at the audio tokens' per-dimension scale
                probe = train[:16] or load(args.eval[0], 16)
                hidden = torch.cat([r.hidden(cache(x["path"]), prompt(x["question"], x["choices"])) for x in probe])
                audio = torch.cat([cache(x["path"]) for x in probe])
                r.think.scale.copy_(audio.pow(2).mean(0).sqrt() / hidden.pow(2).mean(0).sqrt().clamp_min(1e-6))
        params = [p for g in r.params() for p in g["params"]]
        losses, recs, start = [], [], time.time()
        if config["steps"]:
            opt = torch.optim.AdamW(r.params())
            torch.cuda.reset_peak_memory_stats()
            model.lm.train()
            order = list(train)
            for step in range(config["steps"]):
                if step % len(order) == 0:
                    rng.shuffle(order)
                x = order[step % len(order)]
                index = list(range(len(x["choices"])))
                rng.shuffle(index)  # no letter can stand for the answer
                text = prompt(x["question"], [x["choices"][j] for j in index])
                audio = cache(x["path"])
                see = config["reader"] == "soft" or (config["reader"] == "mid" and rng.random() >= config["drop"])
                latents = r.write(audio, text)
                logits = r.read(audio if see else None, text, latents, len(index))
                loss = F.cross_entropy(logits[None], torch.tensor([index.index(x["answer"])], device="cuda"))
                if r.head is not None:
                    rec = r.reconstruction(latents, audio)
                    recs.append(rec.item())
                    loss = loss + config["recon"] * rec
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                opt.step()
                losses.append(loss.item())
                if (step + 1) % 200 == 0:
                    print(f"step {step+1} loss {sum(losses[-200:])/200:.4f} " + (f"recon {sum(recs[-200:])/200:.3f} " if recs else "") +
                          f"{time.time()-start:.0f}s "
                          f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
            model.lm.eval()
        torch.save(r.state(), args.out / "state.pt")
        result.update(recon_by_200=[sum(recs[i:i+200]) / len(recs[i:i+200]) for i in range(0, len(recs), 200)] or None,
                      loss_by_200=[sum(losses[i:i+200]) / len(losses[i:i+200]) for i in range(0, len(losses), 200)],
                      train_seconds=time.time() - start,
                      peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30 if losses else None)
        (args.out / "train.json").write_text(json.dumps(result, indent=1))
    model.lm.eval()
    for name in args.eval:
        score(r, load(name, args.limit), args.out / f"scores-{name}.csv", cache, config["seed"], args.writer_choice_order)
        print("scored", name, flush=True)


if __name__ == "__main__":
    main()
