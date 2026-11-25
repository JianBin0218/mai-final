"""Track B3: the LLM itself writes the K audio latents (two passes through Qwen2-Audio's language model).

Pass 1 reads [audio tokens][question] in the chat template and then K learnable latent tokens in the assistant
turn; the final hidden states at those K positions, rescaled per dimension, become the latents. Pass 2 is the
same decoder used everywhere else (b2_mixtures.Backbone.decode): it sees the question, the answer options and
only the K latents in place of the audio. Nothing else carries audio into pass 2.

Arms
  internal       LLM frozen; trains the K latent tokens and the per-dimension output scale
  internal-lora  as above plus LoRA (q and v projections, all layers) shared by both passes
  full-lora      no bottleneck: full audio features into pass 2, same LoRA; the ceiling for internal-lora
  internal-blind LLM frozen, pass 1 sees the audio but no question (the LLM as a question-agnostic compressor)

    python b3_latent.py train --data synthetic --arm internal --k 4 --seed 17 --out runs/b3/syn-k4-s17-internal
    python b3_latent.py score --data synthetic --split confirm --out runs/b3/syn-k4-s17-internal

bottleneck.py is not modified (track A's locked final evaluation pins its hash).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from b2_mixtures import SUFFIX, Backbone
from b2_train import load_data, score


class LoRA(nn.Module):
    """y = base(x) + (alpha / r) * B(A(x)), with B initialized to zero so training starts at the frozen model."""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (self.up(self.down(x.float())) * self.scale).to(x.dtype)


def add_lora(lm, rank=8, alpha=16):
    modules = []
    for layer in lm.model.layers:
        for name in ("q_proj", "v_proj"):
            wrapped = LoRA(getattr(layer.self_attn, name), rank, alpha)
            setattr(layer.self_attn, name, wrapped)
            modules.append(wrapped)
    return nn.ModuleList(modules)


class Writer(nn.Module):
    """K learnable token embeddings appended after the question; their final hidden states become the latents."""

    def __init__(self, model, k):
        super().__init__()
        self.model = model
        width = model.embed.embedding_dim
        std = model.embed.weight.float().std().item()
        self.tokens = nn.Parameter(torch.randn(k, width, device="cuda") * std)
        self.scale = nn.Parameter(torch.ones(width, device="cuda"))
        self.bias = nn.Parameter(torch.zeros(width, device="cuda"))
        self.templates = {}

    def template(self, stem):
        if stem not in self.templates:
            proc = self.model.proc
            content = [{"type": "audio", "audio_url": "placeholder"}]
            if stem is not None:
                content.append({"type": "text", "text": stem})
            text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            tokens = proc.tokenizer(text, return_tensors="pt").input_ids.cuda()
            index = (tokens[0] == self.model.base.config.audio_token_index).nonzero().item()
            self.templates[stem] = (tokens, index)
        return self.templates[stem]

    def hidden(self, audio, stem):
        tokens, index = self.template(stem)
        text = self.model.embed(tokens)
        sequence = torch.cat((text[:, :index], audio[None].to(text.dtype), text[:, index+1:],
                              self.tokens[None].to(text.dtype)), dim=1)
        out = self.model.lm.model(inputs_embeds=sequence, use_cache=False).last_hidden_state
        return out[0, -len(self.tokens):].float()

    def forward(self, audio, stem):
        return self.hidden(audio, stem) * self.scale + self.bias


class Arm:
    def __init__(self, model, arm, k):
        self.model, self.name = model, arm
        # b2_train.score scores only the "correct" condition for "audio"; the stem conditions are moot for these arms.
        self.arm = "audio" if arm in ("full-lora", "internal-blind") else arm
        self.blind = arm == "internal-blind"
        self.writer = Writer(model, k) if arm != "full-lora" else None
        self.lora = add_lora(model.lm) if arm.endswith("lora") else None

    def latent(self, audio, stem):
        if self.writer is None:
            return audio
        return self.writer(audio, None if self.blind else stem)

    def trainable(self):
        groups = []
        if self.writer is not None:
            groups.append(dict(params=list(self.writer.parameters()), lr=1e-3, weight_decay=0.0))
        if self.lora is not None:
            groups.append(dict(params=list(self.lora.parameters()), lr=1e-4, weight_decay=0.0))
        return groups

    def state(self):
        return {"writer": self.writer.state_dict() if self.writer else None,
                "lora": self.lora.state_dict() if self.lora else None}

    def load(self, state):
        if self.writer is not None:
            self.writer.load_state_dict(state["writer"])
        if self.lora is not None:
            self.lora.load_state_dict(state["lora"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "score"])
    parser.add_argument("--data", choices=["synthetic", "clotho"], required=True)
    parser.add_argument("--arm", choices=["internal", "internal-lora", "full-lora", "internal-blind"])
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--split", default="dev", help="synthetic scoring split: dev or confirm")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N evaluation items (smoke tests)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=Path("cache/features"))
    args = parser.parse_args()
    if args.action == "score":
        config = json.loads((args.out / "train.json").read_text())["config"]
        args.arm, args.k, args.seed = config["arm"], config["k"], config["seed"]
    elif (args.out / "state.pt").exists():
        parser.error("Choose a fresh output directory")
    model = Backbone(args.feature_cache)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    memory = {}

    def cache(path):
        if path not in memory:
            memory[path] = model.features(path).cpu()
        return memory[path].cuda()

    train, evaluation = load_data(args.data, args.split if args.data == "synthetic" else "dev")
    if args.limit:
        evaluation = evaluation[:args.limit]
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    arm = Arm(model, args.arm, args.k)
    if args.action == "score":
        arm.load(torch.load(args.out / "state.pt", weights_only=True))
        model.lm.eval()
        score(arm, evaluation, args.out / f"scores-{args.split}.csv", cache)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    result = {"config": dict(arm=args.arm, k=args.k, seed=args.seed, steps=args.steps, data=args.data,
                             train_questions=len(train), lora=dict(rank=8, alpha=16, modules="q_proj,v_proj")
                             if arm.lora is not None else None)}
    with torch.no_grad():
        probe = train[:16]
        if arm.writer is not None:
            # Start the latents at the audio tokens' per-dimension scale.
            hidden = torch.cat([arm.writer.hidden(cache(x["path"]), None if arm.blind else x["question"]) for x in probe])
            audio = torch.cat([cache(x["path"]) for x in probe])
            arm.writer.scale.copy_(audio.pow(2).mean(0).sqrt() / hidden.pow(2).mean(0).sqrt().clamp_min(1e-6))
            a = arm.latent(cache(probe[0]["path"]), probe[0]["question"])
            b = arm.latent(cache(probe[0]["path"]), probe[0]["donor"])
            other = next(x for x in train if x["path"] != probe[0]["path"])
            c = arm.latent(cache(other["path"]), probe[0]["question"])
            result["start_check"] = dict(latent_norm=a.norm(dim=-1).mean().item(),
                                         audio_token_norm=audio.norm(dim=-1).mean().item(),
                                         rel_change_other_question=((a - b).norm() / a.norm()).item(),
                                         rel_change_other_audio=((a - c).norm() / a.norm()).item())
        if arm.lora is not None:
            x = probe[0]
            before = model.decode(arm.latent(cache(x["path"]), x["question"]), x["question"] + SUFFIX[0])
            for module in arm.lora:
                module.scale, saved = 0.0, module.scale
                module.saved = saved
            frozen = model.decode(arm.latent(cache(x["path"]), x["question"]), x["question"] + SUFFIX[0])
            for module in arm.lora:
                module.scale = module.saved
            assert torch.equal(before, frozen), (before, frozen)
            result.setdefault("start_check", {})["lora_identity"] = True
    print("start check", result.get("start_check"), flush=True)
    opt = torch.optim.AdamW(arm.trainable())
    params = [p for g in arm.trainable() for p in g["params"]]
    order = list(train)
    losses, start = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    model.lm.train()
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        x = order[step % len(order)]
        logits = model.decode(arm.latent(cache(x["path"]), x["question"]), x["question"] + SUFFIX[0])
        loss = F.cross_entropy(logits[None], torch.tensor([x["label"]], device="cuda"))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 200 == 0:
            print(f"step {step+1} loss {sum(losses[-200:])/200:.4f} {time.time()-start:.0f}s "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    model.lm.eval()
    torch.save(arm.state(), args.out / "state.pt")
    result.update(loss_by_200=[sum(losses[i:i+200]) / len(losses[i:i+200]) for i in range(0, len(losses), 200)],
                  train_seconds=time.time() - start, peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30)
    if arm.writer is not None:
        result["scale_mean"] = arm.writer.scale.mean().item()
    (args.out / "train.json").write_text(json.dumps(result, indent=1))
    score(arm, evaluation, args.out / "scores-dev.csv", cache)
    print("done", args.out, flush=True)


if __name__ == "__main__":
    main()
