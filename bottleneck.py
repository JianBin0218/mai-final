"""Reproducible K-slot bottleneck smoke test on held-out ESC-50 recordings."""
import argparse
import hashlib
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from baseline import waveform
from prepare import ESC, download


class Compressor(nn.Module):
    def __init__(self, width, k):
        super().__init__()
        self.keys = nn.Linear(width, 64, bias=False)
        self.queries = nn.Parameter(torch.randn(k, 64) * 0.1)
        self.condition = nn.Linear(width, 64, bias=False)

    def forward(self, audio, question=None):
        k = len(self.queries)
        queries = self.queries
        if question is not None:
            queries = queries + self.condition(question)
        weights = (queries @ self.keys(F.layer_norm(audio, audio.shape[-1:])).T / math.sqrt(64)).softmax(-1)
        pooled = F.adaptive_avg_pool1d(audio.T.unsqueeze(0), k)[0].T
        return pooled + weights @ audio - audio.mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "pool", "fresh"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clotho", action="store_true")
    parser.add_argument("--conditioned", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--init-from", type=Path, help="Warm-start from an audio-only compressor; reset AdamW and zero the question projection")
    parser.add_argument("--pool-ks", type=int, nargs="+", default=[4, 16], help="Slot counts for the fixed pooling evaluation")
    args = parser.parse_args()
    if args.mode == "fresh" and (not args.clotho or args.conditioned):
        parser.error("fresh requires --clotho and audio-only checkpoints")
    if args.mode == "pool" and (not args.clotho or args.conditioned or min(args.pool_ks) < 1):
        parser.error("pool requires --clotho and positive slot counts without --conditioned")
    if args.init_from and (args.mode != "train" or not args.clotho):
        parser.error("--init-from requires train --clotho")
    args.out.mkdir(exist_ok=True, parents=True)
    checkpoint = args.out / "compressor.pt"
    if args.mode == "train" and checkpoint.exists():
        parser.error("Choose a fresh output directory to preserve the previous run.")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    proc = AutoProcessor.from_pretrained("model")
    base = Qwen2AudioForConditionalGeneration.from_pretrained("model", torch_dtype=torch.bfloat16,
                                                              device_map={"": 0}).eval().requires_grad_(False)
    lm = base.language_model
    embed = lm.get_input_embeddings()
    comp = Compressor(embed.embedding_dim, args.k).cuda()
    choices = [proc.tokenizer.encode(x, add_special_tokens=False) for x in "AB"]
    assert all(len(x) == 1 for x in choices)
    ids = [x[0] for x in choices]
    cache = {}
    donor_stems = {}
    suffix = "\nA. yes\nB. no\nAnswer with only A or B."
    if args.clotho:
        manifest = json.loads(Path("data/clotho/manifest.json").read_text())
        def samples(split):
            rows = []
            for item in manifest[split]:
                question = item["question"] + suffix
                donor_stems[item["path"], question] = item["donor"]
                rows.append((item["path"], question, item["label"]))
            return rows
        train, test = samples("train"), samples("val")
        if args.mode == "fresh":
            previous = {r[0] for r in train + test}
            manifest = json.loads(Path("data/clotho/fresh_val.json").read_text())
            test = samples("val")
            assert test and not previous.intersection(r[0] for r in test)
    else:
        with open("data/esc50.csv") as handle:
            metadata = list(csv.DictReader(handle))
        pairs = [p for p in json.loads(Path("data/pairs.json").read_text()) if p["task"] == "event"][:4]
        train, test = [], []
        for pair in pairs:
            q = pair["question"] + "\nA. " + pair["options"][0] + "\nB. " + pair["options"][1] + "\nAnswer with only A or B."
            for label, heldout in enumerate(pair["clips"]):
                category = pair["options"][label].replace(" ", "_")
                row = rng.choice([r for r in metadata if r["category"] == category and r["fold"] == "1"])
                path = Path("data") / row["filename"]
                download(ESC + "/audio/" + row["filename"], path)
                train.append((str(path), q, label))
                test.append((heldout, q, label))

    @torch.no_grad()
    def features(path):
        if path not in cache:
            prompt = proc.apply_chat_template([{"role":"user","content":[{"type":"audio","audio_url":path}]}],
                                                tokenize=False, add_generation_prompt=True)
            inputs = proc(text=prompt, audio=[waveform(path)], sampling_rate=16000, return_tensors="pt").to("cuda")
            inputs["input_features"] = inputs["input_features"].to(base.dtype)
            mask = inputs["input_ids"][0] == base.config.audio_token_index
            def capture(module, positional, kwargs):
                cache[path] = kwargs["inputs_embeds"][0, mask].detach().cpu()
            hook = lm.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                base(**inputs, use_cache=False)
            finally:
                hook.remove()
            assert cache[path].shape[0] > 0
        return cache[path].cuda().float()

    def decode(latent, question):
        messages = [{"role":"user","content":[{"type":"audio","audio_url":"placeholder"},
                                                {"type":"text","text":question}]}]
        template = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = proc.tokenizer(template, return_tensors="pt").input_ids.cuda()
        index = (tokens[0] == base.config.audio_token_index).nonzero().item()
        text = embed(tokens)
        sequence = torch.cat((text[:, :index], latent[None].to(text.dtype), text[:, index+1:]), dim=1)
        return lm(inputs_embeds=sequence, use_cache=False).logits[0,-1,ids].float()

    def compress(path, question, swap=False, *, conditioned=None):
        condition = None
        if conditioned is None:
            conditioned = args.conditioned
        if conditioned:
            stem = donor_stems[path, question] if swap else question.split("\n")[0]
            tokens = proc.tokenizer(stem, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            condition = F.layer_norm(embed(tokens).float().mean(dim=1), (embed.embedding_dim,))
        return comp(features(path), condition)

    @torch.no_grad()
    def evaluate(samples, swap=False, full=False, text_only=False, pool_k=None):
        rows = []
        for path, question, target in samples:
            if text_only:
                template = proc.apply_chat_template([{"role":"user","content":[{"type":"text","text":question}]}],
                                                    tokenize=False, add_generation_prompt=True)
                tokens = proc.tokenizer(template, return_tensors="pt").input_ids.cuda()
                scores = lm(input_ids=tokens,use_cache=False).logits[0,-1,ids].float()
            else:
                if pool_k is not None:
                    audio = features(path)
                    latent = F.adaptive_avg_pool1d(audio.T.unsqueeze(0), pool_k)[0].T
                else:
                    latent = features(path) if full else compress(path, question, swap)
                scores = decode(latent, question)
            rows.append(dict(path=path, question=question, target=target, prediction=scores.argmax().item(),
                             loss=F.cross_entropy(scores[None], torch.tensor([target],device="cuda")).item()))
        return dict(correct=sum(r["prediction"]==r["target"] for r in rows), n=len(rows),
                    loss=sum(r["loss"] for r in rows)/len(rows), rows=rows)

    if args.mode == "fresh":
        print("Fresh validation:", len(test), "questions", flush=True)
        results = {}
        for seed in (17,23,41):
            for k in (4,16):
                name = f"capacity-k{k}-s{seed}-audio"
                if (k,seed) == (16,17):
                    name = "clotho-k16-s17-audio"
                comp = Compressor(embed.embedding_dim,k).cuda()
                comp.load_state_dict(torch.load(Path(name)/"compressor.pt", weights_only=True))
                scored = evaluate(test)
                (args.out / (name+".json")).write_text(json.dumps(scored))
                results[name] = [scored["correct"],scored["n"]]
                print(name, results[name], flush=True)
        (args.out / "summary.json").write_text(json.dumps(results,indent=2))
        return

    if args.mode == "pool":
        pooled = {str(k):evaluate(test, pool_k=k) for k in args.pool_ks}
        (args.out / "pool.json").write_text(json.dumps(pooled, indent=2))
        return

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="mai-audio-bottleneck", name=args.out.name, config={
            "K":args.k, "seed":args.seed, "steps":args.steps, "conditioned":args.conditioned,
            "dataset":"Clotho-AQA" if args.clotho else "ESC-50", "learning_rate":0.001,
        })
        print("RUN_URL",run.url,flush=True)
    if args.init_from:
        comp.load_state_dict(torch.load(args.init_from, weights_only=True))
        nn.init.zeros_(comp.condition.weight)
    if args.mode == "eval":
        comp.load_state_dict(torch.load(checkpoint, weights_only=True))
    for path, _, _ in train + test:
        features(path)
    with torch.no_grad():
        a = train[0]
        b = next(row for row in train if row[0] != a[0])
        fixed = comp(features(a[0]))
        original = decode(fixed,a[1])
        alternative = comp(features(b[0]))
        assert not torch.equal(fixed, alternative)
        drift = (original - decode(fixed,a[1])).abs().max().item()
        assert drift == 0
    result = dict(isolation_AB_logit_delta=drift, k=args.k, seed=args.seed, gpu=torch.cuda.get_device_name())
    if args.mode == "train":
        if args.init_from:
            max_delta = 0.0
            with torch.no_grad():
                for path, question, _ in test:
                    audio = decode(compress(path, question, conditioned=False), question)
                    guided = decode(compress(path, question, conditioned=True), question)
                    torch.testing.assert_close(audio, guided, rtol=0, atol=0)
                    max_delta = max(max_delta, (audio-guided).abs().max().item())
            path, question, label = train[0]
            loss = F.cross_entropy(decode(compress(path, question, conditioned=True), question)[None],
                                   torch.tensor([label],device="cuda"))
            grad, = torch.autograd.grad(loss, comp.condition.weight)
            assert torch.isfinite(grad).all() and grad.norm().item() > 0
            result["warm_start"] = dict(checkpoint=str(args.init_from),
                sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
                steps=args.steps, checked_questions=len(test), max_logit_delta=max_delta,
                question_grad_norm=grad.norm().item())
            result["heldout_before"] = evaluate(test)
        result["before"] = evaluate(train)
        opt = torch.optim.AdamW(comp.parameters(), lr=0.001)
        for step in range(args.steps):
            if step % len(train) == 0:
                rng.shuffle(train)
            path, question, label = train[step % len(train)]
            scores = decode(compress(path, question),question)
            loss = F.cross_entropy(scores[None],torch.tensor([label],device="cuda"))
            opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(comp.parameters(),1.0,error_if_nonfinite=True)
            if step == 0:
                assert norm.item() > 0
                result["first_grad_norm"] = norm.item()
                old = comp.keys.weight.detach().clone()
            opt.step()
            if args.clotho:
                print("METRIC", json.dumps(dict(step=step+1, loss=loss.item(), gradient_norm=norm.item())), flush=True)
            if run:
                run.log({"train/loss":loss.item(), "train/gradient_norm":norm.item()},step=step+1)
            if step == 0:
                result["first_update_max"] = (comp.keys.weight.detach()-old).abs().max().item()
                assert result["first_update_max"] > 0
        torch.save(comp.state_dict(), checkpoint)
    result.update(train=evaluate(train), heldout=evaluate(test), peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    if args.conditioned:
        result["wrong_question"] = evaluate(test, swap=True)
    if args.clotho and not args.conditioned:
        result["full_audio"] = evaluate(test, full=True)
        result["text_only"] = evaluate(test, text_only=True)
    if run:
        for key in ("train","heldout","wrong_question","full_audio"):
            if key in result:
                run.summary[key+"/accuracy"] = result[key]["correct"]/result[key]["n"]
                run.summary[key+"/loss"] = result[key]["loss"]
        result["wandb_url"] = run.url
        run.finish()
    (args.out / (args.mode+".json")).write_text(json.dumps(result,indent=2))
    print(args.mode, result["train"]["loss"], result["heldout"]["correct"], result["peak_allocated_GiB"])


if __name__ == "__main__":
    main()
