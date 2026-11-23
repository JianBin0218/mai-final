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

from baseline import MODEL_DIR, waveform
from prepare import ESC, download


GENERIC_STEM = "What can be heard in this recording?"


class Compressor(nn.Module):
    """K learned queries attend over projected audio features.

    condition="add" adds a linear projection of the question to every query (the original pilot).
    condition="gated" bounds that term: it is unit-normalized and scaled by tanh of a zero-initialized gate.
    """
    def __init__(self, width, k, condition="add"):
        super().__init__()
        self.keys = nn.Linear(width, 64, bias=False)
        self.queries = nn.Parameter(torch.randn(k, 64) * 0.1)
        self.condition = nn.Linear(width, 64, bias=False)
        self.mode = condition
        if condition == "gated":
            self.gate = nn.Parameter(torch.zeros(()))
        self.magnitude = None  # probe mode: replace tanh(gate) with a fixed norm for the question term

    def question_term(self, question):
        term = self.condition(question)
        if self.mode == "gated":
            scale = torch.tanh(self.gate) if self.magnitude is None else self.magnitude
            term = scale * F.normalize(term, dim=-1)
        return term

    def forward(self, audio, question=None):
        k = len(self.queries)
        queries = self.queries
        if question is not None:
            queries = queries + self.question_term(question)
        weights = (queries @ self.keys(F.layer_norm(audio, audio.shape[-1:])).T / math.sqrt(64)).softmax(-1)
        pooled = F.adaptive_avg_pool1d(audio.T.unsqueeze(0), k)[0].T
        return pooled + weights @ audio - audio.mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "pool", "probe", "fresh", "controls",
                                        "intervene", "opposite", "level-probe", "count-evidence",
                                        "count-prompts", "count-confirmation", "natural-capacity",
                                        "natural-components", "event-components", "a-final", "level-contrast"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clotho", action="store_true")
    parser.add_argument("--conditioned", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--init-from", type=Path, help="Warm-start from an audio-only compressor; reset AdamW and zero the question projection")
    parser.add_argument("--condition", choices=["add", "gated"], default="add",
                        help="Question term: original additive projection, or unit-normalized with a zero-init tanh gate")
    parser.add_argument("--center-question", action="store_true",
                        help="Subtract the mean training-stem embedding before normalizing the question vector")
    parser.add_argument("--question-repr", choices=["embed", "hidden"], default="embed",
                        help="Question vector: mean input token embedding, or mean hidden state of the frozen LM")
    parser.add_argument("--question-layer", type=int, default=16, help="LM layer for --question-repr hidden")
    parser.add_argument("--gate-lr", type=float, default=0.01, help="Learning rate of the gate in --condition gated")
    parser.add_argument("--manifest", type=Path, default=Path("data/clotho/manifest.json"))
    parser.add_argument("--train-split", default="train", help="Manifest split used for training")
    parser.add_argument("--extra-split", help="Additional manifest split to evaluate, e.g. val_extra")
    parser.add_argument("--extra-manifest", type=Path,
                        help="Read --extra-split from this file instead, e.g. data/clotho/fresh_val.json with split val")
    parser.add_argument("--feature-cache", type=Path, help="Directory for cached projected audio features")
    parser.add_argument("--model-gpu-memory-gib", type=int,
                        help="GPU weight budget in GiB when keeping some model weights in CPU memory")
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2, 4, 8, 16],
                        help="probe: fixed norms of the question term, evaluated with correct and wrong stems")
    parser.add_argument("--pool-ks", type=int, nargs="+", default=[4, 16], help="Slot counts for the fixed pooling evaluation")
    parser.add_argument("--intervention", choices=["swap","components","factorial"], default="swap")
    parser.add_argument("--donor-seed", type=int, default=17)
    parser.add_argument("--types", type=Path, default=Path("phase2/fresh-capacity/question_types.csv"))
    parser.add_argument("--pair-manifest", type=Path, default=Path("data/paired_evidence/pairs.json"))
    parser.add_argument("--batch-size", type=int, default=20, help="Number of clips per count-prompts batch")
    parser.add_argument("--count-prompt-format", choices=["yes-no", "descriptions"], default="yes-no",
                        help="Answer choice format for count-prompts")
    parser.add_argument("--confirmation-plan", type=Path, help="Fixed plan for count-confirmation")
    parser.add_argument("--confirmation-stimuli", type=Path, help="Held-out stimuli root for count-confirmation")
    parser.add_argument("--capacity-plan", type=Path, help="Fixed plan for natural-capacity")
    parser.add_argument("--capacity-references", type=Path, help="Saved natural-capacity scores for natural-components")
    parser.add_argument("--event-plan", type=Path, help="Fixed plan for event-components")
    parser.add_argument("--final-plan", type=Path, help="Fixed A-line evaluation plan")
    parser.add_argument("--level-stage", choices=("screen", "confirm"), help="Stage for level-contrast")
    parser.add_argument("--level-lock", type=Path, help="Locked screen outcome for level-contrast confirmation")
    args = parser.parse_args()
    audio_only_modes = ("fresh", "controls", "intervene", "count-evidence", "count-prompts",
                        "count-confirmation", "natural-capacity", "natural-components", "event-components", "a-final",
                        "level-contrast")
    if args.mode in audio_only_modes and (not args.clotho or args.conditioned):
        parser.error(f"{args.mode} requires --clotho and audio-only checkpoints")
    if args.mode in ("count-prompts", "count-confirmation") and args.batch_size < 1:
        parser.error(f"{args.mode} requires a positive --batch-size")
    if args.mode == "count-confirmation" and (args.confirmation_plan is None or args.confirmation_stimuli is None):
        parser.error("count-confirmation requires --confirmation-plan and --confirmation-stimuli")
    if args.mode == "natural-capacity" and args.capacity_plan is None:
        parser.error("natural-capacity requires --capacity-plan")
    if args.mode == "natural-components" and args.capacity_references is None:
        parser.error("natural-components requires --capacity-references")
    if args.mode == "event-components" and args.event_plan is None:
        parser.error("event-components requires --event-plan")
    if args.mode == "a-final" and args.final_plan is None:
        parser.error("a-final requires --final-plan")
    if args.mode == "level-contrast" and (args.level_stage is None or args.batch_size < 1
                                         or (args.level_stage == "confirm") != (args.level_lock is not None)):
        parser.error("level-contrast requires --level-stage, a positive --batch-size, and --level-lock only for confirm")
    if args.mode == "pool" and (not args.clotho or args.conditioned or min(args.pool_ks) < 1):
        parser.error("pool requires --clotho and positive slot counts without --conditioned")
    if args.mode == "probe" and (not args.conditioned or args.condition != "gated"):
        parser.error("probe requires --conditioned --condition gated")
    if args.init_from and (args.mode != "train" or not args.clotho):
        parser.error("--init-from requires train --clotho")
    args.out.mkdir(exist_ok=True, parents=True)
    checkpoint = args.out / "compressor.pt"
    if args.mode == "train" and checkpoint.exists():
        parser.error("Choose a fresh output directory to preserve the previous run.")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    placement = {"device_map": {"": 0}}
    if args.model_gpu_memory_gib is not None:
        placement = {"device_map": "auto", "max_memory": {0: f"{args.model_gpu_memory_gib}GiB", "cpu": "20GiB"}}
    base = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, **placement
    ).eval().requires_grad_(False)
    lm = base.language_model
    embed = lm.get_input_embeddings()
    comp = Compressor(embed.embedding_dim, args.k, args.condition).cuda()
    choices = [proc.tokenizer.encode(x, add_special_tokens=False) for x in "AB"]
    assert all(len(x) == 1 for x in choices)
    ids = [x[0] for x in choices]
    cache = {}
    donor_stems = {}
    suffix = "\nA. yes\nB. no\nAnswer with only A or B."
    if args.clotho:
        manifest = json.loads(args.manifest.read_text())
        def samples(split):
            rows = []
            for item in manifest[split]:
                question = item["question"] + suffix
                donor_stems[item["path"], question] = item["donor"]
                rows.append((item["path"], question, item["label"]))
            return rows
        train, test = samples(args.train_split), samples("val")
        if args.mode in ("fresh","controls","intervene"):
            previous = {r[0] for r in train + test}
            manifest = json.loads(Path("data/clotho/fresh_val.json").read_text())
            test = samples("val")
            assert test and not previous.intersection(r[0] for r in test)
        extra = []
        if args.extra_split:
            if args.extra_manifest:
                manifest = json.loads(args.extra_manifest.read_text())
            extra = samples(args.extra_split)
            assert not {r[0] for r in train + test}.intersection(r[0] for r in extra)
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
        extra = []

    def cache_file(path):
        return args.feature_cache / (hashlib.sha1(path.encode()).hexdigest() + ".pt")

    class AudioFeaturesCaptured(Exception):
        pass

    @torch.no_grad()
    def features(path):
        if path not in cache and args.feature_cache and cache_file(path).exists():
            cache[path] = torch.load(cache_file(path), weights_only=True)
        if path not in cache:
            prompt = proc.apply_chat_template([{"role":"user","content":[{"type":"audio","audio_url":path}]}],
                                                tokenize=False, add_generation_prompt=True)
            inputs = proc(text=prompt, audio=[waveform(path)], sampling_rate=16000, return_tensors="pt").to("cuda")
            inputs["input_features"] = inputs["input_features"].to(base.dtype)
            mask = inputs["input_ids"][0] == base.config.audio_token_index
            def capture(module, positional, kwargs):
                cache[path] = kwargs["inputs_embeds"][0, mask].detach().cpu()
                if args.mode in ("count-confirmation", "natural-capacity", "natural-components", "event-components", "a-final",
                                 "level-contrast"):
                    raise AudioFeaturesCaptured
            hook = lm.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                base(**inputs, use_cache=False)
            except AudioFeaturesCaptured:
                pass
            finally:
                hook.remove()
            assert cache[path].shape[0] > 0
            if args.feature_cache:
                args.feature_cache.mkdir(parents=True, exist_ok=True)
                tmp = cache_file(path).with_suffix(f".{torch.cuda.current_device()}.{id(cache)}.part")
                torch.save(cache[path], tmp)
                tmp.replace(cache_file(path))
        return cache[path].cuda().float()

    def decode(latent, question):
        messages = [{"role":"user","content":[{"type":"audio","audio_url":"placeholder"},
                                                {"type":"text","text":question}]}]
        template = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = proc.tokenizer(template, return_tensors="pt").input_ids.cuda()
        index = (tokens[0] == base.config.audio_token_index).nonzero().item()
        text = embed(tokens)
        if latent.ndim == 3:
            text = text.expand(latent.shape[0], -1, -1)
            sequence = torch.cat((text[:, :index], latent.to(text.dtype), text[:, index+1:]), dim=1)
            return lm(inputs_embeds=sequence, use_cache=False, logits_to_keep=1).logits[:, -1, ids].float()
        sequence = torch.cat((text[:, :index], latent[None].to(text.dtype), text[:, index+1:]), dim=1)
        return lm(inputs_embeds=sequence, use_cache=False).logits[0,-1,ids].float()

    stem_cache = {}

    def pooled_stem(stem):
        tokens = proc.tokenizer(stem, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        if args.question_repr == "embed":
            return embed(tokens).float().mean(dim=1)
        if stem not in stem_cache:
            with torch.no_grad():
                hidden = lm(input_ids=tokens, output_hidden_states=True, use_cache=False).hidden_states
                stem_cache[stem] = hidden[args.question_layer].float().mean(dim=1)
        return stem_cache[stem]

    center = None
    if args.center_question:
        with torch.no_grad():
            stems = sorted({q.split("\n")[0] for _, q, _ in train})
            center = torch.cat([pooled_stem(stem) for stem in stems]).mean(0, keepdim=True)

    def question_vector(stem):
        vector = pooled_stem(stem)
        if center is not None:
            vector = vector - center
        return F.layer_norm(vector, (embed.embedding_dim,))

    def compress(path, question, swap=False, *, conditioned=None, stem=None):
        condition = None
        if conditioned is None:
            conditioned = args.conditioned
        if conditioned:
            if stem is None:
                stem = donor_stems[path, question] if swap else question.split("\n")[0]
            condition = question_vector(stem)
        return comp(features(path), condition)

    def decode_text(question):
        template = proc.apply_chat_template([{"role":"user","content":[{"type":"text","text":question}]}],
                                            tokenize=False, add_generation_prompt=True)
        tokens = proc.tokenizer(template, return_tensors="pt").input_ids.cuda()
        return lm(input_ids=tokens, use_cache=False).logits[0, -1, ids].float()

    @torch.no_grad()
    def evaluate(samples, swap=False, full=False, text_only=False, pool_k=None, donors=None, stem=None, conditioned=None):
        rows = []
        for path, question, target in samples:
            if text_only:
                scores = decode_text(question)
            else:
                if pool_k is not None:
                    audio = features(path)
                    latent = F.adaptive_avg_pool1d(audio.T.unsqueeze(0), pool_k)[0].T
                else:
                    latent = features(donors[path] if donors else path) if full else compress(
                        path, question, swap, conditioned=conditioned, stem=stem)
                scores = decode(latent, question)
            rows.append(dict(path=path, question=question, target=target, prediction=scores.argmax().item(),
                             loss=F.cross_entropy(scores[None], torch.tensor([target],device="cuda")).item()))
        return dict(correct=sum(r["prediction"]==r["target"] for r in rows), n=len(rows),
                    loss=sum(r["loss"] for r in rows)/len(rows), rows=rows)

    if args.mode == "level-probe":
        from level_probe import probe
        probe(args, comp, features)
        return
    if args.mode == "level-contrast":
        from level_contrast import measure
        measure(args, comp, features, decode)
        return
    if args.mode == "opposite":
        from opposite_eval import measure
        measure(args, comp, features, decode)
        return

    if args.mode == "count-evidence":
        from count_evidence import measure
        measure(args, comp, features, decode)
        return

    if args.mode == "count-prompts":
        from count_prompt_eval import measure
        measure(args, comp, features, decode)
        return

    if args.mode == "count-confirmation":
        from count_confirmation import measure
        measure(args, comp, features, decode)
        return

    if args.mode == "natural-capacity":
        from natural_capacity import measure
        measure(args, comp, features, decode)
        return

    if args.mode == "natural-components":
        from natural_components import measure
        measure(args, comp, features, decode)
        return

    if args.mode == "event-components":
        from event_components import measure
        measure(args, comp, features, decode)
        return

    if args.mode == "a-final":
        from a_final_test import measure
        measure(args, comp, features, decode, decode_text)
        return

    if args.mode == "intervene":
        from latent_interventions import run_interventions
        run_interventions(args, test, comp, features, decode)
        return

    if args.mode == "controls":
        clips = sorted({row[0] for row in test})
        random.Random(17).shuffle(clips)
        donors = dict(zip(clips, clips[1:] + clips[:1]))
        assert all(k != v for k,v in donors.items())
        (args.out / "donors.json").write_text(json.dumps(donors))
        for name, kwargs in (("text", {"text_only":True}), ("full", {"full":True}),
                             ("unrelated", {"full":True, "donors":donors})):
            scored = evaluate(test, **kwargs)
            (args.out / (name+".json")).write_text(json.dumps(scored))
            print(name, scored["correct"], scored["n"], flush=True)
        return

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
        missing, unexpected = comp.load_state_dict(torch.load(args.init_from, weights_only=True), strict=False)
        assert not unexpected and set(missing) <= {"gate"}, (missing, unexpected)
        if args.condition == "add":
            nn.init.zeros_(comp.condition.weight)
        # In gated mode the zero gate already silences the question term; the projection keeps the parent's init.
    if args.mode in ("eval", "probe"):
        comp.load_state_dict(torch.load(checkpoint, weights_only=True))
    if args.mode == "probe":
        samples_ = extra or test
        query_norm = comp.queries.norm(dim=-1).mean().item()
        probe = dict(query_norm=query_norm, learned_gate=torch.tanh(comp.gate).item(),
                     split=args.extra_split or "val", manifest=str(args.extra_manifest or args.manifest), magnitudes=[])
        for magnitude in args.magnitudes:
            comp.magnitude = magnitude
            right, wrong = evaluate(samples_), evaluate(samples_, swap=True)
            probe["magnitudes"].append(dict(magnitude=magnitude, ratio_to_query=magnitude/query_norm,
                correct_stem=right["correct"], wrong_stem=wrong["correct"], n=right["n"],
                yes_rate=sum(r["prediction"] == 0 for r in right["rows"])/right["n"],
                same_prediction=sum(a["prediction"] == b["prediction"] for a, b in zip(right["rows"], wrong["rows"]))))
            print("PROBE", json.dumps(probe["magnitudes"][-1]), flush=True)
        name = f"probe-{args.extra_manifest.stem}.json" if args.extra_manifest else "probe.json"
        (args.out / name).write_text(json.dumps(probe, indent=2))
        return
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
            params = [comp.condition.weight] + ([comp.gate] if args.condition == "gated" else [])
            grads = torch.autograd.grad(loss, params)
            assert all(torch.isfinite(g).all() for g in grads) and sum(g.norm() for g in grads).item() > 0
            result["warm_start"] = dict(checkpoint=str(args.init_from),
                sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
                steps=args.steps, checked_questions=len(test), max_logit_delta=max_delta,
                question_grad_norm=grads[0].norm().item())
            if args.condition == "gated":
                result["warm_start"]["gate_grad"] = grads[1].item()
            result["heldout_before"] = evaluate(test)
        result["before"] = evaluate(train)
        groups = [dict(params=[p for n, p in comp.named_parameters() if n != "gate"])]
        if args.condition == "gated":
            groups.append(dict(params=[comp.gate], lr=args.gate_lr, weight_decay=0.0))
        opt = torch.optim.AdamW(groups, lr=0.001)
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
                metric = dict(step=step+1, loss=loss.item(), gradient_norm=norm.item())
                if args.condition == "gated":
                    metric["gate"] = torch.tanh(comp.gate).item()
                print("METRIC", json.dumps(metric), flush=True)
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
    if args.condition != "add" or args.center_question or args.extra_split or args.question_repr != "embed":
        result["config"] = dict(condition=args.condition, center_question=args.center_question,
                                question_repr=args.question_repr, question_layer=args.question_layer,
                                gate_lr=args.gate_lr, manifest=str(args.manifest),
                                train_split=args.train_split, extra_split=args.extra_split,
                                extra_manifest=str(args.extra_manifest) if args.extra_manifest else None)
    if args.conditioned and (args.condition != "add" or args.center_question or args.question_repr != "embed"):
        if args.condition == "gated":
            result["gate"] = torch.tanh(comp.gate).item()
        result["generic_question"] = evaluate(test, stem=GENERIC_STEM)
        result["no_question"] = evaluate(test, conditioned=False)
    if extra:
        result["extra"] = dict(heldout=evaluate(extra))
        if args.conditioned:
            result["extra"]["wrong_question"] = evaluate(extra, swap=True)
            result["extra"]["generic_question"] = evaluate(extra, stem=GENERIC_STEM)
            result["extra"]["no_question"] = evaluate(extra, conditioned=False)
        else:
            result["extra"]["full_audio"] = evaluate(extra, full=True)
            result["extra"]["text_only"] = evaluate(extra, text_only=True)
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
