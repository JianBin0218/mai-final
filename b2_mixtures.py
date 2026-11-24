"""Track B2.0: synthetic multi-event recordings for testing when question guidance can help.

    python b2_mixtures.py prepare --plan phase2/b2-mixtures-v1/plan.json --out data/b2mix
    python b2_mixtures.py control --plan phase2/b2-mixtures-v1/plan.json --manifest data/b2mix/manifest.json \
        --feature-cache cache/features --scores runs/b2-control/scores.csv
    python b2_mixtures.py report --plan phase2/b2-mixtures-v1/plan.json --scores phase2/b2-mixtures-v1/scores.csv

The manifest uses the Clotho manifest format (path, question, donor, label with 0 = yes), so the
existing training entry points can read its train_N* / dev_N* splits. The control action loads the model
itself and leaves bottleneck.py unchanged, because track A's locked final evaluation pins its hash.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

RATE = 16000
PHRASES = {
    "airplane": "an airplane", "breathing": "someone breathing", "brushing_teeth": "someone brushing their teeth",
    "can_opening": "a can being opened", "car_horn": "a car horn", "cat": "a cat meowing", "chainsaw": "a chainsaw",
    "chirping_birds": "birds chirping", "church_bells": "church bells", "clapping": "people clapping",
    "clock_alarm": "an alarm clock", "clock_tick": "a clock ticking", "coughing": "someone coughing",
    "cow": "a cow mooing", "crackling_fire": "a crackling fire", "crickets": "crickets", "crow": "a crow cawing",
    "crying_baby": "a baby crying", "dog": "a dog barking", "door_wood_creaks": "a creaking door",
    "door_wood_knock": "knocking on a door", "drinking_sipping": "someone drinking", "engine": "an engine running",
    "fireworks": "fireworks", "footsteps": "footsteps", "frog": "a frog croaking", "glass_breaking": "glass breaking",
    "hand_saw": "a hand saw", "helicopter": "a helicopter", "hen": "a hen clucking", "insects": "insects buzzing",
    "keyboard_typing": "typing on a keyboard", "laughing": "someone laughing", "mouse_click": "a computer mouse clicking",
    "pig": "a pig oinking", "pouring_water": "water being poured", "rain": "rain", "rooster": "a rooster crowing",
    "sea_waves": "sea waves", "sheep": "a sheep bleating", "siren": "a siren", "sneezing": "someone sneezing",
    "snoring": "someone snoring", "thunderstorm": "thunder", "toilet_flush": "a toilet flushing", "train": "a train",
    "vacuum_cleaner": "a vacuum cleaner", "washing_machine": "a washing machine", "water_drops": "water dripping",
    "wind": "wind blowing",
}
SUFFIX = {0: "\nA. yes\nB. no\nAnswer with only A or B.", 1: "\nA. no\nB. yes\nAnswer with only A or B."}


def question(category):
    return f"Can you hear {PHRASES[category]}?"


def load_plan(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def metadata():
    with Path("data/esc50.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pools(plan):
    rows = metadata()
    source = plan["source"]
    dev_fold = [r for r in rows if int(r["fold"]) == source["development_fold"]]
    ids = sorted({r["src_file"] for r in dev_fold})
    random.Random(source["split_seed"]).shuffle(ids)
    development = set(ids[:len(ids) // 2])
    return {"train": [r for r in rows if int(r["fold"]) == source["train_fold"]],
            "development": [r for r in dev_fold if r["src_file"] in development],
            "confirmation": [r for r in dev_fold if r["src_file"] not in development]}


def event_window(path, settings):
    from baseline import waveform
    from count_evidence import fade
    x = waveform(path)
    n = round(settings["event_seconds"] * RATE)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    hop = RATE // 20
    starts = range(0, len(x) - n + 1, hop)
    energy = [float(np.mean(x[i:i+n] ** 2)) for i in starts]
    i = list(starts)[int(np.argmax(energy))]
    sound = fade(x[i:i+n])
    rms = float(np.sqrt(np.mean(sound ** 2)))
    if rms <= 0:
        raise ValueError(f"Silent source {path}")
    sound = sound / rms * settings["event_rms"]
    return sound * min(1.0, settings["peak_limit"] / float(np.abs(sound).max()))


def prepare(args):
    import soundfile as sf
    from prepare import ESC, download
    plan, digest = load_plan(args.plan)
    settings = plan["stimuli"]
    groups = [set(g) for g in plan["questions"]["confusion_groups"]]
    assert set(PHRASES) == {r["category"] for r in metadata()}
    source_dir = args.out / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    by_split = pools(plan)
    plan_splits = (("confirmation", "confirm"),) if args.confirmation else (("train", "train"), ("development", "dev"))
    if getattr(args, "train_only", False):
        plan_splits = (("train", "train"),)
    needed = [r["filename"] for split, _ in plan_splits for r in by_split[split]]
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda name: download(f"{ESC}/audio/{name}", source_dir / name), needed))
    # Confirmation mixtures follow phase2/b2-v2/plan.json: same rules, generation seed 20261125, 100 per N.
    rng = random.Random(20261125 if args.confirmation else settings["generation_seed"])
    total = round(settings["duration_seconds"] * RATE)
    slot = total // settings["slots"]
    event_n = round(settings["event_seconds"] * RATE)
    manifest, sources = {}, {}
    for split, prefix in plan_splits:
        by_category = defaultdict(list)
        for r in sorted(by_split[split], key=lambda r: r["filename"]):
            by_category[r["category"]].append(r)
        categories = sorted(by_category)
        for n in settings["event_counts"]:
            items = []
            for m in range(settings["mixtures_per_count"].get(split, 100)):
                present = rng.sample(categories, n)
                clips = [rng.choice(by_category[c]) for c in present]
                positions = sorted(rng.sample(range(settings["slots"]), n))
                recording = np.zeros(total, dtype=np.float32)
                events = []
                for c, clip, p in zip(present, clips, positions):
                    start = p * slot + rng.randrange(slot - event_n + 1)
                    key = clip["filename"]
                    if key not in sources:
                        sources[key] = event_window(source_dir / key, settings)
                    recording[start:start+event_n] += sources[key]
                    events.append({"category": c, "filename": key, "src_file": clip["src_file"], "start": start / RATE})
                peak = float(np.abs(recording).max())
                gain = min(1.0, settings["peak_limit"] / peak)
                path = args.out / f"{prefix}-n{n}-{m:03d}.wav"
                sf.write(path, recording * gain, RATE, subtype="FLOAT")
                blocked = set().union(*(g for g in groups if g & set(present)))
                absent = [c for c in categories if c not in present and c not in blocked]
                k = min(n, 3) if split == "train" else 1
                yes = rng.sample(present, k)
                no = rng.sample(absent, k)
                asked = [(c, 0) for c in yes] + [(c, 1) for c in no]
                for j, (c, label) in enumerate(asked):
                    donor = asked[(j + k) % len(asked)][0]
                    items.append({"path": str(path), "question": question(c), "donor": question(donor), "label": label,
                                  "category": c, "n": n, "mixture": m, "gain": gain, "events": events})
            manifest[f"{prefix}_N{n}"] = items
            print(split, n, len(items), "questions", flush=True)
    manifest["plan_sha256"] = digest
    if args.confirmation:
        manifest["val"] = [x for n in settings["event_counts"] for x in manifest[f"confirm_N{n}"]]
    else:
        manifest["train"] = [x for n in settings["event_counts"] for x in manifest[f"train_N{n}"]]
        manifest["val"] = [x for n in settings["event_counts"] for x in manifest.get(f"dev_N{n}", ())]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    used = {s: sorted({r["src_file"] for r in by_split[s]}) for s in by_split}
    assert not set(used["development"]) & set(used["confirmation"])
    (args.out / "splits.json").write_text(json.dumps(used, indent=1), encoding="utf-8")


class Backbone:
    """Frozen Qwen2-Audio pieces used by bottleneck.py: projected audio features and decoding from embeddings."""

    def __init__(self, feature_cache):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        from baseline import MODEL_DIR
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL_DIR)
        self.base = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, device_map={"": 0}).eval().requires_grad_(False)
        self.lm = self.base.language_model
        self.embed = self.lm.get_input_embeddings()
        self.ids = [self.proc.tokenizer.encode(x, add_special_tokens=False)[0] for x in "AB"]
        self.cache_dir = feature_cache

    def features(self, path):
        from baseline import waveform
        torch = self.torch
        cached = self.cache_dir / (hashlib.sha1(path.encode()).hexdigest() + ".pt") if self.cache_dir else None
        if cached and cached.exists():
            return torch.load(cached, weights_only=True).cuda().float()
        prompt = self.proc.apply_chat_template([{"role": "user", "content": [{"type": "audio", "audio_url": path}]}],
                                               tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=prompt, audio=[waveform(path)], sampling_rate=16000, return_tensors="pt").to("cuda")
        inputs["input_features"] = inputs["input_features"].to(self.base.dtype)
        mask = inputs["input_ids"][0] == self.base.config.audio_token_index
        box = {}

        class Captured(Exception):
            pass

        def capture(module, positional, kwargs):
            box["x"] = kwargs["inputs_embeds"][0, mask].detach().cpu()
            raise Captured
        hook = self.lm.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            self.base(**inputs, use_cache=False)
        except Captured:
            pass
        finally:
            hook.remove()
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            torch.save(box["x"], cached.with_suffix(".part"))
            cached.with_suffix(".part").replace(cached)
        return box["x"].cuda().float()

    def decode(self, latent, question):
        content = [{"type": "audio", "audio_url": "placeholder"}] if latent is not None else []
        template = self.proc.apply_chat_template([{"role": "user", "content": content + [{"type": "text", "text": question}]}],
                                                 tokenize=False, add_generation_prompt=True)
        tokens = self.proc.tokenizer(template, return_tensors="pt").input_ids.cuda()
        if latent is None:
            return self.lm(input_ids=tokens, use_cache=False).logits[0, -1, self.ids].float()
        index = (tokens[0] == self.base.config.audio_token_index).nonzero().item()
        text = self.embed(tokens)
        sequence = self.torch.cat((text[:, :index], latent[None].to(text.dtype), text[:, index+1:]), dim=1)
        return self.lm(inputs_embeds=sequence, use_cache=False).logits[0, -1, self.ids].float()


def control(args):
    """Score full audio, text only and saved audio-only compressors on the development mixtures, both orders."""
    import torch
    from bottleneck import Compressor
    plan, digest = load_plan(args.plan)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["plan_sha256"] != digest:
        raise ValueError("Stimuli were generated from a different plan")
    items = manifest["val"]
    model = Backbone(args.feature_cache)
    width = model.embed.embedding_dim
    configs = {"full": (0, None), "text": (0, None)}
    for k in (1, 4, 16):
        for seed in (17, 18, 19):
            path = Path(f"runs/guidance/k{k}-s{seed}-audio/compressor.pt")
            if path.exists():
                configs[f"K{k}-s{seed}"] = (k, path)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    partial = args.scores.with_suffix(".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle, torch.no_grad():
        writer = csv.writer(handle)
        writer.writerow(["model", "n", "mixture", "category", "label", "order", "yes_minus_no", "checkpoint_sha256"])
        for name, (k, checkpoint) in configs.items():
            comp, sha = None, ""
            if checkpoint:
                comp = Compressor(width, k).cuda().eval()
                comp.load_state_dict(torch.load(checkpoint, weights_only=True))
                sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            latents = {}
            for x in items:
                if name != "text" and x["path"] not in latents:
                    audio = model.features(x["path"])
                    latents[x["path"]] = comp(audio) if comp is not None else audio
                for order, suffix in SUFFIX.items():
                    logits = model.decode(latents.get(x["path"]), x["question"] + suffix)
                    margin = float(logits[0] - logits[1]) * (1 if order == 0 else -1)
                    writer.writerow([name, x["n"], x["mixture"], x["category"], x["label"], order, f"{margin:.5f}", sha])
            handle.flush()
            print("scored", name, flush=True)
    partial.replace(args.scores)
    report(plan, args.scores)


def report(plan, scores_path):
    rows = list(csv.DictReader(Path(scores_path).open(encoding="utf-8")))
    hits = defaultdict(lambda: defaultdict(list))  # (model, n) -> order -> [(mixture, hit, label)]
    for r in rows:
        yes = float(r["yes_minus_no"]) > 0
        hit = yes == (r["label"] == "0")
        hits[r["model"], int(r["n"])][int(r["order"])].append((int(r["mixture"]), hit, r["label"]))
    models = list(dict.fromkeys(r["model"] for r in rows))
    counts = plan["stimuli"]["event_counts"]
    summary = {}
    print(f"{'model':<10}" + "".join(f"  N={n}: A=yes  B=yes  yes-rec no-rec" for n in counts))
    for model in models:
        line = f"{model:<10}"
        for n in counts:
            orders = hits[model, n]
            acc = [np.mean([h for _, h, _ in orders[o]]) for o in (0, 1)]
            pooled = [x for o in (0, 1) for x in orders[o]]
            rec = [np.mean([h for _, h, l in pooled if l == lab]) for lab in ("0", "1")]
            summary.setdefault(model, {})[n] = {"accuracy_by_order": acc, "yes_recall": rec[0], "no_recall": rec[1]}
            line += f"         {100*acc[0]:5.1f}  {100*acc[1]:5.1f}  {100*rec[0]:5.1f}  {100*rec[1]:5.1f}"
        print(line)
    # Threshold-free view (exploratory, added after the accuracy gate failed): within each recording, does the
    # question about a present category score higher than the one about an absent category? Mean of both orders.
    margins = defaultdict(list)
    for r in rows:
        margins[r["model"], int(r["n"]), int(r["mixture"]), r["label"]].append(float(r["yes_minus_no"]))
    ranking = {}
    for model in models:
        for n in counts:
            mixtures = sorted({key[2] for key in margins if key[:2] == (model, n)})
            ranking.setdefault(model, {})[n] = [float(np.mean(margins[model, n, x, "0"]) > np.mean(margins[model, n, x, "1"]))
                                                for x in mixtures]
    families = defaultdict(list)
    for model in models:
        families[model.split("-s")[0]].append(model)
    print("\nwithin-recording ranking, mean of orders and seeds; difference from full audio with recording bootstrap 95% CI")
    rng = np.random.default_rng(17)
    for family, members in families.items():
        line = f"{family:<6}"
        for n in counts:
            values = np.mean([ranking[m][n] for m in members], axis=0)
            line += f"  N={n}: {100*values.mean():5.1f}"
            summary.setdefault("ranking", {}).setdefault(family, {})[n] = float(values.mean())
            if family not in ("full", "text"):
                gap = np.asarray(ranking["full"][n]) - values
                draws = [rng.choice(gap, len(gap)).mean() for _ in range(10000)]
                low, high = np.percentile(draws, [2.5, 97.5])
                line += f" ({100*gap.mean():+5.1f} [{100*low:+5.1f}, {100*high:+5.1f}])"
                summary.setdefault("ranking_gap_to_full", {}).setdefault(family, {})[n] = [float(gap.mean()), low, high]
        print(line)
    gates = plan["evaluation"]["gates"]
    full_ok = all(min(summary["full"][n]["accuracy_by_order"]) >= gates["full_audio_accuracy_minimum_each_count_each_order"]
                  for n in counts)
    k1 = [m for m in models if m.startswith("K1-")]
    drop = None
    if k1:
        mean = lambda n: np.mean([np.mean(summary[m][n]["accuracy_by_order"]) for m in k1])
        drop = float(mean(counts[0]) - mean(counts[-1]))
    print("full-audio gate:", full_ok, "| K=1 preview drop N=1 -> N=6:", None if drop is None else f"{100*drop:.1f} pts")
    out = Path(scores_path).with_name("summary.json")
    ranking_summary = {key: summary.pop(key) for key in ("ranking", "ranking_gap_to_full")}
    out.write_text(json.dumps({"models": summary, "full_audio_gate": full_ok, "k1_preview_drop": drop,
                               "exploratory_ranking": ranking_summary}, indent=1, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["prepare", "control", "report"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/b2mix"))
    parser.add_argument("--confirmation", action="store_true", help="Generate the held-out fold 2 confirmation half")
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/b2mix/manifest.json"))
    parser.add_argument("--feature-cache", type=Path)
    parsed = parser.parse_args()
    if parsed.action == "prepare":
        prepare(parsed)
    elif parsed.action == "control":
        control(parsed)
    else:
        report(load_plan(parsed.plan)[0], parsed.scores)
