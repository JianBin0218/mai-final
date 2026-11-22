"""Run the matched pilot remotely, and publish scalar metrics using local credentials."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--steps", default=192, type=int)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag.isascii() or not args.tag.replace("-", "").isalnum():
        parser.error("Use an alphanumeric tag, optionally with hyphens")
    lines = Path(".env").read_text(encoding="utf-8-sig").splitlines()
    key = next(line.split("=",1)[1].strip().strip(chr(34)+chr(39)) for line in lines
               if line.strip().startswith("WANDB_API_KEY="))
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_DISABLE_GIT"] = "true"
    for name in ("audio","question"):
        directory = args.tag+"-"+name
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            with wandb.init(project="mai-audio-bottleneck", group=args.tag, name=directory, dir=temp,
                            config=dict(k=16,seed=17,steps=args.steps,condition=name,train_clips=48,val_clips=32),
                            settings=wandb.Settings(console="off",save_code=False)) as run:
                print("RUN_URL",run.url,flush=True)
                command = (f"cd /tmp2/b12902071/m_final && export CUDA_DEVICE_ORDER=PCI_BUS_ID "
                           f"CUDA_VISIBLE_DEVICES={args.gpu} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 && "
                           f".venv/bin/python -u bottleneck.py train --clotho --out {directory} --steps {args.steps} "
                           + ("--conditioned " if name=="question" else "") +
                           f"&& cat {directory}/train.json")
                process = subprocess.Popen(["ssh","-o","BatchMode=yes","cml30",command],stdout=subprocess.PIPE,
                                           text=True,encoding="utf-8")
                remaining = []
                for line in process.stdout:
                    if line.startswith("METRIC "):
                        metrics = json.loads(line[7:])
                        step = metrics.pop("step")
                        run.log({"train/"+k:v for k,v in metrics.items()},step=step)
                    else:
                        remaining.append(line)
                if process.wait():
                    raise RuntimeError("Remote training failed")
                raw = "".join(remaining)
                result=json.loads(raw[raw.index("{"):])
                result["wandb_url"]=run.url
                out=Path("phase2")/directory
                out.mkdir(parents=True,exist_ok=True)
                (out/"train.json").write_text(json.dumps(result,indent=2))
                for part in ("train","heldout","wrong_question","full_audio","text_only"):
                    if part in result:
                        run.summary[part+"/accuracy"]=result[part]["correct"]/result[part]["n"]
                        run.summary[part+"/loss"]=result[part]["loss"]


if __name__ == "__main__":
    main()
