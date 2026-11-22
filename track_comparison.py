"""Run matched trials on cml30. Enable server-side W&B explicitly for monitored runs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--steps", default=192, type=int)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    if not args.tag.isascii() or not args.tag.replace("-", "").isalnum():
        parser.error("Use an alphanumeric tag, optionally with hyphens")
    key = ""
    if args.wandb:
        lines = Path(".env").read_text(encoding="utf-8-sig").splitlines()
        key = next(line.split("=", 1)[1].strip().strip(chr(34)+chr(39))
                   for line in lines if line.strip().startswith("WANDB_API_KEY="))
        if not key:
            parser.error("WANDB_API_KEY is empty")
    for arm in ("audio", "question"):
        name = args.tag + "-" + arm
        out = Path("phase2") / name
        out.mkdir(parents=True, exist_ok=True)
        remote = ["cd /tmp2/b12902071/m_final",
                  f"export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={args.gpu} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4"]
        command = f".venv/bin/python -u bottleneck.py train --clotho --out {name} --steps {args.steps}"
        if arm == "question":
            command += " --conditioned"
        if args.wandb:
            remote += ["IFS= read -r WANDB_API_KEY", "export WANDB_API_KEY WANDB_DISABLE_GIT=true"]
            command += " --wandb"
        remote += [f"set -o pipefail; {command} | tee {name}.log", f"echo BEGIN_RESULT && cat {name}/train.json"]
        proc = subprocess.Popen(["ssh", "-T", "-o", "BatchMode=yes", "cml30", "bash -c "+shlex.quote(" && ".join(remote).replace("set -o pipefail; ", "set -o pipefail && "))],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc.stdin.write((key+"\n").encode() if args.wandb else b"")
        proc.stdin.close()
        result = []
        collecting = False
        for raw in proc.stdout:
            line = raw.decode("utf-8")
            if collecting:
                result.append(line)
            elif line.strip() == "BEGIN_RESULT":
                collecting = True
            else:
                print(line, end="", flush=True)
        if proc.wait():
            raise RuntimeError("Remote training failed")
        (out/"train.json").write_text(json.dumps(json.loads("".join(result)), indent=2))


if __name__ == "__main__":
    main()
