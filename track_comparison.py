"""Run matched audio-only / question-conditioned trials on a GPU host or on this machine.

Defaults reproduce the original cml30 setup. Override the host, project directory and
interpreter with flags or with MAI_HOST, MAI_PROJECT_DIR and MAI_PYTHON. Use --host local
to run in this checkout without SSH. Enable server-side W&B explicitly for monitored runs.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
DEFAULT_REMOTE_DIR = "/tmp2/b12902071/m_final"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, type=int, help="Physical GPU index (PCI bus order)")
    parser.add_argument("--steps", default=192, type=int)
    parser.add_argument("--k", default=16, type=int)
    parser.add_argument("--seed", default=17, type=int)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--host", default=os.environ.get("MAI_HOST", "cml30"),
                        help="SSH host alias, or 'local' to run here (env MAI_HOST, default cml30)")
    parser.add_argument("--project-dir", default=os.environ.get("MAI_PROJECT_DIR"),
                        help=f"Project checkout on the host (env MAI_PROJECT_DIR; default {DEFAULT_REMOTE_DIR} "
                             "over SSH, this checkout with --host local)")
    parser.add_argument("--python", default=os.environ.get("MAI_PYTHON", ".venv/bin/python"),
                        help="Interpreter on the host, relative to the project directory (env MAI_PYTHON)")
    parser.add_argument("--out-root", default=".",
                        help="Directory, relative to the project, for checkpoints and logs")
    parser.add_argument("--init-from",
                        help="Warm-start both arms from this audio-only checkpoint (path on the host)")
    args = parser.parse_args()
    if not args.tag.isascii() or not args.tag.replace("-", "").isalnum():
        parser.error("Use an alphanumeric tag, optionally with hyphens")
    local = args.host == "local"
    project = args.project_dir or (str(ROOT) if local else DEFAULT_REMOTE_DIR)
    key = ""
    if args.wandb:
        lines = (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines()
        key = next(line.split("=", 1)[1].strip().strip(chr(34)+chr(39))
                   for line in lines if line.strip().startswith("WANDB_API_KEY="))
        if not key:
            parser.error("WANDB_API_KEY is empty")
    for arm in ("audio", "question"):
        name = args.tag + "-" + arm
        out = ROOT / "phase2" / name
        out.mkdir(parents=True, exist_ok=True)
        run = str(PurePosixPath(args.out_root) / name)
        remote = [f"cd {shlex.quote(project)}",
                  f"mkdir -p {shlex.quote(args.out_root)}",
                  f"export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={args.gpu} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4"]
        command = [args.python, "-u", "bottleneck.py", "train", "--clotho", "--out", run,
                   "--steps", str(args.steps), "--k", str(args.k), "--seed", str(args.seed)]
        if args.init_from:
            command += ["--init-from", args.init_from]
        if arm == "question":
            command.append("--conditioned")
        if args.wandb:
            remote += ["IFS= read -r WANDB_API_KEY", "export WANDB_API_KEY WANDB_DISABLE_GIT=true"]
            command.append("--wandb")
        remote += [f"set -o pipefail && {shlex.join(command)} | tee {shlex.quote(run + '.log')}",
                   f"echo BEGIN_RESULT && cat {shlex.quote(run + '/train.json')}"]
        script = " && ".join(remote)
        argv = ["bash", "-c", script] if local else ["ssh", "-T", "-o", "BatchMode=yes", args.host,
                                                     "bash -c " + shlex.quote(script)]
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
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
            raise RuntimeError(f"{arm} arm failed on {args.host}")
        (out/"train.json").write_text(json.dumps(json.loads("".join(result)), indent=2))


if __name__ == "__main__":
    main()
