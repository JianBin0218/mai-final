# Audio bottleneck baseline

Run from `/tmp2/b12902071/m_final` on cml30. The local checkout contains code and small result files. The remote `.venv`, audio and checkpoint stay on the SSD.

## Reproduce on the prepared server

Choose one available GPU after checking `nvidia-smi` and the lab limits. Substitute its index for N. Use a persistent SSH terminal and run the commands inside tmux.

```bash
cd /tmp2/b12902071/m_final
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python prepare.py
RESULT_DIR=results_repeat .venv/bin/python baseline.py
.venv/bin/python prepare.py --all-order
PAIR_FILE=data/all_order.json RESULT_DIR=results_order_repeat .venv/bin/python baseline.py
```

The model directory holds Qwen/Qwen2-Audio-7B-Instruct revision 0a095220c30b7b31434169c3086508ef3ea5bf0a. The dataset sources and revisions are in prepare.py. The ESC-50 pilot draws one fold-5 clip per category and pairs different categories. The CompA-Order pilot samples original/reverse pairs from the gijs mirror. The expanded evaluation uses all matching pairs in that mirror. The two order evaluations overlap.

The two clips in each pair share the exact prompt and options. The score is the difference between the logits for B and A at the first output token. No generation or training is required. Swapping the two recordings permutes the two measured predictions exactly, so the script reuses those predictions. Swapped accuracy is scored against the original labels. Text-only accuracy is 50% by construction. The pair-level score counts pairs with both recordings identified correctly.

## Phase 2 bottleneck smoke test

On cml30, use the same project directory, environment and single-GPU selection as above. Training and evaluation are separate entry points. Use a fresh output directory when rerunning training.

```bash
.venv/bin/python bottleneck.py train --k 16 --seed 17 --steps 16 --out smoke16_repeat
.venv/bin/python bottleneck.py eval --k 16 --seed 17 --out smoke16_repeat
```

This is a parallel resampler over projected audio features, with a frozen Qwen decoder. The decoder rebuilds its text embeddings and receives only the slots at the audio placeholder, with no first-stage cache. Four fixed binary tasks use eight training recordings from fold 1 and eight held-out recordings from fold 5. Per-example predictions, losses and memory measurements are in the JSON files. Keep the checkpoint on the server. This smoke test uses audio-only compression and measures engineering feasibility.

## Matched Clotho-AQA pilot

Prepare the audio archive and the question manifest on cml30 from the project directory:

```bash
.venv/bin/python prepare_clotho.py
```

The script verifies the published archive and annotation checksums, uses the official training and validation splits, and chooses one unanimously positive and one unanimously negative question per recording. It selects 48 training and 32 validation recordings with seed 17. The test split is unused.

From the local checkout, with the code synced to cml30, choose a free GPU with enough memory and run both arms sequentially. Replace N with its physical index, and use a new tag for each repetition. The local runner reads WANDB_API_KEY from the ignored .env file and sends only metrics and configuration to W&B. Audio and checkpoints remain on cml30. Keep the local runner connected throughout the pilot.

```powershell
uv run --with wandb python -u track_comparison.py --gpu N --tag clotho-k16-repeat --steps 192
python compare_pilot.py clotho-k16-repeat
```

To evaluate a saved conditioned checkpoint directly on cml30, select the GPU as above and run:

```bash
.venv/bin/python bottleneck.py eval --clotho --conditioned --out clotho-k16-repeat-question
```

Both arms decode the same question and A/B options. Only the conditioned compressor sees the question stem. Wrong-question evaluation uses the other stem for the same recording and keeps the original decoder input. Confidence intervals in compare_pilot.py resample whole recordings.
