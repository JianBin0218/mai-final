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
