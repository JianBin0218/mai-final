# Audio bottleneck baseline

[中文專題進度、暫定規劃與研究分工](PROJECT_HANDOFF.md)

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

From the local checkout, with the code synced to cml30, choose a free GPU with enough memory and run both arms sequentially. Replace N with its physical index, and use a new tag for each repetition. By default the runner stores per-step metrics in the remote log. Add --wandb for direct server-side tracking. The runner passes the local .env key over SSH standard input for the life of the remote process. Audio and checkpoints remain on cml30. Keep the local runner connected throughout the pilot.

```powershell
python -u track_comparison.py --gpu N --tag clotho-k16-repeat --steps 192
python compare_pilot.py clotho-k16-repeat
```

To evaluate a saved conditioned checkpoint directly on cml30, select the GPU as above and run:

```bash
.venv/bin/python bottleneck.py eval --clotho --conditioned --out clotho-k16-repeat-question
```

Both arms decode the same question and A/B options. Only the conditioned compressor sees the question stem. Wrong-question evaluation uses the other stem for the same recording and keeps the original decoder input. Confidence intervals in compare_pilot.py resample whole recordings.


## 共用起點續訓（A、B 組）

在 cml30 專題目錄、選定一張 GPU 後執行。A、B 使用同一份 Clotho manifest、父 checkpoint、seed 17、K=16、192 次更新與學習率 0.001。兩組都重建 AdamW。題幹投影在兩組起始時均歸零；每次續訓前逐一對比 64 個驗證問題的 A/B logits，另在一個訓練樣本上檢查題幹投影梯度。此檢查不更新模型。

```bash
# A：已執行。重跑請換新的 --out。
.venv/bin/python bottleneck.py train --clotho --steps 192 --seed 17 --k 16 --init-from clotho-k16-s17-audio/compressor.pt --out warm-k16-s17-audio
# B：從相同的原始父 checkpoint 起跑。
.venv/bin/python bottleneck.py train --clotho --steps 192 --seed 17 --k 16 --init-from clotho-k16-s17-audio/compressor.pt --out warm-k16-s17-question --conditioned
```

使用 GitHub checkout 的 bottleneck.py；本輪 cml30 共享根目錄上的該版本另存為 bottleneck_a.py，讓其他使用者可繼續使用舊入口。兩組 train.json 內的 warm_start.sha256 應相同。將兩份結果各放入本機 phase2/ 下同名資料夾後，執行 python compare_pilot.py warm-k16-s17。

## A 組容量與固定平均池化對照

沿用 Clotho 小型驗證集。兩個學習式容量版本皆從頭訓練 192 步；這裡的 K=16 比較對象為 clotho-k16-s17-audio，而非額外續訓 192 步的 warm 版本。固定池化沿時間將 projected audio features 做 adaptive average pooling，再交給同一個凍結解碼器，沒有訓練參數。以下在 cml30 專案環境、選好單卡後執行。重跑時換新的輸出路徑。

```bash
.venv/bin/python bottleneck.py train --clotho --k 4 --seed 17 --steps 192 --out capacity-k4-s17-audio
.venv/bin/python bottleneck.py pool --clotho --pool-ks 4 16 --out capacity-fixed
```

在本機 repo 執行 `python compare_capacity.py` 可從已保存的逐題預測重建準確率、同錄音兩題皆對數、問題層級的勝負數，以及以錄音為單位的配對 bootstrap 區間。

### A 線：用多個訓練 seed 重測容量

在準備好的 cml30 專案目錄、選好一張 GPU 後執行 `bash run_capacity_seeds.sh 23 41`。每個 seed 依序跑 K=4 和 16，每次從頭更新 192 步。腳本沿用同一份資料 manifest，已完成且有結果與權重的設定會跳過。每次只啟動一個程序。若在分開的工作目錄使用共用資料，先確認模型、資料和環境路徑正確。

把各 run 的 train.json 放入 phase2/ 下同名資料夾後，本機執行 `python analyze_capacity_types.py`。如只分析特定 seed，可用 `--seeds 17`。程式以錄音、題幹及正解核對資料，輸出逐 seed 成績、yes/no 表現、依題型的淨增答對數及每個 seed 都翻轉的題目。

題型檔 phase2/capacity_question_types.csv 依題目文字預先固定，採單一主要標籤。time_pattern 涵蓋時序、規律與持續性；acoustic_property 涵蓋響度、音色等性質；count_or_relation 涵蓋數量、相對距離與背景位置；scene_context 涵蓋場所與情境推斷；source_or_event 涵蓋聲源及事件辨識。標籤描述提問內容，不能單獨判定模型實際依靠哪一項聲學線索。
