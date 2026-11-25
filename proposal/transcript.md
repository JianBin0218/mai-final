# 提案報告逐字稿（約 4 分 35 秒）

對應 `proposal/slides.typ`（11 頁，圖檔在 `proposal/figures/`）。括號內是每頁的時間預算，以每分鐘約 230 字估算。

## 1. 標題頁（0:10）

大家好，我們是第 29 組。我們的題目是：Do Audio Language Models Use Their Own Latents？也就是，音訊語言模型到底有沒有用到它自己產生的 latent。

## 2. Outline（0:05）

報告分成四個部分：動機、相關研究、方法與初步結果，以及下一步。

## Introduction

### 3. Motivation（0:30）

Latent reasoning 是讓模型在回答之前，先產生一段連續向量來思考，而不是寫出文字的推理過程。CoAT 把這個方法用在 Qwen2-Audio 上，MMAU 準確率從 52.5% 提升到 66.9%。

但是模型回答時，其實還聽得到原始音訊。所以這個提升可能來自 latent，也可能只是微調之後，模型直接聽音訊答得更好。單看準確率，分不出這兩種情況。

### 4. Research Question and Contributions（0:25）

所以我們的研究問題是：在聽得到原始音訊時，音訊語言模型會不會用自己的 latent 來回答？

我們有三點貢獻：針對音訊的因果測試，只改變 latent；用對照組區分微調和 latent 的效果；比較回答時有完整、部分或沒有音訊的設定。

初步結果顯示，音訊也有影像研究中提過的 latent bypass。

## Related Work

### 5. Latent Reasoning in Vision（0:25）

在影像領域，已經有好幾篇研究直接對 latent 做介入，例如歸零、加雜訊，或換成別的 latent，結果準確率幾乎不變，代表 latent 被繞過了。

也有研究提出修正方法：一種是在回答時拿掉影像，讓 latent 變成必要；另一種是在推論時直接最佳化 latent。

### 6. Latent Reasoning in Audio（0:20）

音訊這邊，CoAT、JELAR、DeALR 都提出了 latent 方法，SPARE 也發現推理變長時，注意力會偏離音訊。但這些研究都沒有固定訓練好的模型、只對 latent 做介入。這就是我們要補上的缺口。

## Methodology and Experiments

### 7. Two-Pass Write–Read Architecture（0:35）

我們使用 Qwen2-Audio-7B-Instruct，兩個階段共用一組 LoRA。

第一階段是寫入：模型讀音訊、題目和選項，取 4 個可學習位置的最後一層 hidden state，當作 4 個 latent。第二階段是回答：模型根據題目、選項和這 4 個 latent，輸出各選項字母的分數。

三種回答端只差在有沒有直接音訊：Soft 有音訊，和 CoAT 一樣，是主要測試；Hard 沒有音訊，只能靠 latent；Mid 在訓練時有一半機率拿掉音訊。

### 8. Interventions, Controls, and Data（0:25）

訓練完之後，我們固定模型和輸入，只替換 latent：換成別題的、資料集平均、歸零，或加雜訊，再看有多少答案改變、準確率變多少。

對照組包括只做 LoRA、K 等於 0、加上重建損失，以及不看題目的寫入端。資料方面，用 AVQA 訓練，AVQA 和 MMAU 做開發，MMAR 留到最後才評估。

### 9. Pilot Results: Latent Bypass with Direct Audio（0:35）

這是 Soft 的初步結果。原始模型在 MMAU 是 65.1%，只做 LoRA 是 66.8%，K 等於 0 是 66.0%，而有 4 個 latent 的 Soft 也是 66.0%。有沒有 latent，分數都差不多，所以提升來自微調，不是 latent。

四種介入只改變了 0.1% 到 0.9% 的答案。就算加上重建損失、確定 latent 裡有音訊資訊，也只改變 0.7% 到 1.2%。也就是說，模型直接聽音訊，繞過了 latent，和影像研究看到的現象一樣。

### 10. Pilot Results: Reader Conditions（0:30）

那如果拿掉音訊呢？我們同樣把 latent 換成別題的。Soft 只有 0.2% 的答案改變；Mid 在有音訊時改變 19.7%，沒有音訊時改變 67.5%；Hard 準確率 66.5%，換掉 latent 後掉了 40 分，74% 的答案都變了。

所以 latent 有能力攜帶資訊，但模型只有在聽不到音訊時才會用它。不過，寫入端看得到題目和選項，latent 裡可能只是答案，不一定是聲音證據。

## Next Steps

### 11. Proposed Experiment: Paired Evidence Swap（0:35）

所以下一步，我們用配對錄音來區分這兩種情況。兩段錄音只差在有沒有目標聲音，例如狗叫。回答端聽有狗叫的 Clip A，latent 則改用沒有狗叫的 Clip B 來寫。

如果 Yes 減 No 的分數往 Clip B 的答案移動，代表 latent 帶的是聲音證據。我們也會用不看題目的寫入端，並讓兩端使用不同的選項順序，排除 latent 只是在傳答案的可能。

以上是我們的提案，謝謝大家。

---

## 備用問答（不在報告時間內）

**為什麼原始 Qwen2-Audio 在我們這裡是 65.1%，CoAT 報告的是 52.5%？**
評分方式不同。我們直接比較選項字母的分數，並平均正反兩種選項順序，不是讓模型生成文字再解析答案。

**這些結果可靠嗎？**
目前是 pilot：單一 seed、K = 4、開發集上的結果。之後會用多個 seed 重跑，並且只在保留的 MMAR 上做一次最終評估。

**不看題目的寫入端現在可以用了嗎？**
還不行。Pilot 中 blind Hard 只有 25.1%，接近亂猜，所以要先把它訓練好，才能跑配對交換實驗。
