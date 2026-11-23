# 專題總覽網頁｜需求與維護說明

`docs/index.html` 是給組員、助教和之後加入的人看的專題總覽。網頁之後由健賓主控更新。這份文件記錄當初的需求、設計決定，以及各部分資料放在哪裡，方便手動更新，或交給 Claude／Codex 改。

## 一、這個網頁要做到什麼

原始需求（cshsieh 提出）：

1. 讓人一眼看懂專題的**架構與動機**：如果回答模型只能透過 K 個 latent 取得聲音，壓縮器要保留什麼？
2. 用**視覺化的架構圖**清楚呈現各線的分工，配上流暢的動畫，看得出每條線目前做到哪裡。
3. **預留延伸空間**。這是進行中的專題，之後討論出的新方向，要能整理成新的 track、實作計劃和 todo list。
4. 是一個**誰都能打開的網頁**：單一 HTML 檔，下載後用瀏覽器打開即可，不需要安裝或建置。
5. 風格要**漂亮、簡潔、乾淨、文青、高雅**，像一本安靜的研究手冊，而不是儀表板。

## 二、視覺風格

**整體。** 紙張色背景、細線代替方框、大量留白。文字用襯線體，介面標籤才用無襯線體。不使用陰影、漸層、圓角卡片或 emoji。

**欄寬。** 內文閱讀欄約 680px，圖表欄約 1040px。手機寬度時左右各留 16px，不能出現水平捲動。

**字體**（Google Fonts，是頁面唯一的外部資源）：

| 用途 | 字體 |
|---|---|
| 英文標題、數字 | Cormorant Garamond |
| 中文標題與內文 | Noto Serif TC |
| 介面標籤、說明 | Noto Sans TC |
| 程式碼、公式 | IBM Plex Mono |

**顏色。** 全部定義在 `:root` 的 CSS 變數。深色模式有自己的一組值（`prefers-color-scheme: dark` 與 `[data-theme="dark"]`），不是自動反轉。

| 變數 | 淺色 | 深色 | 用途 |
|---|---|---|---|
| `--paper` | `#f3f3ef` | `#141615` | 背景 |
| `--ink` | `#1c1e1d` | `#e8e7e1` | 主要文字 |
| `--a` | `#2f5379` 靛藍 | `#8eabd0` | A 線：壓縮容量 |
| `--b` | `#a5672a` 琥珀 | `#d49a5d` | B 線：題目導引 |
| `--c` | `#5b7c50` 苔綠 | `#98b68a` | C 線，以及共同工作 |

每條線固定一個顏色，換頁面、換圖表都不改。新增一條線時，在 CSS 加一條 `.track[data-code="X"] { --tc: var(--x); }`，並在 `:root` 和兩個深色區塊各定義一次 `--x`。

**動畫。** 緩慢、克制，主要用 `cubic-bezier(.2,.7,.2,1)`，時間在 0.3–1.6 秒之間。所有動畫都要尊重 `prefers-reduced-motion`：使用者關閉動畫時，直接顯示最終狀態。

## 三、頁面結構與資料位置

| 區塊 | 內容 | 資料在哪裡改 |
|---|---|---|
| 開頭（hero） | 標題、一段說明、各線負責人；canvas 動畫：聲音頻譜點流進 K 個 slot，再交給解碼器；底下是容量刻度條 | `.sub` 段落、`.byline`；canvas 在 `#scope-canvas` 的腳本 |
| 一 動機與架構 | 固定的部分與各線改動的部分（`.premise` 欄位）；可點選的架構節點圖；完整、只有音訊、讀題目、只看文字四種模式切換 | `.premise`；節點說明在 `NODES` |
| 二 分工與進度 | 共同起點的步驟，以及每條線一欄的里程碑時間線與進度條 | `SHARED`、`TRACKS` 陣列 |
| 三 主要結果 | 分頁圖表：A 線題型點圖、B 線導引折線、差異森林圖、劑量反應；每張圖都有 hover 提示與「表格」切換 | `A_TYPES`、`B_ARMS`、`B_EXTRA`、`C_ROWS`、`D_X`、`D_SER`，以及每頁的 `.take` 文字 |
| 四 研究規劃 | 看板：想法、規劃中、進行中、完成；每張卡片是一個計劃和它的 todo | 檔案頂端的 `roadmap-seed` |
| 五 常見疑問 | 名詞解釋，例如 latent、為什麼推論時看得到題目 | 直接改 HTML |
| 頁尾 | 資料來源與更新日期 | `.colophon` |

**`TRACKS` 里程碑格式。** `{ s, t, r, f }`：

- `s`：狀態，`"done"`、`"active"`、`"paused"` 或 `"next"`。
- `t`：標題。
- `r`：一句結果或目的。
- `f`：可展開的細節。

**`roadmap-seed` 格式。**

- `tracks`：`{ code, name, owner, question, slot, order }`。
- `plans`：`{ id, track, status, title, why, todos }`。
  - `status` 是 `idea`、`planned`、`active` 或 `done`。
  - `todos` 是 `[文字, 是否完成]` 的陣列。
- 直接用瀏覽器打開檔案時，看板是唯讀的，內容來自這份 seed。

## 四、內容規則

- **數字和結論都要出自 repo。** 主要來源是 `PROJECT_HANDOFF.md`、`README.md` 和 `phase2/` 底下的結果檔。不要自己做新的分析。
- **保留原本的限定語。** handoff 說「未觀察到」，網頁就不能寫成「沒有」；不要把信賴區間四捨五入成更強的說法。探索性結果標成探索性。
- **用繁體中文**，短句、白話，數字帶單位（題數、百分點、錄音段數）。
- **里程碑**只有在 handoff 報了結果後才標成 `done`；計劃用 `next`，中止的實驗用 `paused`。
- **不改設計。** 只有在加新的一條線或新的結果分頁時，照現有樣式新增。
- **JavaScript 要保持有效**：括號成對、字串引號正確。改完用瀏覽器打開檢查一次，包括手機寬度和深色模式。

## 五、怎麼更新

**手動。** 直接改 `docs/index.html`，用瀏覽器打開檢查，再開 PR。

**自動（選用）。** 網頁可直接手動更新。啟用 `.github/workflows/update-site.yml` 後，相關研究檔案推進 `main` 時，也可以自動產生更新草稿。它以頁首 `site-source` 記錄研究來源提交，只改網頁本身不會觸發。使用前由管理員設定：

1. 在 GitHub Actions Secrets 設定 `CLAUDE_CODE_OAUTH_TOKEN`。憑證持有人可以直接設定 secret。
2. Settings → Actions → General，勾選「Allow GitHub Actions to create and approve pull requests」。
3. 在 GitHub Actions Variables 設定 `MAI_SITE_UPDATES_ENABLED=true`，先從 `main` 手動執行一次，確認草稿內容。

產生頁面的工作使用唯讀的 GitHub token。另一個工作只接收這個 HTML 檔，確認 `main` 仍指向同一個來源版本後開草稿 PR。組員審核數字和頁面後再合併。停用時將 `MAI_SITE_UPDATES_ENABLED` 改為 `false`。

**線上版。** cshsieh 另外在 claude.ai 發布過一份可以共同編輯看板的副本。它不會跟著 repo 更新，以 repo 裡的這份為準。

## 六、目前待更新的內容

網頁已整理 PR #1 的 B 線驗證結果和 `aad71db` 的 A 線官方 test。後續仍可補充：

1. **C 線**的讀取假說和實驗設計在 `codex/track-c-handoff` 分支。合併後可加到分工與規劃。
2. **B 線**的 B2 計劃與結果在 `track-b-next` 分支。合併後更新 B 欄里程碑與看板。
