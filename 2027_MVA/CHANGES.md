# 論文變更記錄

每個論文快照(= release tag)一段,新的在上。每一條都掛標籤:

- **[方法]** — 模組本身改了;受影響的數字全部重測
- **[協議]** — 評測方式有錯或不乾淨;修正後重測
- **[筆誤]** — 稿子寫錯,程式一直是對的
- **[編輯]** — 只動呈現方式,數字不變

**文中的 A3、A25 這類編號是什麼**:實驗記錄編號。每個編號「做了什麼、結果是什麼」見**文末附表**;完整記錄(預登記文件、跑法、原始數據位置、統計結論)在 `RESEARCH_NOTES.md` §10 的同編號列,拿編號去搜即可(例如搜「A25」)。

---

## 未發布 — `gmc_v3.tex`(2026-09-17,審查小組 issue #40:數字/事實修正)

2026-09-13 模擬審查小組(`review_gmc_v3_panel_2026-09-13.md`,本地)發現的數字/事實問題,全部查證後修正。開工前重新查證 FlexHook camera-ready(arXiv 2503.07516 v5):**published V2 = 42.53(不是 repo 記錄的 42.81)**,V1 53.83;我們的複現(42.526 / 53.824)在其發表精度下吻合,DetA/AssA 也逐位吻合 — 小組的 P0 第 1 項(「exceeds published 對 V2 不成立」)因此不成立,論文原主張是對的,只需標註精度。42.81 是 landscape 記憶的舊值,已更正。

### [筆誤] — 稿子寫錯,數據一直是對的
- §3(架構總覽):iKUN 誤寫為「performs object detection and tracking」;改為 scores frozen off-the-shelf tracklets(與 §2.1 一致)。
- §3.4 LOSO 句:「(on the validation set)」與 §4.1 自相矛盾且流程寫反;改為正確協議(逐 fold 留一序列、其餘序列上選 α、全測試集用 fold 中位數)。§4.1 的「No weight is chosen on the sequence it is evaluated on」改為誠實版本,並引早期配置上的 out-of-fold composite 檢查(~88% moving-class 增益存活)。
- §5:「routes only moving-class」→「moving- and static-class」(與 Table 1 / `moving_kw.py` 一致);分母改官方名單(24/150,818 全集留括號)。

### [編輯] — 揭露補強,數字不變
- §3.4 腳註印出完整關鍵字分類器(19 MOVING + 7 STATIC 字根)+「先凍結、後選 α」。
- Table 2:全欄統一 2 位小數、+GMC 列加 ±(0.12/0.03/0.06/0.03)、去掉全 bold;caption 逐列標 published vs reproduced($^*$);iKUN 44.543 複現句加進 §4.1。
- §4.1:± = seed std 定義;19 seqs =「18 V1 + 1 條 V2-only」子句;V1-trained aligner 用於全部四設定(含 V2)句;host 分數尺度句(score std:iKUN 0.50 / TransRMOT 0.35 / FlexHook margin 10.2 — 本輪新量測,解釋 α=7)+ gate 凍結句(0.0;TransRMOT 0.5)。
- §4.3:Welch 補 df 與精確 p(A43 現成:ego 8×10⁻⁶ / 2×10⁻⁶;multiscale 均 1.6×10⁻³)、標明 seed-stability;n=3/n=5 對帳括號;routing +4.57 / ego +4.62 分解句。
- Table 4 caption:逐列 α 政策(變體繼承 (1.0, 0.1),未重選;single-α 列 = 0.35)。
- 版面:正文收在第 5 頁,參考文獻溢出一條到第 6 頁(#41 版面輪會再收)。

## paper-2026-08-31 — `gmc_v3.tex`(對 `paper-2026-08-30`)

### [編輯] — latexdiff 基準改為教授版(release 後修正)

### [編輯] — 消除 diff 噪音:v3 對齊教授版的 token 與來源順序(release 後修正 2)

- 使用者指出 diff 充滿無意義的大小寫/格式標記。將 `gmc_v3.tex` 與 `gmc_v2_1.tex` 逐 token 對齊:`$\alpha_c$` 寫法、`\emph`、`w. GMC`、權重不進 math、bullet 不加粗、恢復教授的「(relative)」「at $I_{t-g}$」「to describe」「(on the validation set)」措辭、Architecture 13cm、qualitative 改用教授的 `qualitative.png`(PR #39)、Table 3 行名 `\quad -`;**Table 1 原始碼移回教授的位置**(§4.2 段後)—— 這消除了 latexdiff 把整段 §4.1 標成整刪整加的假差異,代價是 Table 1 浮回第 4 頁。重生 `diff_gmc_v2_1_vs_gmc_v3.pdf`,現在只標真實改動(α 1.0、事實句、數字、表格)。三個 assets 再更新。

- PR #39(JKaiWang)把教授的原始 `gmc_v2_1.tex` 與圖源(`Architecture.pptx`、`qualitative.pptx`、更新的 `Architecture.png`、新增 `qualitative.png`)進 repo。原 release 附的 `diff_gmc_v3_0830_vs_0831.pdf`(基準 = 我們自己的 0830 版)撤下並自 repo 移除;改為 **`diff_gmc_v2_1_vs_gmc_v3.pdf`(基準 = 教授版 `gmc_v2_1.tex`)**,直接顯示我們在教授改稿之上動了什麼(α 1.0、事實句恢復、表格改動)。release assets 已更新。

### [編輯] — 合併教授改稿(`gmc_v1_by_WTC.pdf`,2026-08-29,基於 v1);Table 1 改列 DetA/AssA 數值;Table 3 改 Moving/Others/Pooled(A45)

- 合併原則:**措辭/結構照教授、數字與後續輪次的事實照 v3**(教授改的是 v1,沒看過 A42–A44)。來源只有 PDF(無 tex),逐頁重建。
- 照教授:摘要與 keywords 重寫(縮短;數字句移到 §1 貢獻點)、§2 重構為 2.1 Referring MOT / 2.2 Motion Compensation(刪 plane-plus-parallax 等四篇引用與 ORB,參考文獻 24→19)、§3 標題(Motion Compensation / Multi-Scale Residual Velocity / Motion-Language Matching)、§3.2 三種 velocity 的 bullet 定義、§3.4 例句式分類說明 + 「left vehicles which are parking」腳註 + 逐 host 權重白話句、§4 敘述(含各列對應資料集的說明句)、§4.3 簡化敘述、Fig.1 新圖說、§5 重寫(刪 future-work 段與「Third, two weights…」限制)、GMC Module→GMC module、stationary→static。
- 保留(教授未見的事實層):A42–A44 全部數字(α=(1.0,0.1)、45.304±0.115、+9.44、21/150、111/862、214/818)、官方 150 條名單句、44.543 重現揭露、7,690 擬合成功、149 FPS、Welch t 值、warm-up 段、1/g 註記、horizon 句(v2 新增,壓縮成一句)、框排除子句、moving-class 用語(作者規則)。
- 表格(作者 2026-08-31 指示):Table 1 改列 DetA/AssA 數值對(reproduced→w. GMC;w/o GMC / w. GMC 欄名照教授)、刪 Gain 欄(增益在 §4.2 文字);Table 3 欄改 **Moving / Others / Pooled**(Others = 非 moving 的 129 條,新算:native 45.87、full 46.05±0.05、single-α 45.99±0.12、−ego 45.77±0.04、−multi 45.96±0.04,`results/moving_kw/others_mkw.json`),STATIC 欄與 §4.3 STATIC 句移除;行名照教授(w/o GMC / w. GMC)。
- 版面:5 頁、正文第 4 頁收、第 5 頁純參考文獻、0 overfull、無未解引用。release `paper-2026-08-30` 的 assets 未動(此輪屬下一個里程碑)。

---

## paper-2026-08-30 — `gmc_v3.tex`(對 `gmc_v2.tex` @ 148b6be,即 paper-2026-08-26 之後的最後修訂)

2026-08-29 起的新一輪:`gmc_v3.tex` 由已 commit 的 `gmc_v2.tex` 複製;v2 不再改。本輪兩件事:A42(協議)、A43(方法),新的在上。

### [編輯] — 用語統一與 Table 1 位置(release 後同日修正,assets 已更新)

- 「和動作有關的 expression」在文中有四種說法(expressions referring to moving objects、movement-related expressions、explicit-motion expressions、MOVING HOTA),依作者要求統一為 **moving-class expressions / moving-class HOTA**:§4.1 首次出現處定義(= 分類器判為 MOVING 的 expression),§1、§4.1、§4.2、Table 2 標題、§5 共 8 處改寫;數字不變。
- Table 1 原本浮到第 4 頁(§4.2 在第 3 頁底),把表的原始碼移到 §4.1 之前並加 `[t]`,現在排在第 3 頁右欄頂;Table 2/3 仍在第 4 頁。頁數 5、第 5 頁純參考文獻、無 overfull。
- release `paper-2026-08-30` 的三個 assets 以此版重傳(PDF、latexdiff、來源包);tag 仍指向 c6d6571,修正 commit 見 release notes。

### [編輯] — 依 RMOT 文獻內容調查補充表格:Table 1 加 ΔDetA/ΔAssA、Table 3 加 STATIC 欄與單-α 列(A44;α 敏感度表因版面退回筆記)

- 調查:`docs/RESEARCH_RMOT_CONTENT_SURVEY_2026_08_30.md`(Zotero 10 篇 RMOT 論文)。9/10 篇主表列 DetA/AssA;5/10 篇有關鍵超參掃描表;消融表慣例列全指標並比對設計替代方案。作者決定:做 A(子指標)、B(α 小表,三個 host,FH V1 補跑 15 次)、E(單-α 列 n=5)、F(STATIC 欄);C/D/G/H 四句(TempRMOT 負結果、參數量、統計句、frame-convention 註腳)只留草稿於調查 §6,未進稿。
- 數字(A44):iKUN native→ship DetA +0.87、AssA +0.46(DetRe +1.52);FH V1 +0.15/+0.15;FH V2 +0.12/+0.04。單 α=0.35(LOSO)n=5:44.95±0.11 / MOVING 32.42±0.25 / STATIC 44.35±0.06,雙權重比它高 +0.33 / +4.57 / +0.18。α 掃描:FlexHook pooled 在 α*/2 到網格最大內變動 ≤0.1;iKUN 單 α 最佳 0.5(45.01),α=2 掉到 42.96。
- 版面:加了 α 敏感度表(Table 4,五列)後正文超出 4 頁約 30 行。作者決定(方案 Z)只退新加的、不動原文:Table 4 與其參照句移除,Table 1/3 標題縮短;保留 Table 1 新欄、Table 3 STATIC 欄與單-α 列;三張表 \footnotesize。重編後正文在第 4 頁結束、第 5 頁只有參考文獻、無 overfull。α 敏感度數字留在 `results/moving_kw/alpha_sweep_mkw.json` 與 RESEARCH_NOTES A44,之後有空間再放。
- 依據:A44,`results/moving_kw/{submetrics,alpha_sweep_mkw,v2_canonical_regroup_mkw_sweep,single_alpha_0.35_n5}.json`;一次性腳本(`submetrics_mkw.py`、`alpha_sweep_mkw.py`)不進 git。

### [方法] — MOVING 類別改用使用者給定的關鍵字清單;分類器同時決定 α 路由與分類報表;iKUN 重跑、FlexHook 重分組(A43)

- 舊分類器把 `turning`、`faster` 視為 MOVING;使用者改定 MOVING = {moving, in motion, driving, walking, running, jogging, crossing, riding, travelling/traveling, braking, brake, accelerat, decelerat, slowing down, speeding up, approaching, overtaking, receding},STATIC 七個字根不變,其餘為 APPEARANCE。統一放在 `gmc_link/moving_kw.py`,三個評測腳本與 V2 canonical regroup 都改為 import 它。
- 類別數量:V1 官方 150 條 MOVING 25 → **21**(`0011+turning-cars/-vehicles`、`0011+cars/vehicles-which-are-faster-than-ours` 改為 APPEARANCE);V2 862 條 canonical MOVING 136 → **111**。
- 同一分類器也是 iKUN 雙權重的路由器,所以 iKUN 15 個目錄(5 seeds × full/−ego/−multiscale)以新路由重跑到 `*_mkw` 樹(舊樹不動);對比舊樹只有兩條 `faster-than-ours` 的 predict.txt 不同(`turning-*` 在 iKUN 分數檔裡本來就沒有分數,任何 α 都是空預測)。FlexHook 兩個設定 α_mot=α_app,預測不變,只重跑 TrackEval 分組。STATIC 列在全部目錄逐位元相同(gate)。
- iKUN 雙權重在新路由下**重跑 LOSO**(受影響的 hold-0005 / hold-0013 兩折全網格 × 3 seeds;hold-0011 折不含改動句子,沿用):兩折 argmax 都是 (1.0, 0.1),第三折照舊 censor → 規則值 **(α_mot, α_app) = (1.0, 0.1)**,取代 (0.7, 0.1)。fold 曲線在 0.7–1.0 間平坦(差 0.03),由規則定;pooled 只差 +0.04,MOVING +0.86。FlexHook α=7 / 5 不受影響(α_mot=α_app)。
- 數字變動(新分類 + 新 α):iKUN native MOVING 25.778 → **27.697**(pooled 44.543、STATIC 43.914 不變);ship 45.158±0.104 → **45.304±0.115**(對已發表 +0.598 → **+0.744**);MOVING 32.902±0.660 → **37.139±0.923(+9.44)**。Table 3(n=5):native 27.70/44.54、full 36.99±0.68/45.28±0.09、−ego 32.37±0.39/44.61±0.07、−multiscale 35.14±0.38/45.00±0.02;Welch ego t=13.1/13.2、multiscale 5.3/6.8,全部 p<0.01;STATIC 43.91/44.53/43.40/44.35。(同路由下若維持 (0.7, 0.1):45.261±0.086 / MOVING 36.279±0.627,留在 `ikun_official150_mkw.json` 作對照。)FH V1 MOVING 44.31 → **47.90**,增益 +0.67±0.13 → **+0.43±0.19**;FH V2 canonical MOVING 38.15 → **38.56**,增益 +0.18±0.06 → **+0.69±0.04**;兩者 pooled 不變。
- 改動位置(全部在 `gmc_v3.tex`;`gmc_v2.tex` 維持 148b6be):§3.4 α 值、摘要、§1 貢獻、§4.1(「one-sixth」→「one-seventh」,21/150 = 14%)、§4.2 文字與 Table 1 iKUN 列、Table 2 標題(21/150、111/862)與三列、§4.3 文字與 Table 3、§5。**待作者處理的文字**:§5 「14 of the 126 direction expressions … treated as appearance」— 新分類器依設計把所有方向/轉彎/快慢句都歸 APPEARANCE,已不是「誤讀」;§3.4 建議加一句「α 在最終分類器下重跑 LOSO 選出;fold 曲線 0.7–1.0 平坦,取規則值」。
- §5 第二項限制改寫(2026-08-30,作者定稿):舊句「誤讀 → 14/126 方向句被當 appearance」在新分類器下不成立(方向/轉彎句依設計全歸 APPEARANCE);改為「分類器只把明確運動句路由到 α_mot;方向/轉彎句(214/818,V1 全部表達式)吃 α_app,模組對它們無助益」。計數:含 direction/turning/faster/slower 的 V1 句 218 條,其中 4 條同時含 moving 而路由到 α_mot。
- 依據:A43,`results/moving_kw/{ikun_official150_mkw_am1.0,loso_two_alpha_mkw,fh_mkw,v2_canonical_regroup_mkw}.json`;一次性腳本(`regroup_fh_mkw.py`、`rescore_official150.py --out-name`、`diagnostics/aggregate_official150.py --tree-suffix`)不進 git。

### [協議] — iKUN 改用 Refer-KITTI V1 官方 150 條測試表達式;iKUN 全部數字重算(A42)

- 舊的 iKUN 評測名單是我們自己列 `expression/{seq}` 資料夾得到的 158 條;benchmark 官方名單(TransRMOT `seqmap.txt` = FlexHook `kitti-1.txt` = iKUN `utils.py` 的 `dropped` 補集)只評 150 條。多出的 8 條(braking×2、horizon×2、back-to-the-camera×4)benchmark 從不評,iKUN 自己的 `test.py` 也跳過;在我們的樹裡它們只有害(braking 零預測對 99 行 GT、men-back-to-the-camera 205 預測對 11 行 GT)。FlexHook 兩個設定早已用官方名單(A31),不受影響。
- 修法:預測檔一個 byte 不動,只換成官方 seqmap 重跑 TrackEval(485 個目錄含 LOSO 折;做法見 `RESEARCH_NOTES.md` A42,一次性腳本不進 git)。LOSO 重選結果不變:雙權重 (0.7, 0.1)、單權重 0.35。
- 數字變動(158 → 150):native 44.224 → **44.543**(iKUN 已發表 44.56,差距 −0.34 → −0.02);ship 44.847±0.107 → **45.158±0.104**(對已發表 +0.283 → **+0.598**);MOVING 25.531→32.606(+7.08)→ **25.778→32.902±0.660(+7.12)**;STATIC 不變(8 條裡沒有 static)。
- 改動位置:摘要、§1 貢獻、§4.1(改寫為「所有 V1 設定皆用官方 150 條名單」、reproduced iKUN 44.543)、§4.2 文字與 Table 1(iKUN 列;Published 改為 iKUN README 的 44.56,原 44.564 是我們自己的 paper-pure 重現值)、Table 2 標題(改為「25/150 for iKUN and FlexHook V1」)與 iKUN 列、§4.3 文字與 Table 3(n=5:native 25.78/44.54、full 32.58±0.64/45.12±0.10、−ego 28.72±0.35/44.47±0.07、−multiscale 30.70±0.20/44.80±0.02;Welch ego t=11.8/12.1、multiscale 6.3/7.0)、§5。
- 依據:A42,`results/official150/ikun_official150.json`、`seqmaps/refer_kitti_v1_test_official_150.txt`。
- 註:本條的 iKUN 數字是本輪的中間值,同一輪內再被下面的 A43(新分類 + α 重選)覆蓋;v3 最終數字以 A43 條為準。這些修改原先寫在 v2 工作樹裡,2026-08-30 v2 還原為 148b6be,A42 只存在於 v3。

---

## paper-2026-08-26 — `gmc_v2.tex`(對 `gmc_v1.tex`)

2026-08-25 起的修訂輪(reviewer 視角逐段檢查)。程式碼僅動一處(A41 惰性 fallback,快取逐位元等值),所有已報 HOTA 數字不變;FPS 重測。

### [編輯] — §4.3 補 Welch 統計量(#29 item 2 / #31 item 1)

- 「Both drops are significant under Welch's $t$-test」→ 加括號:$n{=}5$;−ego $t{=}11.8$(moving-class)/ $11.7$(pooled);−multiscale $t{=}6.2$ / $6.7$;all $p{<}0.01$。
- 依據:A38(2026-08-25 重建)`diagnostics/welch_ablation_n5.py` → `results/ablation_n5_welch.json`,p = 1.75e-5 / 9.81e-6 / 1.84e-3 / 1.76e-3;平均值與 A34 一致。

### [編輯] — §3.2 補 warm-up 棄權規則(#31 item 2)

- Eq. (3) 說明段後加一段:軌跡需 11 幀連續歷史(最長間隔 10 + 1,三個間隔皆可定義)才產生運動特徵;不足時模組棄權、host 分數原樣通過;門檻由最長間隔決定,無自由參數。
- 依據:`filter_warmup_cache.py`(T_MIN=11),所有 ship 快取皆 `_warm11`;A3 量測(25.0% 軌跡幀受影響,1,239/4,950)。依用戶決定只寫機制、不引新數字。

### [編輯] — §3.2 補 1/g 實作註記(#31 item 3)

- 「All velocities are normalized …」句後加:實作省略 $1/g$;等價於第一層投影對每個 gap 的常數重縮放(Sec. 3.3)。公式與 "velocity" 用詞不動。
- 依據:`gmc_link/utils.py:38-60`、`manager.py:379-388` 僅以影像尺寸正規化、不除 gap;線性層可吸收每維常數(表示等價)。

### [筆誤] — §3.1 框遮罩用詞(#31 item 4)

- 「excluding the region where tracked object boxes overlap」→「excluding the interiors of the previous frame's tracked object boxes」。
- 依據:`gmc_link/core.py:77-83` 將前一幀每個偵測框內部整塊置零(`manager.py:295` 傳入 `prev_detections`)。

### [編輯] — §4.1 fallback 成功率擴到全資料集、FPS 重測(A39/A41)

- 「succeeds on all 2,065 adjacent frame pairs across the four evaluation sequences」→「all 7,690 adjacent frame pairs of the 19 training and evaluation sequences」;「$31.8$ FPS」→「$149$ FPS on CPU ($6.7$\,ms per frame)」。
- 依據(成功率):A39 `diag_road_chain.py`,19 條序列 7,690 對相鄰幀,路面擬合 0 次回傳 None(評估 2,065 對有偵測框遮罩;訓練 5,625 對無遮罩,即訓練程式的條件)。
- 依據(FPS):A41。`gmc_link/manager.py` road 模式改為只在路面擬合失敗時才估全域 ORB(從未發生);0011 seed0 快取重建與既有快取 183,872 筆逐值相同(max |Δ| 0.00),HOTA 數字不受影響。FPS 以同一 session、同機器、`profile_inference.py --seq 0011`、16 reps 取 warm 中位數:road 149.3(IQR 147.5–150.8)/ 舊 ship 全域鏈 63.9。A36 的 31.8/42.8 為當日機器狀態數字,存 `results/fps_profile_a36_2026_08_17.json`;不可跨 session 比較。
- 取代本段上方「[編輯] — ORB fallback 敘述全刪」條目中「程式碼保留 fallback,31.8 FPS 沿用」的說法。

### [編輯] — §3.1 補「為何取畫面下半部」的幾何理由

- RANSAC 句後加兩句:前視相機的地平線約在畫面垂直中央;其上是建物與天空(不在路面平面),其下是車前到地平線的路面,故只對下半部擬合。
- 依據:幾何(單應性只對平面精確);A39 光度殘差佐證 —— 起點高於地平線的區域(0.3/0.4)一律較差,0.5–0.7 在近路面打平、但只有 0.5 取樣到中距離物體所在區域(`results/road_diag/road_chain_diag_evalnear.json`)。數字不入文。

### [編輯] — §4.1 補訓練資料句(#29 item 1)

- optimizer 句後加:aligner 以 Refer-KITTI V1 訓練分割(15 條序列,與測試序列不交)訓練;正樣本 = expression × 其標註指涉的 ground-truth 物體之運動特徵(由 GT 框計算);tracker 輸出僅在測試時使用。
- 依據:`gmc_link/train.py:410-414, 717-719`(`--split v1` → 15 seqs)、`gmc_link/dataset.py:884, 926`(`labels_with_ids` GT 軌跡)、`refer-kitti/expression/` 818 條 = V1 發布數(#29 留言 2026-08-25)。

### [編輯] — ORB fallback 敘述全刪

- 刪方法段末句(路面擬合失敗 → ORB 全域單應性 fallback)與 Setup 的 fallback 參數句(1,500 keypoints / 5px 門檻)。
- 依據:路面擬合在四條評測序列 2065/2065 全數成功(A37 的支持測量,`results/road_fallback_rate.json`),fallback 從未觸發、不影響任何已報數字;程式碼保留 fallback,31.8 FPS 沿用。
- 保留「succeeds on all 2,065 adjacent frame pairs」作穩健性敘述;`\cite{orb}` 隨之移除(refs.bib 條目保留,BibTeX 不輸出未引用條目)。

### [筆誤] — Limitations 誤述 fallback 觸發機制

- 原句稱「非平面時退回 global estimate」;實際 fallback 只在特徵不足時觸發(<12 角點 / LK 追蹤失敗 / findHomography 回 None,`gmc_link/core.py:86-94`),與平面性無關 —— 路面不平時擬合多半「成功但有偏」,fallback 不會接手。
- 改為直述模型適用範圍(不作未量測的退化宣稱、不用情態詞):「assumes a flat ground plane; scenes with strong slopes or uneven terrain fall outside the model」。程式一直是對的。方法段 L120「A homography is exact for a plane」保留 —— 那句講的是理想模型類(射影幾何定理),是設計動機;Limitations 講的是估計的適用範圍,分工清楚。

### [編輯] — §2.2 三種速度定義補強(定義句用戶重寫)

- raw / ego 補上公式($v^{raw}_g=(o_t-o_{t-g})/g$、$v^{ego}_g=(\hat o_t-o_{t-g})/g$),單應性應用明寫齊次座標,ego 句補「evaluated at the object's location」。
- raw 的動機句改為字面精確的陳述:「it is the sum of the object's own motion and the motion induced by the camera」(原「is mixed with」語法含混;由 Eq. (2) 移項 $v^{raw}=v^{res}+v^{ego}$,此句字面為真)。
- residual 定義句(camera-motion-compensated,raw 減 ego)作 Eq. (2) 引導;Eq. (2) 補恒等式 $v^{res}_g=(o_t-\hat o_t)/g$(殘差 = 觀測質心對「靜止預測」的偏差),句尾逗號改句號(後接新句子)。
- 定義句草稿兩處事實更正:「consecutive frames」→ gap-$g$ 幀對($I_{t-g}$、$I_t$,gaps 2/5/10);「inter-frame homography」→ 累積單應性 $H_{t-g\to t}$。

## paper-2026-08-22 — `gmc_v1.tex`(對 `gmc.tex`)

配置:**Option B,2026-08-19 拍板**(`docs/SHIP_DECISION_2026_08_16.md`)— 三個 host 設定統一路面 ego 鏈、warm11 遮罩、無 motion EMA、raw cosine、類別權重加法融合。

### [方法] — 配置改了,實驗重跑

每一步的增益都單獨量過(iKUN,累積疊加;出處 `docs/SHIP_DECISION_2026_08_16.md`):

| 累積配置 | pooled | Δ pooled | MOVING | Δ MOVING |
|---|---|---|---|---|
| native(無模組) | 44.224 | — | 25.531 | — |
| + GMC 模組(12D、單 α)= gmc.tex 版 | 44.512 | +0.288 | 30.222 | +4.69 |
| + warmup 遮罩(下述第 1 條) | 44.634 | +0.122 | 30.043 | −0.18(噪聲內) |
| + 刪 EMA、similarity ego(第 2 條) | 44.656 | +0.022 | 30.045 | +0.00 |
| + 路面估法 + 雙權重(第 3、4 條) | **44.847** | +0.191 | **32.606** | +2.56 |

warm11 與刪 EMA 兩步是為 pooled 增益和正確性(訓練/推論一致、不融合垃圾)而採,
MOVING 本來就不預期動;−0.18 在 n=3 種子變異範圍內。

**1. Warmup 遮罩(warm11)—— 沒把握就閉嘴**

- 問題:每條軌跡自己的前 10 幀(連續歷史 ≤ 10 = 最長速度時距)算不出長時距速度,程式補零。零速度看起來就是「靜止」,所以剛進畫面的移動車會被一個錯的分數大力扣分。這種幀佔測試資料 25.1%(實驗記錄 A3,編號說明見檔頭)。
- 作法:這些幀的 GMC 分數從 cache 刪除,融合端查不到就加 0 —— host 的分數原樣通過,等同模組對該幀棄權;第 11 幀起才生效(故名 warm11)。零新超參數。
- 效果:iKUN pooled +0.122。

**2. 刪掉推論期的速度平滑(EMA)—— 考試和練習要用同一套**

- 問題:推論時速度特徵有做平滑,訓練時沒有。模型從沒看過平滑後的資料(A2)。
- 作法:刪掉平滑,推論吃和訓練一樣的原始特徵。
- 效果:+0.022(與同批清理合計)。

**3. 相機運動改用路面估(核心改動)—— 在平的東西上量,量得才準**

- 問題:單應性這種變換只有對「一個平面」才準。舊作法把它擬合在整張畫面的背景點上 —— 建築、桿子、停放車輛,深度都不同,估出來的相機運動被視差帶偏。
- 作法:只用畫面下半部的路面點來估(Shi-Tomasi 角點 + LK 光流,RANSAC 3px)。路面接近一個平面,假設成立。舊的全域估計留作備援。
- 為什麼確定是這個原因:
  - 拆開驗證(A25):只換路面估法,運動類 +1.46(t=3.9);只換另一個候選因素(地面接觸點),+0.18,等於沒動。功勞在路面估法。
  - 在哪裡起作用(A29):三條測試序列裡,舊估法在 seq 0011 幾乎分不出「符合運動描述的車」和「不符合的車」(分離度 0.022);換路面估法後變 0.135,6 倍。0011 也正是運動類進步最多的序列(+2.23)。
  - 「路面點夠不夠」的疑慮:2,065 對評測幀全部擬合成功,備援一次都沒用到。
- 效果:見第 5 條的合併數字。

**4. 融合權重一個變兩個(α_c)—— 運動句大聲、外觀句小聲**

- 問題:模組只懂運動。遇到「black cars」這種外觀句,它的分數是噪聲。只有一個權重時,調大傷外觀句、調小浪費運動句。
- 作法:句子先分類(關鍵字),運動/靜止句用 α_mot,外觀句用 α_app。iKUN 交叉驗證選出 (0.7, 0.1) —— 外觀句幾乎關掉。
- 為什麼 FlexHook 不需要:它底層本來就看運動,模組的分數對它的外觀句不是噪聲,沒東西可分流 —— 交叉驗證自己選出兩個權重相等(7 / 5),等於退回單一權重(A35)。
- 對照組:把雙權重放在「舊」估法上只 +0.016 —— 效果來自「路面估法 × 分流」的組合,不是分流自己(A32)。

**5. 三個 host 統一用路面估法(A37)**

- 代價與收穫:FlexHook 兩設定 pooled 各降 0.03,換到運動類 V1 增益近乎翻倍(+0.49 → +0.67)、V2 翻四倍(+0.048 → +0.184,從噪聲內變成顯著)。
- 和第 3 條一致:路面估法改善的是「運動/靜止分得開」,運動類直接受惠,pooled 因為運動句只佔六分之一所以幾乎無感。
- 定案數字:iKUN 44.847 ± 0.107、FH V1 53.980 ± 0.059、FH V2 42.625 ± 0.032 —— 三行全部超過發表值。

### [協議] — 評測修正後重測

- **FH V1 句子名單**:158(含 8 條格式異常句)→ 官方 150。重現 native 從 53.110 變 **53.824 = 發表值**;舊 Table 1 的「重現落差」註腳消失,因為落差是名單錯誤的產物(A31)。
- **V2 逐類分組**:改寫 slug 分類(108/862 句分錯)→ canonical `raw_sentence`。MOVING baseline 48.02 → 38.15,增益從 −0.07 翻正為 +0.18(A30、A4)。
- **FPS**:舊值 68 是髒量測;乾淨重測(process-only、CPU)路面鏈 31.8 / 全域鏈 42.8(A36)。論文報 31.8。
- **LOSO**:從事後穩健性檢查變成選點程序本身,搜索格加密且無截斷(A24、A37)。任何權重都不在自己被評測的序列上選。
- **消融重定基**在 Option B 工作點,n=5(A34):−ego 為 MOVING −3.81 / pooled −0.64(t≈11.8),−multiscale −1.85 / −0.32。

### [筆誤] — 稿子錯,程式一直對

- **§3.2 ego 速度定義**:舊稿寫成 warp 後質心與「當前」質心的距離 —— 那個量是殘差本身,使公式變成「殘差 = 原始 − 殘差」的自我矛盾。改為 ego 位移 $\hat{o}_t - o_{t-g}$(`gmc_link/manager.py:385`)。

### [編輯]

- 版面:開 `\ninept`;刪 tikz 流程圖(與架構圖重複)。6 頁 → 5 頁,第 5 頁純參考文獻。
- 方法名全篇 GMC-Link → **GMC Module**。
- 架構圖改用 Excalidraw 重繪(`figures/Architecture.excalidraw`),取代 PowerPoint 源;修正語言分支兩處誤標成 Motion 的向量、回饋箭頭補上融合權重、host 分數從向量條改為純量(箭頭標籤)。
- 文獻:刪 `mlstrack`/`cdrmot`/`tellmewhat`,加 STORM(對 CVPR 2026 Findings 核對過);LTTrack **不引用** —— 查無可信來源。
- Setup:基準揭露(重現 iKUN 44.224 vs 發表 44.564)、兩條估計鏈的參數、seeds 與協議句。
- 限制段改寫成三條可查證項目:路面平面假設 + 回退、關鍵字誤路由(V1 方向句 14/126)、FlexHook 雙權重退化。
- 摘要補上代表數字;67 個紅字標記全部拆除(PDF 文字逐字元驗證未變)。
- 作者拍板維持刪除:趨勢解釋段、n=3 但書、TempRMOT 範圍段、消融支撐的貢獻條目。

---

## paper-2026-08-19 — `gmc.tex`(對 `paper/latex/mainv3.tex`,2026-08-05 MMAsia 投稿版)

論文移植到 2026-08-10 簡化配置(「Option A 前身」:12D、單一 α、全域 ORB similarity 鏈)的 ICASSP 版。

### [方法]

- **運動特徵 13D → 12D**:ρ(殘差對背景 SNR)槽位移除,消融顯示無 HOTA 代價(教授指示的簡化,2026-08-10)。
- **融合:逐類配方 $s_{host} + \alpha(sc\cdot\cos + thr)$**(每 host 運動/外觀兩軸,約 18 個手調超參數)→ **單一加法權重** $s_{host} + \alpha\,s_{gmc}$,LOSO 選點(0.5 / 2 / 5)。分數側 sigmoid + EMA 移除;raw cosine。
- 當時數字:iKUN 44.512 ± 0.104、FH V1 53.157 ± 0.022(對重現值 53.110)、FH V2 42.684 ± 0.058。

### [編輯]

- 新開 ICASSP 工作區(`2027_ICASSP/`),spconf 模板;MMAsia 稿凍結為 `paper/latex/mainv3.tex`。
- 移植時多段分析被註解(LOSO、TempRMOT 範圍、趨勢解釋、GPS/IMU);去留在 2026-08-22 那輪定案(issue #23/#24)。

---

## 附:本文引用的實驗記錄一覽

| 編號 | 做了什麼 | 結果 |
|---|---|---|
| A2 | 盤查全管線,發現速度特徵在推論期有 EMA 平滑、訓練期沒有 | 訓練/推論分布不一致確認;移除後 +0.022(與同批清理合計) |
| A3 | 統計測試軌跡的歷史長度,量化「補零速度」幀的比例 | 25.1% 的 track-frame 缺長時距歷史 → 催生 warm11 遮罩 |
| A4 | 檢查 V2 逐類評測的標籤空間 | 分類器跑在改寫過的 slug 上,108/862 句分錯 → 改用 canonical 句 |
| A24 | iKUN 的 LOSO 重跑:搜索格加密、去掉邊界截斷 | 選出的 α*=0.5 不變 → 選點程序本身可信 |
| A25 | 歸因 2×2:「路面估法」與「地面接觸點」兩個候選因素分開開關,每臂獨立訓練 + 全評測,n=3 | 只換路面估法 MOVING +1.46(t=3.9);只換接觸點 +0.18(t≈0.55)→ 功勞在路面估法 |
| A29 | 逐幀分數分離度探針:GT 物件以 IoU≥0.5 對應 tracker 軌跡,量「符合運動句的軌跡平均分數 − 不符合的平均分數」 | seq 0011 分離度 0.022 → 0.135(6 倍),0005 本來就高、不變 → 定位路面估法在哪裡起作用 |
| A30 | V2 逐類評測改用 canonical 句重跑 | MOVING 負值異常消失,四類增益全非負 |
| A31 | FH V1 換官方 150 句名單重評 | 重現 native 53.110 → 53.824 = 發表值 → 「重現落差」是名單錯誤,不是管線問題 |
| A32 | 雙權重 × 路面估法,LOSO 二維格點選 (α_mot, α_app),n=3;另設「雙權重 × 舊估法」對照組 | iKUN pooled 44.847(+0.19)、MOVING +2.56;對照組僅 +0.016 → 增益來自組合 |
| A34 | 兩個 ship 候選配置各做 n=5 消融(固定 LOSO 工作點) | Option B 臂:−ego 為 −3.81/−0.64、−multiscale 為 −1.85/−0.32,全部顯著 |
| A35 | FlexHook 也照跑雙權重 LOSO(V1 三折、V2 四折)+ 格外探針確認最佳點在格內 | 各折增益 ≤ +0.10、出格點全下跌 → FlexHook 維持單權重 |
| A36 | FPS 乾淨重測(seq 0011、n=500 幀、process-only、CPU) | 路面估法 31.8 / 全域估法 42.8 FPS;取代舊的髒量測 68 |
| A37 | FlexHook 在路面估法上重跑單 α LOSO(預登記先 commit 再跑) | V1 α*=7 → 53.980、V2 α*=5 → 42.625;運動類 V1 +0.67、V2 +0.184(轉顯著) |
| A38 | n=5 消融的 Welch 統計量重建(`diagnostics/welch_ablation_n5.py`) | −ego:運動類 t=11.8、pooled t=11.7;−multiscale:6.2 / 6.7;全部 p<0.01;STATIC 只有 −ego 顯著、APPEAR 皆不顯著 |
| A39 | 路面估法逐幀對診斷:19 條序列 7,690 對相鄰幀,以光度殘差當裁判(對 identity、對舊全域估法) | fallback 0/7,690(含訓練序列);HGATE 門檻對路面估法無效(h32 是物理量);路面估法 90% 幀對贏全域;下半幅 = 地平線以下最寬區域;ORB 在路面**找得到點**但擬合較差 |
| A40 | 由既有掃描量出單/雙權重的耦合;ORB 機制與訓練/推論遮罩差異的補充量測 | APPEAR 在 α≈0.2 達峰後單調下降、MOVING 到 1.0 才封頂 → 單一 α 是被迫折衷;ORB 差在取樣點深度(內點靠近地平線),非亮度/門檻;遮罩差異二階(p50 9.8 px),記錄不重訓 |
| A41 | road 模式改為只在路面擬合失敗時才估全域 ORB(從未發生);快取重建逐值相同;FPS 同 session 重測 | 0011 seed0 快取 183,872 筆 max|Δ|=0.00,HOTA 不變;road 149.3 FPS(6.7 ms/幀)/ 舊 ship 63.9;A36 的 31.8/42.8 為當日機器狀態數字 |
| A42 | iKUN 改用官方 150 條測試名單重跑 TrackEval(預測不動;485 目錄含 LOSO 折) | native 44.543(已發表 44.56);ship 45.158±0.104、MOVING +7.12;LOSO 重選 α 不變;消融 t 值 11.8/12.1、6.3/7.0;舊 158 名單是我們自己列資料夾多算 8 條 |
| A43 | MOVING 類別改用使用者給定關鍵字清單(`gmc_link/moving_kw.py`),同一分類器路由 α 與分組;iKUN 15 目錄重跑、兩折 LOSO 重選;FlexHook 重分組 | V1 MOVING 25→21 條、V2 136→111;LOSO 改選 (1.0, 0.1);iKUN 45.304±0.115(對已發表 +0.744)、MOVING +9.44;FH V1 MOVING +0.43、V2 +0.69;STATIC 逐位元不變 |
| A44 | RMOT 文獻內容調查(Zotero 10 篇)→ 子指標、三 host 單 α 掃描(FH V1 補跑 15 次)、單 α=0.35 n=5 | iKUN ΔDetA +0.87 / ΔAssA +0.46;FH V1 +0.15/+0.15、V2 +0.12/+0.04;雙權重比單權重 +0.33 pooled / +4.57 MOVING;FlexHook pooled 對 α 平坦(≤0.1)、iKUN 單 α 最佳 0.5;α 表因版面退回筆記(Z) |
| A45 | 合併教授改稿(措辭照教授、數字照 v3);Table 3 Others 欄新量測(非 moving 129 條,21 目錄 TrackEval) | Others:native 45.87、full 46.05±0.05、−ego 45.77(最低)、−multi 45.96、單α 45.99;Table 1 改列 DetA/AssA 數值;引用 24→19 |
| A46 | TransRMOT 第三架構寫入 v3(8 處):Table 1/2 各加一行、Table 2 依 baseline 排序、§4.2 補回反比主張、§4.1 兩個公開值揭露、§3.4 權重 (0.7, 0.5)、「two-stage」措辭放寬、結論限制改三架構、Intro bullet;子指標補量(4 次 pooled TrackEval) | TransRMOT:native 45.757(官方 46.56 為幀修正前、repo 38.06 為另一 GT 約定)、w. GMC 46.149±0.025、DetA 36.64→36.95、AssA 57.40→57.89、moving-class +1.43±0.13;四設定三架構 gain 對 baseline 單調遞減;diff 基底仍為教授 gmc_v2_1 |
| A46b | §3.4 四句 per-host 權重散文改為 α 總表(`tab:alphas`,4 host × α_mot/α_app;使用者決定)——FlexHook 路由退化(α_mot=α_app)首次表格化可見;表號順移(main/deficit/ablation → 2/3/4) | 編譯 5 頁 0 overfull;diff 重生 |
| A46c | reproduced 揭露改社群慣例(查證 C²RMOT「‡ denotes the baseline ... reproduced using the official implementation」、HFF「♣ ... after frame correction」):Table 2 TransRMOT 值標 ‡ + caption 一句;§4.1 刪 iKUN 揭露句(回教授判斷)、TransRMOT 三句半壓一句、FlexHook 半句保留 | 正文從 3.5 句縮到 2 句;編譯 5 頁 0 overfull;diff 重生 |
| A46d | Table 4(ablation)去 ± :五行 seeded cell 只列平均(使用者決定);n=5 與顯著性由 caption + 正文 Welch t 承載,正文首句保留 36.99±0.68 / 45.28±0.09 當唯一 std 錨點 | 編譯 5 頁 0 overfull;diff 重生 |
| A46e | Table 2 reproduced 記號 ‡ → *(使用者覺得 ‡ 突兀;* 為 reproduced 最常見記號,全文無上標 * 衝突);cell + caption 兩處 | 編譯 5 頁 0 overfull;diff 重生 |
| A46f | 四個無 placement 的 float(tab:alphas/deficit/ablation、fig:qual)補 [t]:p4 右欄原本整欄退化成 float column(Table 4 + Fig.2 中間大段彈性空白、正文跳頁);[t] 禁掉 float-page 路徑後正文回流 | 編譯 5 頁 0 overfull;p4 右欄 Table 4 → Fig.2 → 結論正文連續;diff 重生 |
| A46g | float/caption 間距參數:floatsep 7pt、textfloatsep 9pt、above/belowcaptionskip 4/4pt(caption 在表上方時預設 10/0pt 方向顛倒——float 頂 10pt 死空間、caption 貼表線) | 表2↔表3↔正文間距正常化 |
| A46h | Table 2 改配對列(每 host 一列 baseline + 一列 +GMC;欄位 HOTA/DetA/AssA):原 5 欄 + →對 + colsep 1.5pt 擠到不可讀;caption 改寫(baseline 列 = 發表 HOTA + 我們的 DetA/AssA);* 記號保留(TransRMOT 無可比發表值) | 編譯 5 頁 0 overfull;diff 重生 |
| A46i | Table 2 照社群主表樣式重排(FlexHook/C²RMOT/iKUN 三篇查證):Refer-KITTI / V2 兩節、每 host 配對加 \hline 分組、+GMC 列粗體、表頭 Host→Method;§4.2 刪「哪些列來自 V1/V2」句(表內分節已載明) | 編譯 5 頁 0 overfull;diff 重生 |
| A46j | Table 2 +GMC 列去 ±(caption 改「mean over three seeds」);與 A46d ablation 表同邏輯——std/顯著性由 Table 3 與正文承載,參考論文主表皆單值 | 編譯 5 頁 0 overfull;diff 重生 |
| A46k | 表格側向空白:四表改 tabular*{\linewidth}+extracolsep fill 撐滿欄寬(原本 Table 3 僅 ~60% 欄寬、置中兩側大空白,caption 卻全寬);float 間距統一 floatsep 10pt / textfloatsep 11pt(左欄 ~18pt vs 右欄 ~10pt 不一致) | 編譯 5 頁 0 overfull;diff 重生 |
| A46l | latexdiff 後處理擴充(perl 三規則):舊 sed 只換 DIFaddendFL+hline;新表的 DIFaddbeginFL+hline 與 DIFaddbeginFL+multicolumn(展開 \omit 必須是 cell 首 token)也要搬——marker 移到 hline 後 / multicolumn cell 內 | diff rc=0、0 error、5 頁 |
| A46m | §4.3 正文去 ±(36.99±0.68 / 45.28±0.09 → 均值):A46d 砍表格 ± 時留的散文錨點也移除;顯著性由 Welch t(n=5, p<0.01)與 Table 3 承載 | 編譯 5 頁 0 overfull;diff 重生 |
| A46n | Table 4 caption 補回 "seeds"((n=5) → (n=5 seeds)):d54a96a ICASSP 頁預算縮 caption 時砍的,與 Table 3 的 "n=3 seeds" 不一致且歧義 | 編譯 5 頁 0 overfull;diff 重生 |
| A46o | §3.4 首次出現處引入縮寫 "leave-one-sequence-out (LOSO)":Table 4 caption 的 (LOSO) 原本是全文唯一縮寫、未定義 | 編譯 5 頁 0 overfull;diff 重生 |
| A46p | §4.3 Welch 句改明確(比較對象寫清楚:full module 五 seed vs 各 variant 五 seed;t 值逐一配 metric;來源 welch_full_vs_arm);Table 4 −ego/−multiscale 去掉 \quad 縮排(log am1.0 證實兩臂在雙權重 ship 點跑,原縮排誤示為 single-α 子變體) | 編譯 5 頁 0 overfull |
| A46q | Fig.2 正文開頭 run-up 句合併("Figure 2 shows a sample tracking result." 併入下一句) | diff 重生 |
| A46r | latexdiff 配方再修:加 --config PICTUREENV=(?:picture|DIFnomarkup|table)[\\w\\d*@]* 把 table 環境設為不可標記區塊(結構性改表時 latexdiff 會把新舊列黏同格、把被刪散文塞進新表 cell——Table 1 爆版、Table 4 出現 32.30±0.6336.99);表格顯示最終版、散文照常標記;perl 三規則保留 | diff 5 頁 0 error,四表乾淨 |
| A46s | Table 2 周圍再調:belowcaptionskip 4→7pt(caption 末行貼表頂線)、arraystretch 1.1(分節列被 hline 夾死、列高擠);Fig.2 開頭句改回原兩句寫法(使用者:原寫法較清楚,A46q 的合併撤銷) | 編譯 5 頁 0 overfull;diff 重生 |
| A46t | diff 配方第 4 條 perl 規則:latexdiff 會把有改動的表整個包成 \DIFaddbegin \begin{table}(未改的表是裸的)→ 對這類表在 caption 開頭注入 \DIFaddFL{[changed]} 藍字標記;表本體仍顯示最終版 | 4 表標記,diff 5 頁 0 error |
| A46u | §4.2 exceed-published 句壓一行(刪 +0.744/+0.156/+0.099 三個 delta,主張保留:gains across all settings, including over published where directly comparable;TransRMOT vs our reproduction):防 reproduction-slack 質疑的功能不變,省 ~2 行(MVA 4 頁限制的頭期款) | 編譯 5 頁 0 overfull;diff 重生 |
| A46v | A46u 句再改寫(使用者:where directly comparable 難讀):點名式兩短句——For iKUN and both FlexHook settings, the fused model also exceeds the host's published score; for TransRMOT, the comparison is against our reproduction | 編譯 5 頁 0 overfull;diff 重生 |
| A46w | §4.2 Table 3 段四值枚舉改引兩端(from +9.44 on iKUN down to +0.43 on FH V1):數值由表承載,正文只撐量級差+單調主張;再省 ~1.5 行 | 編譯 5 頁 0 overfull;diff 重生 |
| A46x | diff 重生配方固化為 2027_MVA/rebuild_diff.sh(latexmk v3 → latexdiff PICTUREENV → perl 四規則 → latexmk diff → 頁數/overfull/標記數摘要);之後每輪一條指令 | v3 5 頁 0 overfull;diff 5 頁 0 error、4 表標記 |
| A46y | Table 1 改 [b] 沉 p3 右欄底(原 [t] 卡右欄頂,把 §3.4 的論述切成兩半);§4.2 兩個極值數字刪(order-of-magnitude 主張留,值由 Table 3 承載,再省 ~1 行);rebuild_diff.sh 規則 4 regex 放寬 [t]→[tb] | 編譯 5 頁 0 overfull;diff 5 頁 0 error、4 表標記 |
| A46z | Table 1 正式歸 Setup(使用者決定):§3.4 刪 listed-in-Table-1 句(收在 median over folds + FlexHook 退化句),§4.1 LOSO 段補該句、表源隨行([b] 保持)——方法章只講程序、選出的值屬實驗配置;表現落 p3 右欄底、緊鄰指涉段 | 編譯 5 頁 0 overfull;diff 5 頁 0 error、4 表標記 |
