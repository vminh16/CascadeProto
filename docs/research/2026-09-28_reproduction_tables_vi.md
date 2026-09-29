# Tái lập các bảng của CascadeProto: hai phần, và còn thiếu gì

Ngày 2026-09-29 (viết lại bản 2026-09-28). Sắp theo bảng của paper. Số liệu là mIoU % trên S3DIS, 2-way 1-shot, giao
thức `fixed100` (1.500 episode), một seed, trừ khi ghi khác. Nguồn chi tiết: `2026-09-21_reproduction_report.md`.

## 0. Cách đọc: hai phần, cùng một bộ checkpoint

| Phần | Nội dung | Mức mIoU | Có hợp lệ như kết quả few-shot không |
| :--- | :--- | ---: | :--- |
| **A** | Cài đặt đúng theo paper, chấm đúng giao thức (class test chưa từng thấy lúc train). Làm trước khi phát hiện leakage | ~50 (baseline 49.08, full 57.15) | Có |
| **B** | Thử đổi cách chấm để leakage xảy ra: checkpoint fold S0 chấm trên class của fold S1 và ngược lại (class đã thấy lúc train) | ~77 (baseline 77.32, full 81.28) | **Không.** Chỉ là chẩn đoán (D-22): khớp với số của paper, không chứng minh paper làm vậy |

Hai phần **dùng chung checkpoint**. B không cần train thêm: chỉ chấm lại cùng checkpoint trên fold đổi chỗ, bằng
`eval.py --allow_seen_classes true` (`experiments/run_seen.sh` là script duy nhất đặt cờ này). Nên mỗi lượt train mới
cho ra hai cột (A và B).

Cách map cột: cột "S0" của phần B là model train trên S0, chấm trên 6 class của fold 1 (đã thấy); cột "S1" là model S1
chấm trên 6 class của fold 0. Số "77" của phần B là checkpoint `last`; với `best` thì thấp hơn (baseline 71.40, full 75.99).

**Không nhầm với leakage thứ hai.** Repo có hai thứ đều gọi là leakage:

1. **Chấm trên class đã thấy** (D-21, D-22, report §3.5–3.6): đây là phần B.
2. **Shortcut vị trí query** (D-35, D-37): head VIP-Seg đọc thứ tự query trong episode. Nó nâng số phase 16 (E1: 73.20 `last`
   / 75.05 `best` trên S1) lên +19.94 so với head sạch. Không thuộc các bảng của paper; nằm ở mục 3.

---

## 1. Paper có 6 bảng

| Bảng | Nội dung | Loại | Phần A hiện có | Phần B hiện có |
| :--- | :--- | :--- | :--- | :--- |
| Table 1 | So sánh định tính (pre-training, modality, backbone) | không có số | chưa viết (không cần chạy) | không áp dụng |
| Table 2 | S3DIS: 2/3-way × 1/5-shot × S0/S1/Avg, 3 modality | kết quả chính | một ô: 2-way 1-shot, text, chỉ S0 | một ô: 2-way 1-shot, text, chỉ S0 (full) |
| Table 3 | ScanNet: cùng lưới | kết quả chính | **chưa** (không có dữ liệu ScanNet) | chưa |
| Table 4 | Ablation (LMA, Entropy Gate, Cascade, ADRM) | ablation | đủ 5 hàng trên S0; baseline có S1 | baseline đủ S0 và S1; full chỉ S0; 3 hàng giữa chưa chấm |
| Table 5 | Độ sâu cascade T = 1…6 | ablation | T = 1 và T = 4, S0 | chưa |
| Table 6 | Params / FLOPs / mIoU (S0) | độ phức tạp | params đã đếm; FLOPs chỉ baseline | không áp dụng |

---

## 2. Phần A: cài đặt đủ, chưa phát hiện leakage (mức ~50)

### 2.1 Kiểm tra pipeline (điều kiện để tin mọi số sau)

| Kiểm tra | Của mình | Tham chiếu |
| :--- | ---: | ---: |
| Checkpoint S0 của VIP-Seg, chấm bằng `eval.py`, data và metric của mình | 71.97 | 72.20 (paper Tab.6 / log VIP-Seg) |
| Model VIP-Seg train bằng loop của mình, 2.400 episode | 69.48 | 68.9 (script gốc của VIP-Seg, cùng thời điểm) |

Data, sampler, loss, optimizer và metric khớp VIP-Seg. Khoảng cách với paper không đến từ pipeline.

### 2.2 Table 4: ablation thành phần (S0, lịch train đầy đủ của paper)

| Cấu hình | Của mình (best) | Paper S0 | Chênh |
| :--- | ---: | ---: | ---: |
| Baseline | 49.08 | 82.72 | −33.6 |
| + LMA | 49.65 | 83.98 | −34.3 |
| + Entropy Gate (T = 1) | 56.72 | 85.34 | −28.6 |
| + Cascade (T = 4) | 56.55 | 87.89 | −31.3 |
| + ADRM (full) | 57.15 | 88.53 | −31.4 |

| Bước | Của mình | Paper | Kết luận |
| :--- | ---: | ---: | :--- |
| Tổng: baseline → full | +8.07 | +5.81 | tái lập (lớn hơn) |
| + ADRM | +0.60 | +0.64 | tái lập |
| + LMA | +0.57 | +1.26 | cùng dấu, nhỏ hơn một nửa |
| + Entropy Gate (1 stage) | +7.07 | +1.36 | lớn gấp 5 lần |
| + Cascade (1 → 4 stage) | −0.17 | +2.55 | không tái lập |

Baseline trên S1: 51.91 (paper 79.83). Mức tuyệt đối thấp hơn paper khoảng 30 điểm; không quyết định nào của mình trên
đường đi của baseline giải thích được khoảng này (report §3.3).

### 2.3 Table 5: độ sâu T (S0)

| T | 1 | 2 | 3 | 4 | 5 | 6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Của mình | 56.72 | — | — | 57.15 | — | — |
| Paper | 85.21 | 86.43 | 87.78 | 88.53 | 88.51 | 88.37 |

Paper tự mâu thuẫn: Table 4 hàng 3 và Table 5 T = 1 là cùng một cấu hình nhưng in 85.34 và 85.21.

### 2.4 Table 2: chỉ ô 2-way 1-shot, text

| Phương pháp | S0 | S1 | Avg |
| :--- | ---: | ---: | ---: |
| VIP-Seg (checkpoint phát hành, chấm lại) | 71.97 | — | — |
| VIP-Seg (paper) | 72.20 | 76.09 | 74.15 |
| CascadeProto (Text), của mình | 57.15 | — | — |
| CascadeProto (Text), paper | 88.53 | 84.53 | 86.53 |

### 2.5 Table 6: độ phức tạp

| | Params | FLOPs (fvcore) |
| :--- | ---: | ---: |
| Baseline của mình (encoder 2.37M + head 0.20M) | 2.58M | 19.4 G* |
| Full của mình (đếm theo thiết kế: + LMA, 4 EPPM, ADRM = 0.47M) | ≈ 3.04M | chưa đo (tool lỗi) |
| Paper: VIP-Seg / CascadeProto | 2.76M / 2.88M | 8.48 / 8.86 G |

\* fvcore không đếm kernel CUDA tuỳ biến (Mamba, FPS), nên FLOPs chỉ là cận dưới, không so trực tiếp với paper. Riêng
các module thêm vào đã cần 0.47M, nhiều hơn mức ≈ 0.12–0.31M mà paper ngụ ý.

### 2.6 Các chỗ mơ hồ đã chọn bằng đo (harness ngắn, 3 seed)

| Chỗ mơ hồ | Kết quả |
| :--- | :--- |
| D-02: gate đặt lên prototype hay feature | không khác biệt |
| D-10: có chia 1/√D ở Eq.23 không | không chia; chia làm mất 5.5 điểm |
| D-18: scale trong softmax Eq.14 | giữ như bản in; chuẩn hoá làm mất 0.9 |

---

## 3. Phần B: thử đổi leakage, đạt ~77 (chẩn đoán, không phải kết quả)

### 3.0 Hai run trực tiếp: cùng một baseline, chỉ đổi fold chấm (chạy lại 2026-09-29 trên VM)

Cùng một checkpoint: baseline của Table 4 (hàng 1: `use_lma=false`, `num_stages=0`, `l2norm_point_proto=false`, D-17),
`log_phase14/s3dis_S0_N2_K1_point_T0/last.pt` (epoch 50, train trên S0). `fixed100`, 1.500 episode, một seed.

| Run | Fold chấm | Class chấm | `protocol_check` | mIoU | Số cũ |
| :--- | :--- | :--- | :--- | ---: | ---: |
| 1. Baseline dựng đầu tiên | 0 | beam, board, bookcase, ceiling, chair, column (chưa thấy) | `clean` | **48.92** | 49.07 |
| 2. Baseline chứng minh leakage | 1 | door, floor, sofa, table, wall, window (đã thấy) | `SEEN-CLASS DIAGNOSTIC` | **77.32** | 77.32 |

* Chênh giữa hai run: **+28.4 điểm** cho cùng trọng số, chỉ vì class chấm đã hoặc chưa được thấy lúc train. Đây là số
  của cột S0 phần A và phần B.
* Run 2 khớp số cũ (0.773237 so với 0.773239). Run 1 lệch số cũ 0.15 điểm (48.92 so với 49.07): cùng checkpoint và
  cùng lệnh, nên là chênh do episode cache của fold 0 hoặc CUDA không tất định (AGENTS §6). Không thay đổi kết luận.
* Với `best.pt` (epoch 20) số cũ là 49.08 (fold 0) và 71.40 (fold 1): chênh +22.3. Hai run mới chỉ dùng `last.pt`.
* Kết quả: `results/seen_b2/A_baseline_unseen_last.json`, `ctrl_baseline_last.json`, log `run2.log`, `seen_b2.log`.
  `run.sh` trong thư mục đó là hàng đợi 7 lượt đã bị hủy sau lượt đầu; chỉ lượt `ctrl_baseline_last` chạy.
* Hai lượt này chạy khi run D-49 vẫn chiếm GPU. Không ảnh hưởng số mIoU; có thể làm chậm run D-49.

### 3.1 Table 4 theo cách chấm trên class đã thấy (`last`)

| Hàng | S0 (model S0 chấm class fold 1) | S1 (model S1 chấm class fold 0) | Avg | Paper S0 / S1 / Avg |
| :--- | ---: | ---: | ---: | :--- |
| Baseline | **77.32** (best 71.40) | 71.58 | 74.45 | 82.72 / 79.83 / 81.28 |
| + LMA | chưa chấm | chưa có model | — | 83.98 / 80.99 / 82.49 |
| + Entropy Gate (T = 1) | chưa chấm | chưa có model | — | 85.34 / 82.48 / 83.91 |
| + Cascade (T = 4) | chưa chấm | chưa có model | — | 87.89 / 84.05 / 85.97 |
| + ADRM (full) | **81.28** (best 75.99) | chưa có model | — | 88.53 / 84.53 / 86.53 |
| VIP-Seg phát hành | 79.13 | chưa chấm | — | 72.20 / 76.09 / 74.15 (chấm đúng giao thức) |

Nguồn: `results/seen/` (baseline, full, VIP-Seg), report §3.6. Khoảng cách tới paper trên cột Avg của baseline: 30.8 điểm
(giao thức chuẩn) còn 6.8 điểm (chấm trên class đã thấy). Ba dòng khác cũng cho thấy khớp:

* Có thứ tự S0 > S1 như paper (77.32 > 71.58); mọi phương pháp khác trong Table 2 có thứ tự ngược lại.
* Baseline chấm trên class đã thấy (77.32) đứng trên VIP-Seg chấm đúng giao thức (72.20), giống Table 4 của paper.
* Ở fold 1, chấm trên class đã thấy lợi 22–25 điểm; chỉ khác 2.8 điểm giữa hai fold khi class chưa thấy.

### 3.2 Leakage một phần: train trên cả 12 class (D-21, 20 epoch)

| | Chia class chuẩn | Class test đã thấy | Tăng | Paper |
| :--- | ---: | ---: | ---: | ---: |
| Baseline | 49.08 | 63.54 | +14.46 | 82.72 |
| Full | 57.15 | 69.24 | +12.09 | 88.53 |
| Full − baseline | +8.07 | +5.70 | | +5.81 |

Đây là leak một phần (mỗi class test chỉ có 1.600 slot thay vì 8.000; chưa hội tụ), không phải cận trên.

### 3.3 Điều B không nói được

* Không chứng minh paper đã làm như vậy: code tác giả chưa công bố. Còn lại 5–8 điểm chưa tách (seed, random600 so với
  fixed100, chọn checkpoint).
* Số B không dùng làm số cải tiến. Mọi so sánh của phase 16 chỉ dùng chấm đúng giao thức (D-22).

---

## 4. Ngoài bảng của paper: phase 16 (tham khảo, không thuộc A hay B)

Các số này là fold S1, không so trực tiếp với A và B.

| Mốc | mIoU | Ghi chú |
| :--- | ---: | :--- |
| E1 (head VIP-Seg trong pipeline của mình), `last` / `best` | 73.20 / 75.05 | dựa trên shortcut vị trí (D-35, D-37): không dùng để so sánh |
| VIP-Seg phát hành, best-of-validation | 75.36 | |
| CR (head sạch, thứ tự query ngẫu nhiên), `last` | 54.84 | base của mọi arm sau D-37 (`results/phase16_d37/`) |
| CR + luật inference U + both | 57.63 | D-39 |
| CR + label propagation (D-40) | 58.55 | D-40 |

Các thí nghiệm D-19…D-34 chạy trước khi phát hiện shortcut; chỉ dùng được như cơ chế. D-38…D-49 chạy trên base sạch CR.
Mục tiêu 77 ở kế hoạch hiện tại (mục tiêu của "base") là con số này, đo trên head sạch, chứ không phải số của phần B.

---

## 5. Còn thiếu gì, và phải làm gì

Đơn vị việc: **một lượt train** cho ra một ô của cả A và B (B chỉ thêm một lần chấm, không train).

| Bảng | Việc còn thiếu | Số lượt train | Cho phần |
| :--- | :--- | ---: | :--- |
| Table 1 | Viết bảng định tính (nguồn: paper) | 0 | không áp dụng |
| Table 4 | Train S1 cho 4 hàng (LMA, gate, cascade, full) | 4 | A và B |
| Table 4 | Chấm 3 checkpoint S0 (LMA, gate, cascade) trên fold 1 | 0 (chỉ chấm) | B |
| Table 4 | Chấm 4 checkpoint S1 mới trên fold 0 | 0 | B |
| Table 5 | T = 2, 3, 5, 6 trên S0 và S1 (T = 1 và T = 4 trùng hàng gate và full của Table 4) | 8 | A, rồi B |
| Table 2 (text) | 2-way 5-shot, 3-way 1-shot, 3-way 5-shot, mỗi ô 2 fold | 6 | A và B |
| Table 2 (image, audio) | Cả 4 cài đặt × 2 fold × 2 modality | 16 | A và B |
| Table 3 (ScanNet) | Lấy dữ liệu, tiền xử lý, rồi 4 cài đặt × 2 fold × 3 modality | 24 | A và B |
| Table 6 | Sửa tool để đo FLOPs của full model | 0 | A |

Tổng số lượt train mới trên S3DIS: 4 + 8 + 6 + 16 = **34**; ScanNet thêm **24**. Thời gian mỗi lượt: chưa đo cho các cấu hình
5-shot và 3-way. Trước đây ước tính khoảng 2.5 h cho một cặp trên 3090 (ước tính, chưa đo).

### 5.1 Các điều kiện tiên quyết

1. **Image và audio (D-47).** Đã commit (`2636a67`): `models/clip_image.py`, `models/clip_audio.py`,
   `preprocess/fetch_modality_images.py`, `tests/test_modalities.py`, bảng hàng dựng bằng `experiments/d47_build.py`
   (`results/phase16_d47/`). Hàng audio: 10 trên 14 transcript đúng; 4 lỗi gồm background, ceiling, wall, clutter, và
   hàng audio của wall và clutter gần hàng background nhất, nên với hai class đó audio mang sai class. Hàng image: 12 trên 14
   gần hàng text của chính nó. Điều kiện còn lại trên VM: repo trên VM đang ở commit `42aa7da` (chưa có D-47) và
   `datasets/modality/images` chưa có; 70 ảnh nằm ở `/workspace/d47src/datasets/modality`, cần chép sang (ảnh không nằm
   trong git).
   **Script:** `experiments/run_table2.sh` (`smoke`, `list`, `full`, `SUBSET=text|image|audio`). Nó dùng hàng đợi
   `experiments/phase14.py`, ưu tiên mới P5 (16 lượt image và audio) và cờ `--only`; text gồm `full_S1_N2K1` và P3 (7 lượt).
   Mỗi ô là một lượt train riêng; script từ chối chạy khi đang có `train.py` hoặc `eval.py` khác (`SHARE_GPU=1` để chia
   GPU). Chưa chạy trên GPU; `smoke` chạy test, dựng lại hàng D-47 và dry-run train ba cấu hình.
2. **ScanNet.** `datasets/` chỉ có S3DIS. Cần tải ScanNet (phải đồng ý điều khoản), chạy
   `preprocess/collect_scannet_data.py` và `room2blocks.py`, thêm `meta/scannet_classnames.txt` và
   `scannetv2-labels.combined.tsv` (spec 04 §2). Lịch train ScanNet 30 epoch × 800 episode đã có trong code (D-12), chưa
   từng chạy.
3. **Số của paper để so sánh.** `experiments/summarize.py` đã ghi Table 2 (text, 4 cài đặt) và Tables 4, 5. Chưa có trong
   repo: các hàng image và audio của Table 2, S0 và S1 riêng của Table 3 (spec 05 §5 chỉ có Avg), và các hàng phương pháp
   khác. Cần chép từ file PDF của paper (không có trong repo).
4. **Checkpoint của Table 4.** Nằm trên VM (`/workspace/CascadeProto`), không có trong máy này. Đã đối chiếu 2026-09-29 bằng
   cách đọc `config` và `args` trong từng file `.pt` (chỉ đọc, CPU) cùng với `eval_*_fixed100.json`. **Tên thư mục
   trùng nhau, dễ nhầm** (ví dụ `s3dis_S0_N2_K1_point_T0` có ở 5 thư mục `log_*`, với cấu hình khác nhau):

   | Hàng | Thư mục trên VM | mIoU đã ghi (best / last) |
   | :--- | :--- | :--- |
   | Baseline S0 | `log_phase14/s3dis_S0_N2_K1_point_T0` (`l2norm_point_proto=false`) | 49.08 / 49.07 |
   | Baseline + L2 (aside) | `log_phase15_bl2/s3dis_S0_N2_K1_point_T0` (`l2norm_point_proto=true`) | 52.44 / 50.19 |
   | + LMA | `log_phase15_t4rows/s3dis_S0_N2_K1_text_T0` | 49.65 / 48.32 |
   | + Entropy Gate (T = 1) | `log_phase15_t4rows/s3dis_S0_N2_K1_text_T1` | 56.72 / 56.72 |
   | + Cascade (T = 4) | `log_phase15_t4rows/s3dis_S0_N2_K1_text_T4_noadrm` | 56.55 / 56.03 |
   | + ADRM (full) | `log_phase14/s3dis_S0_N2_K1_text_T4` | 57.15 / 56.70 |
   | Baseline S1 | `log_s1/s3dis_S1_N2_K1_point_T0` | 51.91 (best) |
   | Full, class-leak 20 epoch (D-21) | `log_leak/s3dis_S0_N2_K1_text_T4_leak` | không phải hàng Table 4 |

   Các số phần B trong `results/seen/*.json` trỏ đúng vào các checkpoint này (S0 baseline 77.32 = `log_phase14/…point_T0/last.pt`
   trên fold 1; S1 baseline 71.58 = `log_s1/…/last.pt` trên fold 0; full 81.28 = `log_phase14/…text_T4/last.pt`). Các thư mục
   `log_cascadeproto`, `log_b1`, `log_chk_a`, `log_chk_b` cũng có `point_T0` nhưng là run khác (b1 là lịch của VIP-Seg, 49.01):
   không dùng cho Table 4.
5. **Queue hiện có.** `experiments/phase14.py` liệt kê 42 lượt (toàn S3DIS: 26 text + 16 image và audio). Lượt image và audio đã có (P5, 16 lượt). Chưa có lượt cho ScanNet, và
   cho chấm phần B trừ `run_seen.sh`. Chưa kiểm tra lượt nào của 26 đã xong ngoài những gì báo cáo này ghi.
6. **Quy tắc của repo (AGENTS §2.4, D-22).** Bảng phần B phải mang nhãn "chẩn đoán, không phải kết quả few-shot". Nếu muốn
   thêm nó thành mục chính thức, cần ghi thành decision mới (D-50) trước khi chạy.

### 5.2 Thứ tự đề nghị

1. Table 4: train 4 lượt S1 và chấm B (cái rẻ nhất, hoàn thành hai bảng Table 4 A và B).
2. Table 5: 8 lượt.
3. Table 6: sửa FLOPs.
4. Table 2 text: 6 lượt.
5. Commit và kiểm chứng D-47, rồi Table 2 image và audio: 16 lượt.
6. Table 3: chỉ sau khi dữ liệu ScanNet sẵn sàng.

Với paper cải tiến, các bảng này sẽ so **phương pháp của mình trên base sạch** với VIP-Seg và CR theo cùng lưới. Theo kế
hoạch hiện tại, chạy đủ lưới chỉ nên làm sau khi base sạch đạt khoảng 70–77, để không phải chạy lại. Phần A và B của
paper là tái lập, và không phụ thuộc vào điều đó.
