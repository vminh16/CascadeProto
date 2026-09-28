# Tái lập CascadeProto: bảng nào đã có, bảng nào chưa

Ngày 2026-09-28. Bản rút gọn của `2026-09-21_reproduction_report.md`, sắp theo bảng của paper. Số liệu là mIoU %
trên S3DIS, 2-way 1-shot, giao thức `fixed100` (1.500 episode), một seed, trừ khi ghi khác.

## 0. Paper có 6 bảng

| Bảng | Nội dung | Loại | Mình đã có bảng tương tự? |
| :--- | :--- | :--- | :--- |
| Table 1 | So sánh định tính các phương pháp: pre-training, modality (Text/Audio/Image/Point), backbone | không có số | chưa làm (chỉ cần viết, không cần chạy) |
| Table 2 | S3DIS: 2/3-way × 1/5-shot × S0/S1/Avg, 3 biến thể modality | kết quả chính | **một phần**: chỉ 2-way 1-shot, text |
| Table 3 | ScanNet: cùng lưới như Table 2 | kết quả chính | **chưa** (ScanNet chưa chạy) |
| Table 4 | Ablation từng thành phần (LMA, Entropy Gate, Cascade, ADRM), S0/S1/Avg | ablation | **có**: đủ 5 hàng trên S0; hàng baseline có cả S1 |
| Table 5 | Độ sâu cascade T = 1…6 | ablation | **một phần**: T = 1 và T = 4 |
| Table 6 | Params / FLOPs / mIoU trên S0 | độ phức tạp | **một phần**: params đã đếm, FLOPs chỉ có baseline |

Hiện có số liệu tương ứng cho 4 trên 6 bảng (2, 4, 5, 6), trong đó chỉ Table 4 là đủ. Muốn đủ bộ, việc lớn nhất là lưới
của Table 2 và Table 3 (mục 4).

---

## 1. Tái lập trước khi phát hiện leakage (phase 14–15)

### 1.1 Kiểm tra pipeline (điều kiện để tin mọi số phía sau)

| Kiểm tra | Của mình | Tham chiếu |
| :--- | ---: | ---: |
| Checkpoint S0 của VIP-Seg, chấm bằng `eval.py`, data và metric của mình | 71.97 | 72.20 (paper Tab.6 / log VIP-Seg) |
| Model VIP-Seg train bằng loop của mình, 2.400 episode | 69.48 | 68.9 (script gốc của VIP-Seg, cùng thời điểm) |

→ Data, sampler, loss, optimizer và metric đều khớp VIP-Seg. Khoảng cách với paper không đến từ pipeline.

### 1.2 Bảng tương ứng Table 4 (ablation thành phần, S0, lịch train đầy đủ của paper)

| Cấu hình | Của mình (best) | Paper S0 | Chênh |
| :--- | ---: | ---: | ---: |
| Baseline | 49.08 | 82.72 | −33.6 |
| + LMA | 49.65 | 83.98 | −34.3 |
| + Entropy Gate (T = 1) | 56.72 | 85.34 | −28.6 |
| + Cascade (T = 4) | 56.55 | 87.89 | −31.3 |
| + ADRM (full) | 57.15 | 88.53 | −31.4 |

Phần tăng thêm từng bước:

| Bước | Của mình | Paper | Kết luận |
| :--- | ---: | ---: | :--- |
| Tổng: baseline → full | +8.07 | +5.81 | **tái lập** (còn lớn hơn) |
| + ADRM | +0.60 | +0.64 | **tái lập** |
| + LMA | +0.57 | +1.26 | cùng dấu, nhỏ hơn một nửa |
| + Entropy Gate (1 stage) | +7.07 | +1.36 | lớn gấp 5 lần |
| + Cascade (1 → 4 stage) | −0.17 | +2.55 | **không tái lập** |

Baseline trên S1: 51.91 (paper 79.83).

### 1.3 Bảng tương ứng Table 5 (độ sâu T, S0)

| T | 1 | 2 | 3 | 4 | 5 | 6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Của mình | 56.72 | — | — | 57.15 | — | — |
| Paper | 85.21 | 86.43 | 87.78 | 88.53 | 88.51 | 88.37 |

Paper tự mâu thuẫn: Table 4 hàng 3 và Table 5 T = 1 là cùng một cấu hình nhưng in hai số khác nhau (85.34 và 85.21).

### 1.4 Bảng tương ứng Table 2 (chỉ ô 2-way 1-shot, text)

| Phương pháp | S0 | S1 | Avg |
| :--- | ---: | ---: | ---: |
| VIP-Seg (checkpoint phát hành, chấm lại) | 71.97 | — | — |
| VIP-Seg (paper) | 72.20 | 76.09 | 74.15 |
| CascadeProto (Text), của mình | 57.15 | — | — |
| CascadeProto (Text), paper | 88.53 | 84.53 | 86.53 |

Các ô 5-shot, 3-way, Image và Audio chưa chạy. Image và Audio chưa được cài đặt.

### 1.5 Bảng tương ứng Table 6 (độ phức tạp)

| | Params | FLOPs (fvcore) |
| :--- | ---: | ---: |
| Baseline của mình (encoder 2.37M + head 0.20M) | 2.58M | 19.4 G* |
| Full của mình (đếm theo thiết kế: + LMA, 4 EPPM, ADRM = 0.47M) | ≈ 3.04M | chưa đo (tool lỗi) |
| Paper: VIP-Seg / CascadeProto | 2.76M / 2.88M | 8.48 / 8.86 G |

\* fvcore không đếm kernel CUDA tuỳ biến (Mamba, FPS), nên FLOPs chỉ là cận dưới và không so trực tiếp được với paper.
Riêng các module thêm vào đã cần 0.47M, nhiều hơn mức ≈ 0.12–0.31M mà paper ngụ ý.

### 1.6 Các chỗ mơ hồ đã chọn bằng đo (harness ngắn, 3 seed)

| Chỗ mơ hồ | Kết quả |
| :--- | :--- |
| D-02: gate đặt lên prototype hay feature | không khác biệt |
| D-10: có chia 1/√D ở Eq.23 không | không chia; chia làm mất 5.5 điểm |
| D-18: scale trong softmax Eq.14 | giữ như bản in; chuẩn hoá làm mất 0.9 |

**Tóm tắt phần 1.** Pipeline đúng. Phần tăng tổng và ADRM tái lập được; cascade depth thì không. Mức tuyệt đối thấp hơn
paper khoảng 30 điểm, và không có quyết định nào của mình trên đường đi của baseline giải thích được khoảng này
(report §3.3).

---

## 2. Thử nghiệm leakage (class test đã được thấy lúc train)

### 2.1 Train trên cả 12 class (D-21, leak một phần, 20 epoch)

| | Chia class chuẩn | Class test đã thấy | Tăng | Paper |
| :--- | ---: | ---: | ---: | ---: |
| Baseline | 49.08 | 63.54 | +14.46 | 82.72 |
| Full | 57.15 | 69.24 | +12.09 | 88.53 |
| Full − baseline | +8.07 | +5.70 | | +5.81 |

### 2.2 Chấm trên class đã thấy (đổi fold: model S0 chấm trên class của S1 và ngược lại)

| Baseline | S0 | S1 | Avg | Cách paper (Avg) |
| :--- | ---: | ---: | ---: | ---: |
| Paper, Table 4 | 82.72 | 79.83 | 81.28 | — |
| Giao thức chuẩn | 49.08 | 51.91 | 50.50 | 30.8 |
| Chấm trên class đã thấy | 77.32 | 71.58 | 74.45 | **6.8** |

Full model chấm trên class đã thấy: 81.28 (paper 88.53). VIP-Seg phát hành: 79.13, so với 72.20 khi chấm đúng giao thức.

**Tóm tắt phần 2.** Chấm trên class đã thấy trong lúc train kéo mình về sát mức của paper (còn cách 5–8 điểm), và tái
hiện đúng thứ tự S0 > S1 của paper. Mọi phương pháp khác trong Table 2 đều có thứ tự ngược lại. Điều này **khớp** với
số của paper, nhưng **không chứng minh** paper đã làm như vậy, vì code của tác giả chưa công bố. Từ đây repo có
cơ chế chặn việc chấm trên class đã thấy (D-22).

---

## 3. Các thí nghiệm ngắn trước D-30 (đầu phase 16)

**Lưu ý:** các thí nghiệm dùng head của VIP-Seg (route B: D-25 PEM, D-26…D-29) chạy **trước khi phát hiện shortcut vị
trí** (D-36/D-37), nên chỉ dùng được như cơ chế, không dùng làm số so sánh.

| Decision | Thử gì | Kết quả |
| :--- | :--- | :--- |
| D-19 | Thêm một số hạng giữ kênh vào Eq.19 | chỉ phân tích tĩnh; không cải thiện (3.9 % → 4.2 % năng lượng phân lớp); không train |
| D-20 | Baseline dùng encoder đã train của VIP-Seg | 47.37 (54.97 khi L2): baseline không tăng lên 82 được nhờ encoder |
| D-23 | Một S′ chung cho cả episode ở Eq.13 | +0.33 (t = 0.65), không có ý nghĩa |
| D-24 | EPPM-S, stage rút gọn | 49.27, thua stage gốc 3.79 điểm và thua không dùng stage 6.75 điểm |
| D-25 | Thay stage bằng PEM của VIP-Seg | 68.52 (+15.47 so với stage gốc); có shortcut vị trí → chuyển sang route B |
| D-26 | EM tinh chỉnh prototype theo query | ≈ 0 (từ −1.21 đến +0.37); dùng nhãn thật thì +8 đến +22 |
| D-27 | Hiệu chỉnh background bằng base class | −0.03 đến −0.16, dừng |
| D-28 | EM có lọc theo base margin | lọc càng mạnh càng kém (từ +0.51 xuống +0.47 trên valid), dừng |
| D-29 | Distill theo hướng oracle lúc train | +0.06 [−0.26, +0.39], dừng |

**Tóm tắt phần 3.** Các cách sửa EPPM của paper đều không hiệu quả. Head của VIP-Seg tốt hơn nhiều nhưng dính shortcut vị
trí. Các luật lúc inference và distill không lấy lại được khoảng cách tới oracle. Những kết quả này dẫn tới base sạch CR
(D-37) và các probe P6–P10.

---

## 4. Để có đủ bảng tương tự paper

| Bảng | Còn thiếu | Khối lượng (ước tính, chưa đo) |
| :--- | :--- | :--- |
| Table 1 | Bảng định tính | viết tay, không cần GPU |
| Table 2 | 2-way 5-shot, 3-way 1/5-shot, cả S0 và S1 | 8 ô × mỗi phương pháp; khoảng 2.5 h cho mỗi cặp lượt train trên 3090 |
| Table 3 | ScanNet: tiền xử lý dữ liệu và cùng lưới | tiền xử lý, rồi 8 ô × mỗi phương pháp |
| Table 4 | S1 cho các hàng 2–5 | 4 lượt train |
| Table 5 | T = 2, 3, 5, 6 trên S0 và S1 | 8 lượt train |
| Table 6 | FLOPs của full model (sửa lỗi tool) | vài phút |

Với paper cải tiến, các bảng này sẽ so **phương pháp của mình trên base sạch** với VIP-Seg và CR theo cùng lưới. Theo
kế hoạch hiện tại, việc chạy đủ lưới chỉ nên làm sau khi base đạt khoảng 70–77, để không phải chạy lại.
