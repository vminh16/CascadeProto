# Trần của cả lĩnh vực point cloud segmentation, và chỗ có thể có đột phá

Ngày 2026-09-29. Ghi chú nghiên cứu, **không phải decision, không phải spec**. Đây là phần tiếp của
`2026-09-29_first_principles_layer_audit_vi.md`: bỏ ràng buộc của paper CascadeProto và của guardrail, rồi hỏi trần của
cả bài toán 3D segmentation (không riêng few-shot) nằm ở đâu, và đột phá có thể đến từ đâu.

Nguồn: ba cuộc khảo sát độc lập (trần và quỹ đạo SOTA; các lần chuyển paradigm 2023–2026; lý thuyết từ gốc), đối chiếu
chéo với nhau và với các số đã đo trong repo.

Nhãn:
* **[readme]**: đọc trực tiếp trong README của repo GitHub. Mình đã tự đọc lại README của Pointcept, DITR và LAM3C.
* **[snippet]**: đoạn trích từ công cụ tìm kiếm. arXiv, CVF, OpenReview và Hugging Face bị chặn khỏi sandbox, nên **phải
  kiểm lại trước khi trích dẫn**.
* **[recall]**: trí nhớ, chưa kiểm.
* **[E]**: kết quả toán, kiểm được từng dòng.
* **[repo]**: đo trong repo này.
* **[C]**: phỏng đoán hoặc ước lượng.

---

## 0. Kết luận

1. **Bài toán có giám sát đầy đủ với tập class đóng đã gần bão hoà về kiến trúc.** Từ 2022, backbone train từ đầu chỉ tăng
   khoảng 0.4–0.8 điểm mỗi năm. Các bước tăng sau đó đều đến từ **dữ liệu và prior bên ngoài**: đa dataset, self-supervised
   pretraining, đặc trưng 2D. Khoảng trống còn lại trên ScanNet-20 và S3DIS chủ yếu do nhãn và định nghĩa class.
2. **"Khoảnh khắc attention + scale" của 3D đã xảy ra, nhưng không nằm ở toán tử point cloud.**
   * Về kiến trúc: serialization của PTv3 (đường cong lấp không gian) biến tập điểm thành chuỗi để attention scale được.
     Nó chủ yếu mua compute, đáng khoảng +2 điểm.
   * Về hình học: VGGT và π3 tái tạo 3D từ ảnh hoặc video bằng transformer thuần [snippet].
   * Về ngữ nghĩa: 3D **mượn scale của 2D** qua phép tương ứng pixel↔điểm. Linear probe trên đặc trưng đóng băng đi từ
     21.8 (MSC) lên 72.5 (Sonata) rồi 77.3 (Concerto) trên ScanNet [snippet]: đặc trưng đóng băng cộng một lớp tuyến tính đã
     bằng PTv3 train đầy đủ (77.6 [readme]).
3. **Point cloud không nên là modality chính.** Nó là sản phẩm phụ có mất mát của khoảng 1.650 khung RGB-D mỗi scan. Cách
   đặt bài toán có đòn bẩy lớn nhất coi 3D là *chỉ mục hình học và ràng buộc nhất quán* cho đặc trưng 2D và video.
4. **Toán học không có "viên đạn bạc".** Có ba chỗ toán giúp được, mỗi chỗ vài điểm: giải mã tối ưu cho metric IoU, mô
   hình nhiễu khi hợp nhất nhiều view, và đầu ra phân cấp (ultrametric) gọi tên theo ngữ cảnh.
5. **Với bài toán của repo:** thứ đứng ở vị trí ràng buộc chính trong few-shot (nguồn thông tin của biểu diễn) chính là
   thứ cả lĩnh vực đã giải bằng dữ liệu. Đã có một đối chứng sạch để thử: **LAM3C**, chỉ pretrain trên point cloud tái
   tạo từ video phòng lấy trên web, **không có scan thật, không có S3DIS**. Linear probe của nó trên S3DIS Area 5 đạt
   **69.5** [readme].

---

## 1. Trần hiện tại của bài toán có giám sát đầy đủ

| Benchmark | ≤ 2021 | 2024 (backbone từ đầu) | Tốt nhất 2025–26 | Trần ước lượng [C] |
| :--- | :--- | :--- | :--- | :--- |
| S3DIS Area 5 | KPConv 67.1, PTv1 70.4 [snippet] | PTv3 73.6; PTv3 + PPT 75.4 [readme] | DITR 74.1, D-DITR 75.0 [readme]; Sonata 76.0, Concerto 77.4, Utonia 78.1 [snippet] | 85–88 |
| S3DIS 6-fold | KPConv 70.6, PTv1 73.5 [snippet] | PTv3 77.7; + PPT 80.8 [snippet] | Sonata 82.3 [snippet] | 88–90 |
| ScanNet val | MinkUNet 72.2 [recall] | PTv3 77.6; + PPT 78.5 [readme] | DITR 80.5 [readme]; Concerto 80.7, Utonia 81.1 [snippet] | 88–92 |
| ScanNet200 val | — | PTv3 35.3 [readme]; ODIN 40.5 [readme] | DITR 41.2 (DINOv2) / 42.3 (DINOv3) [readme] | 65–75 |
| ScanNet++ (top-1 / top-3) | — | PTv3 48.8 / 73.3 [snippet] | Concerto 50.7 [snippet] | khoảng 70 |
| nuScenes val | — | PTv3 80.3 [snippet] | DITR (DINOv2-g) 84.2 [readme] | 88–90 |

**Tốc độ.** ScanNet test đi từ 78.1 (Mix3D, 2021) lên 79.7 (DITR, 2025), khoảng +0.4 điểm mỗi năm. S3DIS Area 5 với
backbone train từ đầu tăng khoảng +0.8 mỗi năm từ 2022 [C, từ các số trên]. Chỉ ScanNet200 và ScanNet++ còn đi nhanh, vì
đuôi dài của phân bố class.

**Class nào đang giới hạn điểm.**
* **S3DIS Area 5:** beam ≈ **0.0** với PTv1 và Stratified Transformer [snippet]. Mức 0 này đến từ việc Area 5 là một toà
  nhà khác và block quá nhỏ để thấy chỗ beam nối với tường. Chỉ riêng beam đã chặn trần mIoU ở 12/13 ≈ 92.3.
* Tiếp theo: column 38–46, window 60–63, clutter 59–64, board.
* **ScanNet200:** nhóm head đứng yên khoảng 55 suốt ba năm, nhóm tail đi từ 13 lên 20 [snippet]. Thêm 4.5k scene gán
  nhãn tự động (ARKit LabelMaker) tăng tail +5.5 [snippet]. Đây là giới hạn của dữ liệu, không phải của mô hình.

**Few-shot so với có giám sát đầy đủ trên cùng class** (PTv1, Area 5, trung bình theo fold [E, tính từ bảng per-class]):
* Fold S1 (door, floor, sofa, table, wall, window): **81.0**.
* Fold S0 (beam, board, bookcase, ceiling, chair, column): **61.8**.

Vậy "80+ few-shot trên S1" bằng đúng mức có giám sát đầy đủ trên chính các class đó. Protocol khác nhau (2-way, có mặt
được bảo đảm, chấm trên mọi area), nên đây chỉ là phép kiểm tra hợp lý, không phải chứng minh. Nhưng nó nhất quán với việc
số 73–75 của E1 và VIP-Seg phụ thuộc vào shortcut [repo, D-35, D-37].

## 2. Mô hình trần: vì sao kiến trúc không còn là đòn bẩy

**(a) Nhiễu nhãn [E].** Giả sử một tỉ lệ ε điểm của class c mang nhãn khác, và một lượng bằng thế từ class khác mang nhãn
c. Khi đó ngay cả dự đoán đúng sự thật cũng chỉ đạt IoU = (1 − ε)/(1 + ε): 0.90 tại ε = 0.05, và 0.82 tại ε = 0.10. Có
73 % scene val của ScanNet mang lỗi nhãn mức instance mà hai người kiểm cùng xác nhận [snippet]. Nhãn ScanNet++ mơ hồ đến
mức top-1 là 48.8 còn top-3 là 73.3 [snippet].

**(b) Vùng biên [C].** Tỉ lệ bề mặt nằm trong khoảng δ quanh biên là khoảng Pδ/A. Tại δ = 2 cm: tường 4 × 2.5 m là 2.6 %;
mặt ghế 0.45 m là 18 %; bức tranh 0.4 × 0.3 m là 23 %. Vật nhỏ mất 10–20 điểm IoU chỉ vì dải biên này.

**(c) Khuếch đại theo tỉ lệ điểm [E].** Gọi r là recall, f là tỉ lệ false positive trên các điểm còn lại, π là tỉ lệ điểm
của class. Khi đó:

IoU_c = r / (1 + f(1 − π)/π)

* Tường (π 0.3, r 0.9, f 0.02) đạt 0.86.
* Class đuôi (π 5·10⁻⁴, r 0.6, f 5·10⁻⁴) chỉ đạt 0.30.

Class đuôi là một bài toán false positive ở thang 10⁻⁴.

**(d) Phụ thuộc vào độ đo [E].** IoU theo điểm = ∫_{A∩B} ρ dσ / ∫_{A∪B} ρ dσ, nên phụ thuộc mật độ lấy mẫu ρ. Trên mesh
gần đều của ScanNet thì ảnh hưởng nhỏ. Trên protocol few-shot kế thừa, khoảng 22.6 điểm là do mật độ [repo].

**Mô hình tổng hợp [C]:**

mIoU ≈ (1/C) Σ_c κ(ε_c) · r_c / (1 + f_c/π_c), với κ(ε) = (1−ε)/(1+ε).

Trong đó:
* 1 − r_c = m_biên + a(R) · N_c^(−α)
* f_c = g_c · b(R) · N_c^(−α)
* N_c là số instance độc lập của class c.
* R là chất lượng biểu diễn; pretraining làm a và b nhỏ đi.
* α khoảng 0.3–0.5, chưa ai đo trong 3D.

Hệ quả:
* Class head bị chặn bởi κ (nhiễu nhãn) và vùng biên.
* Class đuôi bị chặn bởi f/π, và chỉ R hoặc N kéo được nó.
* **Kiến trúc chỉ tác động qua a và b; không kiến trúc nào vượt được κ hay π.**

**Số mẫu hiệu dụng [C].** ScanNet có khoảng 1.8·10⁸ điểm có nhãn. Nhưng các điểm trong một instance gần như trùng lặp:
n_eff = n/(1 + (m − 1)ρ) ≈ 3.8·10⁴, xấp xỉ số instance, tức khoảng 10³ mẫu mỗi class trên ScanNet-20. Head class không
thiếu dữ liệu; đuôi thì thiếu. Kho pretrain của 3D (Sonata 140k, Utonia khoảng 250k point cloud [snippet]) kém 2D
(LVD-1689M của DINOv3) khoảng 10³–10⁴ lần. **3D thiếu độ đa dạng, không thiếu điểm.**

## 3. Các cú chuyển paradigm 2023–2026, xếp theo bằng chứng

| Hướng | Bằng chứng | Scale | Chi phí với 1 GPU |
| :--- | :--- | :--- | :--- |
| **A. Đặc trưng 2D foundation nâng hoặc chưng cất vào 3D** | DITR: ScanNet200 35.3 → 41.2 / 42.3 [readme]. Concerto: linear probe +4.8 so với Sonata. Lexicon3D: DINOv2 lifted đạt 62.8 so với Swin3D 35.2 khi probe trên ScanNet [snippet] | Theo scale 2D (10⁹ ảnh) | Thấp nếu đóng băng |
| **B. Video → 3D (VGGT, π3) làm động cơ dữ liệu** | LAM3C: 49k scene từ video, không scan thật; S3DIS A5 LP / FT 69.5 / 75.5, ScanNet 69.5 / 79.5 [readme], ngang Sonata khi fine-tune | Lớn nhất (hàng chục triệu video); chưa ai đo đường cong quá 49k scene | Tái tạo rẻ; pretrain lớn thì đắt |
| C. Ghép các mô hình promptable đóng băng (SAM3, VLM) ở mức điểm số | Pipeline không train đóng khoảng 33 % khoảng cách tới SOTA đã train; giữ nguyên phân bố điểm số thay vì argmax: +6.4 HM trên ScanNet200 [snippet] | Cao | Thấp |
| D. 3D SSL thuần ở quy mô lớn (Sonata, Utonia) | Frozen mạnh, fine-tune gần phẳng: Utonia dùng khoảng 6× dữ liệu Concerto mà chỉ +0.4 [snippet] | Chặn ở khoảng 250k cloud | Pretrain không khả thi |
| E. LLM 3D, video LLM | Mạnh cho QA và grounding; không có segmentation dày dạng few-shot | Cao | Cao |
| F. Head hoặc kiến trúc mask mới | Bão hoà, +1–2 | — | Thấp |

**Chưa có mô hình nào cho thấy few-shot hay in-context segmentation mức cảnh xuất hiện một cách tự nhiên.** PIC / PIC++
chỉ làm ở mức vật thể và bộ phận [snippet]. Đây là một khoảng trống thật.

Ba lưu ý về protocol:
* ODIN cho thấy Mask3D rơi từ 55.2 xuống 43.9 mAP khi dùng point cloud từ cảm biến thay vì lấy mẫu từ mesh [snippet]:
  xếp hạng phụ thuộc cách lấy mẫu.
* Weight của Sonata, Concerto, Utonia và LAM3C đều là CC-BY-NC [readme].
* Encoder pretrain trên point cloud dày, nên có thể suy giảm trên block 2048 điểm (Invaria [snippet]). Phải đo, không
  giả định.

## 4. Toán học: ở đâu giúp thật, ở đâu là ngõ cụt

Quy tắc [C]: một cách đặt lại bài toán chỉ có lợi khi nó **mang thông tin mới vào** (prior 2D, view, task), hoặc **khớp
đơn vị quyết định với metric**. Nó không có lợi khi chỉ sắp xếp lại cùng một bộ đặc trưng. Toàn bộ phase 16 là bằng chứng
cho vế sau: khoảng 17 biến thể head và inference bão hoà ở +2.8 [repo].

| Ý tưởng | Toán cốt lõi | Bắt vào nút thắt nào | Test rẻ nhất để bác bỏ | Xác suất có ích [C] |
| :--- | :--- | :--- | :--- | :--- |
| **I1. Động cơ dữ liệu có đo số mũ scaling** | Fit e(N) = A·N^(−α) theo từng nhóm tần suất class, có và không có dữ liệu từ video hoặc gán nhãn tự động | N_c của class đuôi | Train SpUNet hoặc PTv3 nhỏ trên {10, 25, 50, 100} % ScanNet ± 4.5k scene LabelMaker, cùng số bước; chết nếu đuôi tăng < 1 điểm mỗi lần nhân đôi | 0.65 (đã được chứng minh, ít mới) |
| **I2. Hợp nhất nhiều view với mô hình nhiễu phụ thuộc view** | Mỗi view là một "người gán nhãn" có ma trận nhầm lẫn Π_v(θ); θ gồm độ sâu, góc tới, khoảng cách tới biên mask 2D. Suy luận p(y_i \| ŷ_iv) bằng EM kiểu Dawid–Skene cộng prior trơn 3D. Phương sai sau hợp nhất là σ²(ρ + (1−ρ)/V) [E]: lỗi hệ thống giữa các view không tự triệt tiêu khi lấy trung bình | Lỗi hệ thống của prior 2D | 100 scene val ScanNet, một segmenter 2D đóng băng; so vote đa số, trung bình xác suất và DS-EM; chết nếu hơn < 1 mIoU (khoảng 3 GPU-h) | 0.3 |
| **I3. Nhóm theo ultrametric, gọi tên in-context** | Ultrametric trội dưới d_U(i, j) = min qua mọi đường của cạnh lớn nhất (single linkage trên MST), khả vi hầu khắp (Chierchia–Perret). Một cây chứa mọi mức độ chi tiết. Gọi tên = chọn nút của cây query giống nút của cây support nhất: O(n) ứng viên thay vì 2ⁿ, và quyết định theo nút làm giảm f | Lỗi gọi tên cả vùng (homophily 0.94; mean của chính query đưa 55 lên 81 [repo]); nhãn lồng nhau | Episode S3DIS S1 trên CPU, cây dựng từ feature CR hoặc Sonata cộng normal và màu; chết nếu oracle nút tốt nhất < 75, hoặc hơn prototype điểm < 1 điểm trên cả fixed100 lẫn leak-free | 0.35 (few-shot, open-vocab); 0.1 (tập class đóng) |
| **I4. Giải mã tối ưu cho metric** | Với IoU cấp corpus, thêm điểm i vào class c làm tăng IoU kỳ vọng khi và chỉ khi p_ic > J*/(1 + J*) = F*/2 [E]. Hiệu chỉnh nhiệt độ, fit ngưỡng từng class, thêm xác suất class có mặt trong cảnh | Argmax không tối ưu IoU khi π nhỏ (§2c) | Checkpoint PTv3 ScanNet200 phát hành, chia val làm đôi A/B; chết nếu gain trên B < 0.5 | 0.5 cho +1–3 trên benchmark nặng đuôi; khoảng 0 trên ScanNet-20 |
| I5a. Bất biến mật độ theo lý thuyết độ đo | Đặc trưng ∫K(x, y)h(y)dσ(y) với trọng số 1/ρ̂; loss và metric theo diện tích | Protocol rò mật độ | Chấm lại dự đoán cũ với trọng số Voronoi; bỏ nếu không thay đổi xếp hạng | 0.1 cho độ chính xác; 0.5 là nó đổi kết luận trên S3DIS và few-shot |
| I5b. Khám phá bộ phận theo MDL | L(mô hình) + L(dữ liệu \| mô hình) | — | Đường purity theo K so với VCCS và SPT | 0.05 (ngõ cụt: ngữ nghĩa không phải mã ngắn nhất của hình học) |

**Kết luận.** Đóng góp kỳ vọng, lớn nhất trước: **dữ liệu ≫ cách đặt bài toán > kiến trúc > toán mới** [C]. "Khoảnh khắc
transformer" của 3D, nếu có, sẽ gồm:
* dữ liệu là video;
* mục tiêu tự do là nhất quán phép chiếu giữa các view (pointmap kiểu VGGT, embedding chung 2D–3D kiểu Concerto);
* đầu ra là mask hoặc cây phân cấp, gọi tên theo ngữ cảnh hoặc bằng ngôn ngữ.

Các mảnh đều đã có; lợi ích sẽ thấy trên đuôi, open-vocabulary, few-shot và chuyển miền, không phải trên ScanNet-20.

## 5. Với một người nghiên cứu, một GPU, 6–12 tháng: chương trình đề xuất

Đích: **few-shot, in-context 3D segmentation mức cảnh trên đặc trưng foundation đóng băng**, đánh giá trên protocol sạch.
Đây là chỗ bằng chứng mạnh nhất (§3 A, B), khoảng trống rõ nhất (chưa có in-context mức cảnh), và vừa với ngân sách.

1. **P1: đặc trưng tổng quát, có đối chứng sạch.** Chạy trên các episode fixed100 S0/S1, luật cosine prototype, không
   train, so bốn nguồn đặc trưng:
   * **LAM3C**: không thấy S3DIS, nên là đối chứng sạch;
   * Sonata: đã thấy S3DIS không nhãn;
   * Concerto;
   * CR: base sạch, U 55.74.

   Báo cáo trên fixed100, leak-free và protocol đảo thứ tự query. Nếu LAM3C cao hơn CR nhiều, thông tin đến từ biểu diễn
   tổng quát, không phải từ việc đã thấy cảnh test. Cần **D-50**, vì guardrail 1 hiện cấm weight pretrain; kết quả để
   trong bảng riêng.
2. **P2: dịch chuyển mật độ.** Probe LAM3C và Sonata trên block 2048 / 8192 / mật độ đầy đủ. Nếu mất nhiều điểm, chưng cất
   từ teacher dày sang student 2048 điểm là một đóng góp riêng.
3. **P3: I3.** Cây ultrametric trên đặc trưng tốt nhất của P1, cộng normal và toạ độ mét, gọi tên theo nút. Đây là phiên
   bản đúng chỗ của "cascade purification".
4. **P4: entropy đặt đúng chỗ.** So hợp nhất bằng argmax, bằng tích các phân bố đã hiệu chỉnh, và bằng cổng entropy,
   trên logit prototype 3D và logit nâng từ 2D. Nếu cổng entropy không thắng tích phân bố ít nhất 0.5 (3 seed), ý tưởng
   entropy gate của CascadeProto không đóng góp gì.
5. **I4** dùng như một lớp giải mã chung cho mọi arm, trên các class có π nhỏ.

Kỳ vọng [C, chưa đo]: nếu P1 cho LAM3C hoặc Concerto với U ≥ 65 trên giao thức chuẩn sạch, và cao hơn CR ≥ 10 trên
leak-free, hướng này thành paper phương pháp. Nếu không, đóng góp là bài phân tích benchmark cộng mô hình trần ở §2.

## 6. Điều không nên kết luận

* Mọi số [snippet] chưa đối chiếu với bảng trong paper.
* Các dải "trần ước lượng" ở §1 là phán đoán, không phải phép đo đồng thuận giữa người gán nhãn; chưa benchmark 3D nào
  công bố con số đó.
* Chưa có phép đo nào cho thấy đặc trưng tổng quát thắng trên protocol few-shot của repo; đó là việc của P1.
