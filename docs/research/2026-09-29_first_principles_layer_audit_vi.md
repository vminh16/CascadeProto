# Kiểm tra từ gốc: sai ở tầng nào — dữ liệu, biểu diễn, phương pháp hay cách đặt bài toán

Ngày 2026-09-29. Ghi chú nghiên cứu, **không phải decision, không phải spec**: không dòng nào ở đây là chuẩn tắc. Mọi hướng
đi đề xuất ở §6 cần một decision (D-50 trở đi) trước khi viết code (AGENTS §2).

Nguồn: bốn cuộc kiểm tra độc lập (dữ liệu/benchmark, biểu diễn/backbone, phương pháp/cách đặt bài toán, sổ bằng chứng
D-19…D-49), cùng ba cuộc kiểm tra trước đó (toán/thống kê, code, tổng quan tài liệu), rồi đối chiếu chéo.

Nhãn: **[repo]** đo hoặc in trong repo này (kèm file); **[paper]** đọc trong nguồn gốc, hoặc qua ghi chú đã kiểm của repo;
**[snippet]** chỉ từ đoạn trích của công cụ tìm kiếm, vì arXiv, OpenReview, CVF và Semantic Scholar bị chặn khỏi sandbox,
**phải kiểm lại trước khi trích dẫn**; **[mine]** suy luận hoặc số học của ghi chú này.

Mọi số là S3DIS S1, 2-way 1-shot, `last.pt`, `fixed100`, trừ khi ghi khác.

---

## 0. Kết luận trong năm dòng

1. **Code đúng.** Mô hình bám paper khớp spec 02 theo từng phương trình, metric là `evaluate_metric` của VIP-Seg, và head
   sạch không còn shortcut vị trí. Khoảng 30 điểm tới paper là do giao thức (chấm trên class đã thấy), không phải bug.
2. **Head là ngõ cụt.** Bỏ shortcut đi thì head bằng prototype matching thuần: CR 54.84, U 55.74, luật support 55.02.
   Khoảng 17 thí nghiệm ở tầng head và inference bão hoà ở khoảng +2.8.
3. **Ràng buộc chặt nhất là nguồn thông tin của biểu diễn.** Chưa có thí nghiệm nào (0 trên ~30) thay đổi nó. Encoder
   chỉ học từ 6 class base cộng background, nên bị neural collapse: PR 5.56 ≈ C − 1. Nó không thu nhỏ được lệch
   instance giữa support và query; khoảng hai phần ba khoảng cách của U không giảm khi thêm shot.
4. **Cách đặt bài toán cũng sai.** Điểm chuẩn chủ yếu đo mật độ lấy mẫu (khoảng 22–27 điểm), vị trí query (khoảng 20 điểm
   ở head VIP-Seg), và vị trí theo chiều cao trong block đã chuẩn hoá. Nó không đo "nhận ra class mới trong một cảnh". Mục
   tiêu 80+ trên giao thức này chỉ đạt được qua các kênh đó.
5. **Hướng đột phá có bằng chứng:** tách bài toán thành *nhóm điểm* (không phụ thuộc class) và *gọi tên nhóm*
   (in-context), trên một biểu diễn **không** học từ 6 class. Đó là con đường 2D đã đi (PFENet → BAM → Matcher, GF-SAM,
   SINE). Trong 3D few-shot chưa ai làm, theo tìm kiếm có giới hạn. Ba test sàng lọc chỉ cần inference (§6) quyết định
   hướng này có đáng làm hay không trong vài ngày.

---

## 1. Bản chất bài toán, viết lại từ gốc

"Few-shot 3D semantic segmentation" thật là: cho **một** ví dụ có nhãn của một khái niệm mới, tìm mọi điểm của khái niệm
đó trong một cảnh khác. Về lý thuyết, việc này tách thành hai phần:

* **Nhóm (grouping):** điểm nào thuộc về cùng một vật hay một bề mặt. Không cần biết tên class, và học được từ hình học,
  màu, hay dữ liệu không nhãn.
* **Gọi tên (naming):** nhóm nào giống ví dụ support. Đây là phần few-shot thật sự. Nó đòi một không gian đặc trưng mà hai
  instance khác nhau của cùng khái niệm nằm gần nhau.

Benchmark kế thừa (attMPTI → VIP-Seg) thay bài toán đó bằng một bài toán khác (§2). Dòng phương pháp "không pretrain"
thì giải cả hai phần bằng **một** encoder học từ 6 nhãn (§3), rồi dồn toàn bộ sáng tạo vào head (§4).

---

## 2. Tầng dữ liệu và benchmark: đo cái gì?

| # | Thuộc tính của pipeline | Bằng chứng | Hệ quả |
| :--- | :--- | :--- | :--- |
| D1 | **Mật độ.** Class mà block được lấy mẫu cho bị lấy hai lần: tỉ lệ π(2 − π), tăng từ 0.36 lên 0.59 | `dataloaders/loader.py:39-52`; φ = 1.001 (`results/phase16_p8/SUMMARY.md`) [repo] | U 55.74 trên fixed100 so với 33.11 trên draw đều: **khoảng 22.6 điểm của điểm chuẩn là mật độ** (`results/phase16_d46/SUMMARY.md`) [repo] |
| D2 | **Thứ tự query cố định.** Block query b luôn được lấy mẫu cho way b + 1 | `loader.py:181-223`; D-35, D-37 [repo] | Head VIP-Seg đọc vị trí: đáng +19.94; đảo thứ tự làm 75.36 rơi xuống 0.87 [repo] |
| D3 | **Encoder chỉ thấy XYZ chuẩn hoá min–max theo từng trục trong mỗi block**, cộng RGB. Toạ độ theo mét bị bỏ | `models/encoder.py:644` (`x[:,:,6:]`, `x[:,:,3:6]`); `loader.py:65-74` [repo] | Sàn nằm ở z ≈ 0, trần ở z ≈ 1 trong gần như mọi block. Trục z (khoảng 3 m) bị nén như x, y (1 m), tức méo khoảng 3 lần. Mức "chiều cao nói hộ class" chưa đo |
| D4 | **Class mới mang nhãn background lúc train**, trong chính các cảnh sẽ dùng để test | 63.3 % background lúc train là điểm class test; 95 % block test xuất hiện trong danh sách train (`docs/research/2026-09-28_d48_math_debate.md` §0) [repo] | Mạng được thưởng khi kéo feature của sàn và tường về prototype background. Trong 2D, đây là vấn đề "latent novel class" (Yang et al., ICCV'21: đào lại chúng được +3.7 / +7.0) [snippet]. Chưa đo trong repo |
| D5 | **Có mặt được bảo đảm.** Mỗi block query chứa class của nó ≥ max(5 %, 100 điểm) | `dataloaders/s3dis.py:54-57` [repo] | Prior "class có mặt" được cho sẵn; cảnh thật không bảo đảm điều này |
| D6 | **Block 1 m là mảnh vụn.** Ghế 0.5 × 0.5 m chỉ nguyên vẹn trong khoảng 25 % block; tường chỉ góp 20–30 % của nó | `preprocess/room2blocks.py:34-49` [mine] | Support và query là các mảnh; với sàn, tường, trần thì mảnh chỉ là một mặt phẳng |
| D7 | **Chọn checkpoint trên class test** | D-15, D-22 [repo] | `best` − `last` = 0.3–1.9 |

**Đọc lại con số.** Điểm chuẩn ≈ phát hiện foreground dày + (với head VIP-Seg) giải mã vị trí + prior chiều cao và mặt
phẳng trên mảnh block. Mức trung thực trên draw đều là **28–35**, và oracle của draw đều chỉ khoảng **61**
(`results/phase16_d37/leakfree_cr.json`) [repo]. Giao thức đã sửa của COSeg (CVPR'24) cho SOTA khoảng 37–45 trung bình
(COSeg 36.95, MM-FSS khoảng 44–45.5, WARM khoảng 44.6) [repo/snippet].

**Giả định sai ở tầng này, xếp theo tác động:**
1. "mIoU giao thức chuẩn đo few-shot segmentation, nên 80+ là mục tiêu có nghĩa."
2. "Oracle gap là lỗi prototype đạt tới được": oracle còn mang presence và mật độ (D-35 điểm 3–4).
3. "Train và test tách biệt": thực ra tách theo class, trùng cảnh, và nhãn của class mới bị lật thành background.

---

## 3. Tầng biểu diễn (backbone): ràng buộc chặt nhất

### 3.1 Cơ chế

Cross-entropy với C = 7 class dẫn tới neural collapse (Papyan et al., PNAS 2020): class mean tạo simplex C − 1 chiều, và
biến thiên trong class co về 0. Repo đo được PR = 5.56/128, và mọi checkpoint S1 cho 3.7–7.3, kể cả bản phát hành của
VIP-Seg (6.39). 86 % hướng phân biệt của class mới nằm trong span của các mean base (D-44, D-45) [repo].

Tài liệu về transfer nói cùng một điều:
* Kornblith et al. (NeurIPS'21): tách class nguồn càng chặt, linear transfer càng tệ (ρ = −0.93) [paper, qua ghi chú repo].
* Galanti et al. (ICLR'22): collapse chỉ chuyển sang class mới khi có *nhiều* class nguồn [paper].
* Masarczyk et al. (NeurIPS'23, "tunnel effect"): task nguồn đơn giản tạo tunnel sâu và làm hại transfer [snippet].

### 3.2 Collapse không làm mất khả năng tách trong block, mà làm mất khả năng chuyển giao giữa các instance

* Trong block query, với nhãn: oracle cosine 80.8–83.4, oracle LDA 96.0 [repo]. Thông tin để tách *có* ở đó.
* Từ support sang query: lệch instance a = 0.156, gấp 3.3 lần phương sai 1-shot c = 0.047 (D-46). Đường K-shot U = 56.3 /
  60.2 / 61.6 / 62.7 fit ra U∞ ≈ 64.4 [repo, mine]. Vậy **một prototype mức class hoàn hảo** (từ text, ảnh, audio hay vô
  số shot) cũng chỉ tới khoảng 64 trên giao thức chuẩn.
* **Vì sao VICReg (D-45) thất bại:** nó đổi hình học (PR 42–95) nhưng không thêm thông tin, vì tín hiệu ngữ nghĩa duy nhất
  vẫn là 6 class. Theo bất đẳng thức xử lý dữ liệu, không regularizer nào thiếu thông tin về class mới lại nâng được
  I(Z; nhãn mới). Các chiều mới bị lấp bằng mật độ, vị trí, màu, nên U rơi 5–11 điểm [repo, mine].

### 3.3 Sổ bằng chứng: tầng này chưa từng được thử

Chưa thí nghiệm nào đổi **nguồn** thông tin của encoder. D-20 dùng encoder VIP-Seg nhưng nó học từ cùng 6 class. D-43,
D-45, D-46 và D-49 chỉ đổi kiến trúc hoặc loss trên cùng nguồn giám sát. D-47 là modality mức class. Bốn can thiệp phía
biểu diễn đều **đổi điểm draw đều lấy điểm chuẩn**: M1 +1.55 / −6.33, M5-A +2.56 / −3.65, CB +2.11 / −5.81 [repo].

### 3.4 Biểu diễn tổng quát hiện có (để đối chiếu)

| Mô hình | Điều đáng chú ý | Chấp nhận được với guardrail? |
| :--- | :--- | :--- |
| Sonata (CVPR'25), PTv3 self-distillation | Linear probe ScanNet 72.5 [snippet]. Tập pretrain gồm S3DIS Areas 1–6 **không nhãn** (config Pointcept, đã đọc) | Không, theo guardrail 1 hiện tại; cần decision, và bảng riêng "pretrain không nhãn, trùng cảnh" |
| Concerto (NeurIPS'25) | 3D + DINOv2-giant; linear probe 77.3 [snippet]; inference chỉ cần điểm | Như trên, thêm việc lộ khái niệm qua 2D |
| Lexicon3D (NeurIPS'24) | DINOv2 nâng lên 3D mạnh nhất (ScanNet seg 62.8); CLIP 3.4 [snippet] | Cần ảnh và pose (2D-3D-S), repo chưa có |
| MM-FSS (ICLR'25) | Backbone chưng cất từ LSeg; khoảng +8.6 so với COSeg ở giao thức sạch [snippet] | Tiền lệ được chấp nhận là few-shot SOTA |

Trong phạm vi tìm được, **chưa ai** chạy feature tổng quát đóng băng cộng prototype matching trên benchmark episode. Đó là
một khoảng trống, và là một test rẻ (§6, S2).

---

## 4. Tầng phương pháp (head): ngõ cụt, và vì sao

* Bỏ shortcut đi thì head ≈ prototype matching: CR 54.84, U 55.74, luật support 55.02 (D-37, D-38) [repo].
* Khoảng 17 thí nghiệm head và inference (D-19, D-23…D-29, D-33, D-34, D-37…D-40, D-48): bão hoà ở +2.8 (both + LP) [repo].
* **Module của paper không mang thông tin class.** Entropy gate là hàm lẻ, đơn điệu theo |x| trên từng kênh, xấp xỉ
  x·(0.40 + 0.08|x|). Diffusion cộng cùng một vector vào mọi hàng class. GMMN chỉ khớp *trung bình* hai hàng foreground
  (D-14, D-19; `docs/research/2026-09-24_text_integration_independent.md`) [repo, mine].
* **Lỗi là lỗi gọi tên theo vùng, không phải lỗi nhóm.** Vùng bị bỏ sót bị bỏ trọn (seed recall cục bộ 0.04–0.09),
  homophily 0.94. Thay support prototype bằng mean của chính query thì 55 lên 81. Đó là dấu hiệu "nhóm đúng, gọi sai tên".
  Mọi luật khởi động từ dự đoán đầu tiên (EM, LP, OT, self-support) không lật được một vùng sai đồng nhất; bổ đề ở
  `2026-09-26_beyond_prototypes.md` §2 nói đúng điều đó [repo].
* **Bài học từ 2D** [recall/snippet, cần kiểm lại]:

  | Bước nhảy | Do đâu |
  | :--- | :--- |
  | PANet 48 → PFENet khoảng 60 (PASCAL-5i) | Backbone **đóng băng** để không collapse về class base |
  | HSNet 64–66 | Head tương quan: +4–6, sau đó head chỉ còn +1–3 mỗi paper |
  | BAM 67.8 | **Đặt lại bài toán**: loại class base một cách tường minh |
  | PerSAM khoảng 23 → Matcher 52.7 (COCO-20i) | Cùng SAM để nhóm; **đặc trưng gọi tên** DINOv2 đáng khoảng +29 |
  | Matcher 52.7 → GF-SAM 58.7 | Cùng mô hình; logic nhóm-rồi-khớp đáng +6 |
  | GF-SAM → SINE 64.5 | Bộ khớp học trên **nhiều task đa dạng**, backbone đóng băng |

  Head chỉ có giá trị khi biểu diễn đã tốt và đa dạng.

---

## 5. Tầng đánh giá và thống kê

* **Nhiễu seed bị đánh giá thấp.** sd = 0.5 đến từ 2 seed. Gộp các cặp seed trong repo cho khoảng 1.5 mỗi run (hiệu hai run
  khoảng 2.1). Mọi quyết định có train quanh ±2 điểm chưa được xác lập; các hiệu ≥ 3.6 (D-43, D-45, D-46, D-49) thì vững
  [repo, mine].
* **Oracle không phải cận trên của thứ đạt được.** LDA 96 là điểm resubstitution: fit trên chính 2048 điểm đem chấm.
  Không nên dùng oracle để xếp hạng arm (P8.4) [repo].
* **D-49 "−5.8 trên chuẩn" là prior shift.** Model bất biến mật độ buộc phải thua trên draw mà 88–93 % foreground là điểm
  dày, trừ khi hiệu chỉnh prior. Đó không phải bằng chứng phương pháp hỏng [mine].
* **Chưa làm:** seed thứ hai của CR, S0 cho base sạch, lịch E1 trên head sạch, và test C3 (đảo thứ tự query) trên checkpoint
  route A [repo].

---

## 6. Hướng đi: ba test sàng lọc chỉ cần inference, rồi một kiến trúc

Nguyên tắc: **đổi nguồn thông tin và cách đặt bài toán, không đổi head.** Mọi arm được báo cáo trên ba bộ chấm cùng lúc:
fixed100, draw đều (leak-free), và đảo thứ tự query. Một cải tiến thật có thể trông như thua trên fixed100.

### S1 — Nhóm hay gọi tên? (không cần decision về guardrail; CPU + checkpoint CR)

* **E1, cận trên của bước nhóm:** over-segment block query không dùng nhãn, với K ∈ {8, 16, 32, 64}, bằng:
  (a) region growing hoặc VCCS trên normal + màu + xyz theo mét, và mặt phẳng RANSAC;
  (b) k-means trên feature CR;
  (c) kết hợp.
  Mỗi segment nhận nhãn đa số; đo purity-mIoU và purity của các segment chứa phần foreground CR bỏ sót.
* **E2, gọi tên với nhóm oracle:** thành phần liên thông ground-truth, gọi tên bằng cosine giữa mean segment và support
  prototype của CR.
* **E3/E4, pipeline không nhãn:** segment của E1, gọi tên theo segment. Descriptor = [mean feature CR; hướng normal; chiều
  cao theo mét; độ phẳng; kích thước; thống kê màu], khớp với *segment của support* (đa prototype kiểu Matcher), cổng
  coverage kiểu GF-SAM.
* **Luật đề xuất (cần đăng ký trước khi chạy):**
  * purity(32) ≥ 90, E2 ≤ 65, và E3/E4 − 58.55 ≥ +3 (CI > 0, dương trên cả ba random600) → theo hướng nhóm-rồi-gọi-tên.
  * E2 ≥ 75 nhưng E3 ≤ 58.55 → bộ tách segment là khâu yếu.
  * E4 ≤ E3 → chính feature CR là giới hạn, sang S2.

### S2 — Nguồn thông tin của biểu diễn (cần D-50: guardrail 1 hiện cấm weight point-cloud pretrain)

Cùng các episode, không train:

| Arm | Feature | Ghi chú |
| :--- | :--- | :--- |
| F0 | CR 128-d | tham chiếu |
| F1 | CR 900-d **trước** `bn`/`fc` (`models/vipseg_backbone.py:76-77`) | trong guardrail; thử hiệu ứng projection head / tunnel |
| F2 | Sonata trên chính 2048 điểm của episode (xyz theo mét, rgb, normal PCA) | ngoài phân phối: thưa hơn lúc pretrain |
| F3 | Sonata trên toàn bộ điểm thô của block, lấy tại các điểm đã chọn | lệch giao thức: cũng xoá kênh mật độ |
| F4 | Concerto (base), như F2 | |

Đo cho mỗi arm: U, luật mean đẳng hướng, oracle cosine và LDA, PR, a và c của đường K-shot, b̂ own/other, U leak-free.

Luật đề xuất:
* **G1:** max(F2, F4) − F0 ≥ +5 trên fixed100 (CI > 0, dương trên ba random600), hoặc ≥ +5 trên leak-free → biểu diễn là
  ràng buộc chặt nhất; xây kiến trúc ở dưới.
* **G3:** ≤ 0 trên cả hai draw → không phải backbone; khoảng cách là của chính bài toán (lệch instance cộng mật độ).

Chi phí [mine]: 1–2 h dựng môi trường (spconv, torch_scatter), khoảng 3–4 h GPU. Weight Sonata là CC-BY-NC.

### S3 — Nhiễu ở tầng dữ liệu (không cần decision; inference trên CR)

* **Luật chỉ dùng hình học:** histogram chiều cao chuẩn hoá, |n_z| và khoảng cách tới biên block, lấy từ support; không
  dùng mạng. Nó cho biết bao nhiêu phần của sàn, trần, tường là do hình học "cho không".
* **Phá chuẩn hoá:** XYZ_z = (z − z_min) / 3 m, làm qua wrapper; loader giữ nguyên.
* **Kiểm tra presence:** thêm block query không chứa class đích, đếm false positive mỗi block.
* **Quét cả phòng:** một support, mọi block của một phòng test, lấy mẫu đều, ghép lại. Đây là con số thực tế.

### Kiến trúc nếu S1 và S2 cùng dương: "CascadeProto đặt đúng chỗ"

Giữ ba ý tưởng của paper, nhưng đặt vào chỗ chúng có thông tin:

1. **Đa phương thức = đặc trưng dày theo từng điểm** (3D tổng quát, hoặc 2D nâng lên điểm), không phải một vector tên
   class. Tên class chỉ còn là prior yếu.
2. **Nhóm trước:** segment không phụ thuộc class (hình học cộng feature tổng quát), ở cả support lẫn query.
3. **Entropy = độ bất định của posterior khi gọi tên *segment*,** dùng để chọn segment tự tin làm self-support. Cascade là
   các vòng lặp của việc đó. Đây là dạng có nghĩa của "entropy-aware purification" (gần RePRI và TIM, nhưng ở mức segment).
4. **Meta-train bằng lớp giả (tuỳ chọn, sau cùng):** episode sinh từ các segment không nhãn, với cặp support/query cố ý
   khác instance và khác mật độ, để bộ khớp học bất biến với đúng loại lệch đã đo. Cần decision: dùng vùng không nhãn
   trong các area train.

Kỳ vọng [mine, dải rộng, chưa đo]:
* Giao thức chuẩn, head sạch, feature tổng quát: **55–70**, trung tâm khoảng 62.
* Draw đều hoặc giao thức COSeg: **38–52**, trung tâm khoảng 45, tương đương hoặc vượt nhẹ SOTA sạch.

Trên giao thức sạch, **80+ không có đường nào có bằng chứng**, kể cả với hướng này.

---

## 7. Đóng góp chắc chắn đã có trong tay

Độc lập với §6, repo đã giữ một kết quả mới và đo cẩn thận:
* số của CascadeProto chỉ tái lập được khi chấm trên class đã thấy;
* head VIP-Seg đọc vị trí query (+19.94; đảo thứ tự còn 0.87);
* mật độ lấy mẫu được định lượng bên trong một head gọi tên bằng prototype;
* chọn checkpoint trên class test đáng +1.3 đến +1.9.

Trong phạm vi tìm được (có giới hạn), chưa ai báo cáo shortcut vị trí trong FS-3DSeg. Thí nghiệm giá trị nhất cho phần này
là **chạy test đảo thứ tự query trên các phương pháp cùng họ có code hoặc checkpoint công khai** (Seg-PN; TaylorSeg,
DyPolySeg nếu có checkpoint), vài GPU-h. Nếu shortcut phổ biến trong cả họ, đây là một bài về benchmark (TMLR, hoặc
NeurIPS Datasets & Benchmarks). Nên liên hệ riêng với tác giả trước khi công bố, và dùng cách nói "phù hợp với", không
cáo buộc.

## 8. Thứ tự đề nghị

1. **Bảo vệ nền tảng (GPU rẻ):** seed thứ hai của CR, CR trên S0, test C3 trên checkpoint route A.
2. **S1 và S3** (không cần decision về guardrail): 1–2 ngày, phần lớn trên CPU.
3. **Viết D-50** (sàng lọc biểu diễn tổng quát, bảng riêng), rồi chạy **S2**: khoảng 4 h GPU.
4. Theo luật của S1/S2: kiến trúc ở §6, đánh giá trên fixed100 + leak-free + giao thức COSeg.
5. Song song: test đảo thứ tự trên các phương pháp cùng họ, cho bài về benchmark.
