# Audit: CascadeProto paper vs. repo (HEAD `f78d049`)

> **Lịch sử (2026-09-19):** audit này mô tả code trước khi viết lại. Các lỗi đã được sửa ở phase 8–13; xem [docs/CHANGELOG.md](../CHANGELOG.md) để biết lỗi nào được sửa ở bước nào.

> **Errata (kiểm chứng lại 2026-09-17, chạy loader/metric VIP-Seg gốc và bản tham chiếu toán học):**
> * L6: `room2blocks.py` với tham số mặc định tạo thư mục `blocks_bs1_s1`, không phải `blocks_bs1.0_s1.0`; tên có `.0` chỉ xuất hiện khi truyền `--block_size`/`--stride` tường minh.
> * M2: `P_diffuse` giống nhau giữa các class là do chính Eq.(15)–(18) không có chỉ số class, không phải do ReLU; ReLU chỉ làm `c_unique = 0` khi mọi channel có mean dương ở cả hai nhánh.
> * M6: 2.76M của VIP-Seg trong Table 6 là toàn bộ model (encoder 2.37M + VIP module 0.19M theo log chính thức), không phải riêng backbone.

## 1. Tóm tắt

**Kết luận:** Ở trạng thái hiện tại, repo **không thể tái lập** số liệu của paper. Các module toán học mới (LMA, EPPM, ADRM, loss) phần lớn khớp *từng chữ* với các phương trình mà paper nêu rõ. Tuy vậy, pipeline end-to-end có năm lỗi chặn:

1. `train.py`/`eval.py` chỉ chạy trên dữ liệu ngẫu nhiên.
2. Prototype điểm bị sai khi dùng mask nhị phân thật.
3. Support set bị gộp thành một đám mây điểm trước backbone.
4. Backbone âm thầm rơi về một `TransBlock` có lỗi `batch_first`, thay vì Mamba của VIP-Seg.
5. Dataloader S3DIS không khởi động được với layout do script preprocessing tạo ra.

Ngoài ra còn một số điểm lệch so với paper: split Area 5, CE có trọng số, công thức cross-attention, MMD có `sqrt`, thiếu modality image/audio, và batch size 4 không được dùng.

| Mức độ | Số lượng |
| :--- | :---: |
| Critical (chặn tái lập) | 5 |
| High (model/protocol lệch paper) | 8 |
| Medium | 7 |
| Low (docs, CLI, dead code) | 9 |
| **Tổng** | **29** |

Ngoài 29 finding này còn **17 lỗi trong spec/README/AGENTS** (mục 4), phần lớn là nguồn gốc của các lỗi code.

---

## 2. Nguồn & phương pháp

- **Paper (ground truth):** `C:\Users\USER\Downloads\10069.pdf`, *CascadeProto: Cascaded Cross-Modal Prototype Purification via Entropy-Aware Learning for Few-Shot 3D Point Cloud Segmentation* (Wang et al.). Tôi đọc bản `pdftotext` và **đọc trực tiếp ảnh các trang 2, 5–14**, gồm Table 1, Fig. 1, Eq. (1)–(28), Table 2–6, để kiểm tra các công thức và bảng bị vỡ trong bản text.
- **Repo:** `C:\Users\USER\Desktop\CascadeProto`, branch `master`, HEAD `f78d049a4808cff62f47d20d9956df7cd08b2ef1`. Phạm vi: toàn bộ `git ls-files` (trừ `pointnet2_ops_lib/`) cộng với file untracked `preprocess/download_and_prepare_s3dis.py`. Phân biệt nguồn gốc file: code do commit `2a92e57 init` mang vào (backbone, dataloader, preprocess, README, AGENTS, spec) và code viết sau init (LMA, EPPM, ADRM, loss, train/eval, tests).
- **Những gì đã chạy** (venv `.venv`, torch 2.5.1; mọi script nằm trong scratchpad, không sửa repo):
  - `python -m pytest tests -q` → **25 passed**.
  - Kiểm tra dependency: `mamba_ssm`, `pointnet2_ops`, `h5py`, `transforms3d`, `plyfile`, `whisper` đều **MISSING**. Có `clip` (cache `ViT-B-16.pt`) và `fvcore`.
  - Đếm tham số theo từng module; tính giải tích số tham số Mamba (`d_state=16, d_conv=4, expand=2`).
  - Kiểm tra `TransBlock`: gradient giữa các điểm và giữa các sample.
  - Forward một episode có **shape đúng như `batch_task_collate`** (support `[2,1,9,2048]`, mask nhị phân, query `[2,9,2048]`).
  - Kiểm tra dấu của feature backbone và độ suy biến của diffusion; tính miền giá trị thực của gate entropy.
  - Chạy `model.eval()` hai lần với cùng input.
  - Đo FLOPs bằng `fvcore` (2-way 1-shot, 1 query, 9 kênh).
  - Đo thời gian load CLIP; tính MMD 1-vs-1 dạng đóng.
  - Khởi tạo `S3DISDataset` với layout do `download_and_prepare_s3dis.py` tạo ra.
  - `train.py --dry_run True --dataset scannet` và `eval.py --dry_run True --dataset scannet --modality audio`.
  - Thử các cờ CLI trong README: `--save_dir`, `--cascade_t`, `--data_path`.
- **Quy ước:** **CONFIRMED** = đã kiểm chứng bằng đọc code và/hoặc chạy. **PLAUSIBLE** = paper mơ hồ, nên kết luận phụ thuộc cách hiểu.

---

## 3. Findings

### 3.1 Critical: chặn tái lập

#### C1. `train.py`/`eval.py` không bao giờ dùng dữ liệu thật (CONFIRMED)
- **Paper:** §4.1 *Datasets*: "We adopt the standard block-based partition with 2048 randomly sampled points per block, using Areas 1, 2, 3, 4, 6 for training and Area 5 for testing". *Evaluation Protocol*: "We follow the standard N-way K-shot episodic protocol [34] … across 600 randomly sampled episodes."
- **Code:**
  - `train.py:54-86` `generate_dry_run_batch` tạo `support_x = torch.randn(...)` (`train.py:66`) và `query_y = torch.randint(...)` (`train.py:76`).
  - Vòng train thật cũng gọi chính hàm này: `train.py:215-220`.
  - `eval.py:17` import `generate_dry_run_batch` và dùng ở `eval.py:96-101`.
  - `grep` cho `MyDataset|MyTestDataset|DataLoader` trong `train.py`, `eval.py`, `models/`, `loss/`, `tests/` không có kết quả.
  - Cờ `--data_path` (`train.py:93`) được parse nhưng không bao giờ dùng.
- **Tác động:** Toàn bộ training và mIoU đều là trên nhiễu. Chạy `eval.py --dry_run True` cho "Foreground mIoU: 13.15%" với trọng số ngẫu nhiên và nhãn ngẫu nhiên.
- **Sửa:** Nối `dataloaders/loader.py::MyDataset` (train) và `MyTestDataset` (test) cùng `batch_task_collate` vào train/eval; dùng `sampled_classes` (ID toàn cục) cộng `class2type` để tạo prompt CLIP.

#### C2. Prototype điểm sai với mask nhị phân thật: fg2..N = 0, fg1 = hợp của mọi way (CONFIRMED, đã chạy)
- **Paper:** Eq. (3) và câu sau: "M_fg^(k) = {i | Y_i^s = 1, class(i) = k} … with k = 1, …, N"; §3.1: "Y_i^s ∈ {0,1}^{N_s} is the corresponding binary foreground mask".
- **Code:**
  - `models/vipseg_backbone.py:137-143`:
    ```python
    for k in range(1, n_way + 1):
        fg_mask = (mask == k)
    ```
  - Loader sinh mask nhị phân theo từng way: `dataloaders/loader.py:82-83` `groundtruth = labels == sampled_class`.
  - `models/cascadeproto.py:128-129` flatten `[n_way, k_shot, N]` thành `[1, n_way·k_shot·N]`, nên thông tin way bị mất.
- **Kết quả chạy:** Với support 2-way 1-shot mask {0,1}, chuẩn các hàng `p_point` là `[bg 4.39, fg1 4.37, fg2 0.0]`. `P_fg^(1)` là trung bình foreground của **cả hai** way.
  - Thêm nữa, `models/cascadeproto.py:159` suy ra `n_way = support_y.max()`, tức là 1 với mask nhị phân nếu không truyền `n_way`.
  - Smoke test che lỗi này vì gán mask giá trị `2.0` (`tests/test_smoke_episode.py:38-39`), trái với chính spec 05 ("`Y_s` … ∈ {0.0, 1.0}").
- **Tác động:** Với N ≥ 2, class 2..N không có prototype, nên phân đoạn few-shot thật không hoạt động.
- **Sửa:** Tính `P_fg^(k)` từ `support_y[k-1]` (mask nhị phân của way k, gộp qua K shot) trước khi flatten. `P_bg` = trung bình các điểm có mask 0 trên mọi shot.

#### C3. Support N·K block bị gộp thành một đám mây điểm trước backbone (CONFIRMED)
- **Paper:** Eq. (2): "F = f_enc(X) ∈ R^{N_s×D}", với X_s^i ∈ R^{N_s×3} là **từng** point cloud của support set (§3.1). Ngầm hiểu mỗi block được encode độc lập, như trong VIP-Seg/AttMPTI.
- **Code:**
  - `models/cascadeproto.py:127-129` `x_flat = x_perm.view(1, n_way * k_shot * N_pts, C)`, rồi `models/cascadeproto.py:218` `F_s = self.backbone(s_x)`.
  - Tọa độ dùng cho FPS/kNN là cột `XYZ` đã chuẩn hóa **theo từng block** về [0,1] (`dataloaders/loader.py:70-74`, `models/encoder.py:663` `pos = x[:, :, 6:]`), nên các block chồng lên nhau trong cùng một hộp đơn vị.
  - Số tâm FPS cố định theo `input_points=2048` (`models/encoder.py:511,517-521`): luôn 1024/512/256 tâm dù đám mây có 4096 điểm (2-way 1-shot) hay 30720 điểm (3-way 5-shot).
  - Kết quả chạy: support chuẩn hóa có shape `(1, 4096, 9)`.
- **Tác động:** kNN trộn điểm của các block/class khác nhau, độ phân giải bị hạ theo N·K, và feature của một shot phụ thuộc vào các shot khác. Feature support không còn là feature VIP-Seg.
- **Sửa:** Encode support dưới dạng batch `[N·K, 2048, C]`, rồi reshape về `[N, K, 2048, D]` để tính prototype.

#### C4. Backbone không phải VIP-Seg: khi thiếu `mamba_ssm` thì rơi về `TransBlock`, và `TransBlock` sai `batch_first` (CONFIRMED, đã chạy)
- **Paper:** §4.1: "The shared encoder is VIP-Seg [23] with feature dimension D = 128". Table 1: backbone "VIP-Seg". Table 6: VIP-Seg 2.76M params.
- **Code:**
  - `models/encoder.py:57-59`:
    ```python
    if Mamba is None or mamblock is None:
        return TransBlock(embed_dim=d_model, num_heads=4)
    ```
  - `models/mamba_block.py:20` `nn.MultiheadAttention(embed_dim, num_heads)` với `batch_first=False` mặc định, trong khi `models/encoder.py:552` truyền `[B, G, C]`.
  - `requirements.txt` không liệt kê `mamba_ssm`, và venv hiện tại **không có** `mamba_ssm`.
- **Kết quả chạy:**
  - `batch_first: False`.
  - Với B=1: gradient của output điểm 0 theo các điểm khác = **0.0**, tức không có attention giữa các điểm.
  - Với B=2: gradient của sample 0/điểm 0 theo sample 1/điểm 0 = **19.8**, tức rò rỉ giữa các sample cùng batch (các query của một episode được đưa vào cùng batch, xem M1).
  - Tham số `TransBlock` là 3.64M, so với khoảng 1.98M nếu dùng Mamba (tính giải tích). Backbone là 4.23M, trong khi dùng Mamba sẽ vào khoảng 2.58M, gần với 2.76M của VIP-Seg.
- **Tác động:** Kiến trúc backbone khác paper, không có mô hình hóa tầm xa, và output phụ thuộc vào thành phần batch.
- **Sửa:** Bắt buộc `mamba_ssm` (fail thay vì fallback). Nếu vẫn giữ fallback thì dùng `batch_first=True` và residual đúng (`x + attn(ln(x))`).

#### C5. Pipeline dữ liệu không khởi động được như tài liệu hướng dẫn (CONFIRMED, đã chạy)
- **Paper:** §4.1 (S3DIS/ScanNet blocks 2048 điểm).
- **Code:**
  - `dataloaders/s3dis.py:15` đọc `os.path.join(os.path.dirname(data_path), 'meta', 's3dis_classnames.txt')`.
  - `preprocess/download_and_prepare_s3dis.py:195-197` ghi meta vào `data/s3dis/meta/` và in hướng dẫn `train.py --data_path data/s3dis` (`:217`), nên loader tìm `data/meta/…`.
  - Kết quả chạy: `FileNotFoundError: …\data\meta\s3dis_classnames.txt`.
  - `dataloaders/loader.py:8-9` import `h5py` và `transforms3d`, nhưng venv thiếu cả hai (`ModuleNotFoundError: No module named 'h5py'`).
  - `preprocess/collect_s3dis_data.py:73` cần `datasets/S3DIS/meta/s3dis_classnames.txt`, file không được track (`.gitignore` bỏ qua `datasets/`).
  - ScanNet cần `meta/scannet_classnames.txt` và `scannetv2-labels.combined.tsv` (`preprocess/collect_scannet_data.py:124-125`), cả hai đều không có trong repo.
- **Tác động:** Kể cả khi đã sửa C1, loader vẫn không chạy với dữ liệu do script tạo ra.
- **Sửa:** Thống nhất vị trí meta (ví dụ `data_path/../meta` hoặc `data_path/meta`), commit các file class-names, bổ sung dependency, và thêm hướng dẫn cho ScanNet.

### 3.2 High: model hoặc protocol lệch paper

#### H1. Split Area 5 (S3DIS) và 1201/312 scenes (ScanNet) không được hiện thực (CONFIRMED lệch; PLAUSIBLE về ý định của paper)
- **Paper:** §4.1: "using Areas 1, 2, 3, 4, 6 for training and Area 5 for testing under two category splits S0 and S1". ScanNet: "split into 1201 training and 312 validation scenes partitioned into 36350 blocks".
- **Code:**
  - `dataloaders/s3dis.py:47` `glob(…'data', '*.npy')` gom block của **mọi** area vào cả pool train lẫn pool test. Chỉ tách theo class (`dataloaders/s3dis.py:20-31`).
  - `preprocess/download_and_prepare_s3dis.py:146` gom cả 6 area vào một thư mục.
  - `dataloaders/scannet.py:48` cũng không tách scene.
- **Tác động:** Protocol train/test khác mô tả của paper. Lưu ý: protocol chuẩn của AttMPTI [34] (loader thừa kế) chỉ tách theo class. Paper viết Area 5 nhưng lại so sánh với các baseline dùng protocol [34], xem mục 7.
- **Sửa:** Hỏi tác giả hoặc kiểm tra repo chính thức. Nếu thật sự dùng Area 5 thì lọc `scan_name` theo tiền tố `Area_5` cho test.

#### H2. `w_cls` bị áp dụng hai lần: CE có trọng số, trong khi paper dùng CE chuẩn (CONFIRMED)
- **Paper:** §3.6 Eq. (27): "The segmentation loss is the standard cross-entropy applied to the dynamically routed final logits L_final" (không có trọng số). `w_cls` chỉ xuất hiện trong EPPM (§3.4): "we apply class-specific weights w_cls = [0.8, 1.0, …, 1.0], yielding P_weighted = P_attended ⊙ w_cls".
- **Code:**
  - Trong EPPM: `models/eppm.py:266-271` (đúng paper).
  - Trong loss: `loss/segmentation_loss.py:47-56` `F.cross_entropy(…, weight=weights)` với `weights[0] = 0.8`. `train.py:150-157` ghi chú "CrossEntropy with w_cls=[0.8, 1, 1]".
- **Tác động:** Loss có trọng số background 0.8, lệch Eq. (27) và làm thay đổi cân bằng fg/bg.
- **Sửa:** `F.cross_entropy(logits, targets)` không có `weight`.

#### H3. Cross-attention khác Eq. (13)–(14) (CONFIRMED lệch; PLAUSIBLE vì Eq. (14) không khớp chiều)
- **Paper:** Eq. (13) "Q′ = φ(F^q), S′ = φ(F^s) … where φ is a 1 × 1 convolution" (cùng một φ). Eq. (14) "A = softmax(Q′S′ᵀ/√d) ∈ R^{Nq×Ns}, P_cross = A · ψ(P^{t−1}), where ψ is a learnable linear projection".
- **Code** (`models/eppm.py:95-98, 119-140`):
  - Hai phép chiếu riêng `proj_q` và `proj_s` (không chia sẻ φ).
  - Thêm `F_qs = A_qs·F_s`, thêm `proj_proto`, `proj_qs` và attention thứ hai `A_proto ∈ [B,N+1,N_q]`.
  - `P_cross = A_proto · F_qs`, tức là tổ hợp lồi của feature support đã căn theo query. Giá trị `ψ(P)` **không** đi vào `P_cross`; prototype chỉ dùng làm query của attention.
  - Công thức này đến từ spec 01/02 (xem S3), không phải từ paper.
- **Tác động:** Kiến trúc và số tham số EPPM khác paper, và cơ chế "refine prototype" khác.
- **Sửa:** Vì Eq. (14) không khớp chiều (A là Nq×Ns, ψ(P) là (N+1)×D), cần xác nhận từ code chính thức. Tối thiểu nên dùng chung φ và ghi rõ đây là lựa chọn diễn giải.

#### H4. Vị trí entropy gating: chỉ gate prototype, và prototype đã gate chỉ ảnh hưởng tới trọng số attention (PLAUSIBLE)
- **Paper:** §3.4: "for a feature vector x ∈ R^D, we compute its per-channel entropy … The gated feature is x_gated = x ⊙ g", tiếp đó "After entropy gating, we apply cross-attention". §1: EPPM "suppress high-entropy background features". Eq. (14) và Eq. (21) dùng `P^{t−1}`.
- **Code:**
  - `models/eppm.py:325` `P_gated, entropy = self.gating(P_prev)`.
  - `P_gated` chỉ vào `proj_proto` (`models/eppm.py:132`). Diffusion dùng `F_s`/`F_q` chưa gate (`:331`), và residual dùng `P_prev` chưa gate (`:334`, `:274`).
  - Miền giá trị gate đo được ở θ=0.5 là **[0.405, 0.731]** (H ∈ [0, ln2]), tức gate gần như là một phép scale đều.
- **Tác động:** Hiệu ứng "entropy purification" rất yếu, trong khi Table 4 quy +1.42% cho Entropy Gate.
- **Sửa:** Làm rõ x là `F^s`/`F^q` hay `P^{t−1}`, và áp gate vào nhánh thực sự tạo ra `P_cross`/`P^t`.

#### H5. GMMN: code lấy `sqrt` (Eq. (7) là bình phương chuẩn RKHS) và tính MMD 1-vs-1 theo từng class (CONFIRMED phần sqrt; PLAUSIBLE phần per-class)
- **Paper:** Eq. (7): "MMD(P,Q) = ‖(1/|P|)Σφ(x) − (1/|Q|)Σφ(y)‖²_H". Eq. (8): "L_GMMN = 0.1·MMD(P_modal^bg, P_point^bg) + 1.0·MMD(P_modal^fg, P_point^fg)".
- **Code:**
  - `loss/gmmn_loss.py:134` `use_squared: bool = False` → `:164` chọn `compute_mmd`, tức `sqrt(mmd_sq + eps) − sqrt(eps)` (`:119`).
  - `:172-179` vòng lặp theo từng class k, mỗi class là một cặp 1-vs-1, sau đó lấy trung bình.
- **Kết quả chạy:** MMD² 1-vs-1 = `2·(6 − k(x,y))` = 6.14, trong khi code trả về 2.48. Nghĩa là loss GMMN theo code chỉ là một khoảng cách kernel theo từng điểm, không phải MMD giữa hai phân phối.
- **Tác động:** Độ lớn và gradient của L_GMMN khác Eq. (7)–(8), nên cân bằng λ = 1.0 với L_seg cũng thay đổi.
- **Sửa:** Mặc định `use_squared=True`. Cân nhắc MMD trên tập `P^fg` (N mẫu), hoặc mở rộng mẫu bằng nhiều z hay bằng feature từng điểm (xem mục 7).

#### H6. Không có modality image và audio; `--modality` bị bỏ qua (CONFIRMED)
- **Paper:** Table 1 (CascadeProto: Text ✓, Audio ✓, Image ✓). Fig. 1 ("Audio → Whisper → CLIP", "Image → CLIP"). Eq. (4) m ∈ {text, image, audio}. Table 2/3 có các hàng "CascadeProto (Audio)" và "(Image)".
- **Code:**
  - Chỉ có `TextModalityAdapter` (`models/lma.py:93`) và prompt text (`models/lma.py:36-44`).
  - `train.py:101-102` và `eval.py:42` nhận `--modality` nhưng không dùng. Chạy `eval.py --modality audio` vẫn dùng CLIP text.
  - Không có code cho Whisper hay CLIP vision.
- **Tác động:** Không tái lập được 2/3 số hàng kết quả trong Table 2/3.
- **Sửa:** Thêm pipeline image (CLIP visual encoder, 512-d) và audio (Whisper → CLIP), mỗi modality một adapter; báo lỗi khi gặp modality chưa hỗ trợ.

#### H7. Batch 4 episode, số epoch theo dataset và `--dataset` không được hiện thực (CONFIRMED)
- **Paper:** §4.1: "The batch size is 4 episodes, trained for 50 epochs on S3DIS and 30 epochs on ScanNet."
- **Code:**
  - `train.py:107-108` định nghĩa `--batch_size` nhưng không dùng. `train.py:61` `B = 1`, mỗi episode một `optimizer.step()` (`train.py:222-226`).
  - `--epochs` mặc định 50 cho mọi dataset (`train.py:105`).
  - `get_s3dis_class_names` luôn được dùng, kể cả khi `--dataset scannet` (`train.py:163`, `eval.py:87`). Kết quả chạy: "Dataset: SCANNET … Unseen Test Classes: ['beam', 'board', …]".
- **Tác động:** Nhiễu gradient và thống kê BN khác paper; ScanNet dùng sai class.
- **Sửa:** Tích lũy 4 episode cho mỗi step (hoặc batch thật); lấy class từ `ScanNetDataset`; mặc định 30 epoch cho ScanNet.

#### H8. Protocol đánh giá khác [34] (PLAUSIBLE)
- **Paper:** §4.1: "We follow the standard N-way K-shot episodic protocol [34] … Performance is reported as mean IoU (mIoU) averaged over S0 and S1 across 600 randomly sampled episodes." Paper không đưa ra công thức mIoU.
- **Code:**
  - `eval.py:110-113` tính mIoU **theo từng episode** (bỏ qua class NaN) rồi lấy trung bình.
  - `eval.py:95` lấy mẫu class ngẫu nhiên.
  - `eval.py:77-84` âm thầm chạy với trọng số ngẫu nhiên khi `--checkpoint` thiếu hoặc sai đường dẫn.
  - `MyTestDataset` của loader (`dataloaders/loader.py:255-256`) sinh tập test cố định gồm 100 episode cho mỗi tổ hợp class (S3DIS 2-way: C(6,2)·100 = 1500). Tập này không được dùng.
  - Theo code công khai của AttMPTI (không có trong repo), TP/FP/FN được **cộng dồn trên toàn bộ tập test** theo class toàn cục rồi mới tính IoU. Paper không nói rõ điều này.
- **Tác động:** Số mIoU không so sánh được với Table 2/3; lấy trung bình theo episode thường cho giá trị khác so với cộng dồn.
- **Sửa:** Dùng `MyTestDataset` và cộng dồn TP/FP/FN theo class toàn cục; báo lỗi nếu không load được checkpoint.

### 3.3 Medium

#### M1. Query `[n_way·n_queries, C, N]` vs support B=1: broadcasting ngầm (CONFIRMED, đã chạy)
- **Paper:** §3.1 định nghĩa một query X_q, và Eq. (14)–(23) được viết cho một query.
- **Code:** `models/cascadeproto.py:209,219` cho `F_q` B=2 và `F_s` B=1. `torch.matmul` ở `models/eppm.py:124` và phép cộng ở `:274` broadcast ngầm. Kết quả chạy: `stage_prototypes[0]` có shape `(2, 3, 128)`, `w_gate` `(2, 4)`, trái với docstring `[B, N+1, D]` (`models/cascadeproto.py:42-44`).
- **Tác động:** Về số học, mỗi query có một quỹ đạo prototype riêng (chấp nhận được), nhưng: (a) contract shape sai; (b) query dùng chung batch nên chịu rò rỉ từ C4; (c) sẽ vỡ khi B_support > 1.
- **Sửa:** Lặp tường minh theo từng query hoặc `expand` support tới B_q, và assert shape.

#### M2. Prototype diffusion suy biến: P_diffuse gần như hằng và giống nhau cho mọi class (CONFIRMED, đã chạy; code khớp nguyên văn Eq. (15)–(18))
- **Paper:** Eq. (15) "q_ch = σ(mean(F^q, dim=1))", "m_common = 𝟙[q_ch > τ] ⊙ 𝟙[s_ch > τ]", Eq. (16)–(18), "with α = 0.5, τ = 0.5".
- **Code:**
  - `models/eppm.py:178-196` hiện thực đúng công thức.
  - Nhưng head của backbone kết thúc bằng `BatchNorm1d + ReLU` (`models/vipseg_backbone.py:47-54`), nên F ≥ 0, suy ra σ(mean) ≥ 0.5.
  - Kết quả chạy: tỉ lệ channel có m=1 là **1.0**, `c_unique = 0`, `P_diffuse` giống nhau giữa các class, giá trị trong [0.296, 0.300].
- **Tác động:** Nhánh diffusion gần như chỉ là một bias hằng. Phần lớn là hệ quả của công thức paper khi đi cùng backbone có ReLU cuối, nhưng vẫn khiến fusion `w` vô nghĩa.
- **Sửa:** Xác nhận với tác giả xem feature có đi qua ReLU trước EPPM không, hoặc τ được áp dụng sau chuẩn hóa nào.

#### M3. LMA lấy mẫu nhiễu z mới khi inference, nên dự đoán không tất định (CONFIRMED; PLAUSIBLE về ý định)
- **Paper:** Eq. (6) "random noise z ∼ N(0, I)". Eq. (9) "The fused initial prototype **used for training** is P^0 = P_point + P_modal". §3 mở đầu hứa "we present the training objectives and inference strategy", nhưng không có mục inference strategy.
- **Code:** `models/lma.py:184-185` `noise = torch.randn_like(e_adapted)` luôn được gọi, kể cả ở `eval()`. Kết quả chạy: hai lần `model.eval()` cùng input có logits lệch tối đa **0.249**.
- **Tác động:** Kết quả đánh giá không lặp lại được giữa các lần chạy.
- **Sửa:** Ở eval dùng z = 0 hoặc trung bình nhiều mẫu z (cần xác nhận với tác giả).

#### M4. Biến thể CLIP phụ thuộc cache cục bộ; CLIP bị load lại mỗi episode (CONFIRMED)
- **Paper:** Chỉ nói "CLIP [17]" và "each LMA projects CLIP embeddings from R^512", không nêu biến thể.
- **Code:** `models/lma.py:70` `model_name = "ViT-B/16" if os.path.exists(~/.cache/clip/ViT-B-16.pt) else "ViT-B/32"`, và `:71` `clip.load` được gọi mỗi lần vì `train.py:80` và `eval.py:96` không truyền `clip_model`. Kết quả chạy: mỗi lần gọi mất 3.07 s và 2.37 s.
- **Tác động:** Hai máy khác nhau cho ra embedding khác nhau; train 50 epoch × 100 episode sẽ tốn khoảng 3 giờ chỉ để load CLIP.
- **Sửa:** Cố định biến thể qua CLI, load một lần, và cache embedding theo class.

#### M5. Kênh input và số điểm không nhất quán (CONFIRMED)
- **Paper:** §4.1: "S3DIS contains RGB point clouds", "2048 randomly sampled points per block". Paper không nêu tập thuộc tính input.
- **Code:**
  - Dry-run và spec chỉ dùng xyz 3 kênh, nên `models/vipseg_backbone.py:70-79` pad **RGB = 0** và chuẩn hóa XYZ về [−1,1].
  - Loader dùng XYZ ∈ [0,1] (`dataloaders/loader.py:71-74`) và `encoder.py:663` lấy cột 6:9 làm tọa độ.
  - `MyDataset` mặc định `num_point=4096`, `pc_attribs='xyz'` (`dataloaders/loader.py:117`). Nếu nối loader mà không truyền `pc_attribs='xyzrgbXYZ'`, `num_point=2048` thì sẽ mất màu hoặc sai số điểm.
- **Tác động:** Phân phối input khác nhau giữa các đường chạy; RGB bị bỏ.
- **Sửa:** Truyền cố định `num_point=2048`, `pc_attribs='xyzrgbXYZ'`, và assert C == 9.

#### M6. Params/FLOPs lệch Table 6 (CONFIRMED đo; một phần do paper tự mâu thuẫn)
- **Paper:** Table 6: "CascadeProto (Ours) 2.88 [M] 8.86 [G]". §4.4: "a marginal overhead of 0.12M parameters and 0.38 GFLOPs over the VIP-Seg backbone (2.76M, 8.48G)".
- **Code (đo):** tổng 4.77M, gồm backbone 4.23M (encoder 4.03M, trong đó `TransBlock` 3.64M), LMA 0.148M (adapter 0.082M, generator 0.066M), cascade 0.382M (0.095M/stage), ADRM 0.0005M. Phần thêm vào là **0.53M**, so với 0.12M của paper. FLOPs theo `fvcore` (2-way 1-shot, chỉ các op được hỗ trợ): tổng **18.8G**, gồm backbone 11.8G và cascade 7.0G.
- **Tác động:** Không khớp Table 6. Chú ý: riêng adapter 512→128→128 với LayerNorm đã là 0.082M, và một attention Nq×Ns×d = 2048·4096·72 ≈ 0.6G phép nhân cho mỗi stage. Vì vậy con số 0.12M / 0.38G khó đạt được ngay với công thức của paper (mục 7).
- **Sửa:** Sau khi sửa C3/C4/H3, đo lại bằng cùng công cụ và cấu hình với paper.

#### M7. "Epoch" = 100 episode tổng hợp; không có validation hay chọn checkpoint (CONFIRMED; PLAUSIBLE vì paper không định nghĩa)
- **Paper:** §4.1 chỉ nói "50 epochs … StepLR scheduler that halves the rate every 10 epochs". Paper không nêu số episode mỗi epoch và không nhắc validation.
- **Code:** `train.py:212` `num_episodes_per_epoch = 100` bị hard-code; `scheduler.step()` mỗi 100 episode (`:230`); chỉ lưu `cascadeproto_epoch_{n}.pt` mỗi 10 epoch (`:235-243`).
- **Tác động:** Tổng số bước train và lịch học tùy ý, không truy được về paper.
- **Sửa:** Định nghĩa epoch theo `MyDataset(num_episode=…)`, đưa thành tham số CLI, ghi rõ vào log/README.

### 3.4 Low

| ID | Vấn đề | Bằng chứng code | Trạng thái |
| :--- | :--- | :--- | :--- |
| L1 | Lệnh trong README không chạy được | `README.md:196-197` `--cascade_t`, `--save_dir` → `train.py: error: unrecognized arguments` (đã chạy); `README.md:207` `eval.py --data_path` → lỗi (đã chạy); `README.md:213` `best_model.pth` không bao giờ được lưu (`train.py:236`); `README.md:110-111,126` trỏ tới `dataloaders/preprocessing/…` và `prepare_scannet_blocks.py`, không tồn tại (thực tế là `preprocess/`); `room2blocks.py` không có `--save_path` (`preprocess/room2blocks.py:177-183`) | CONFIRMED |
| L2 | Dead code / import hỏng | `models/vipseg_learner.py:5` → `ImportError: cannot import name 'VIPSeg'` (đã chạy); `GatingNetwork` (`models/vipseg_backbone.py:152-176`) trùng ADRM và không dùng; `hidden_dim=256` không dùng (`models/cascadeproto.py:67`); `model.criterion` (`models/cascadeproto.py:105`) không dùng vì `train.py:151` tạo criterion riêng | CONFIRMED |
| L3 | Tắt kiểm tra TLS toàn cục khi import `models.lma` (scope creep, rủi ro bảo mật) | `models/lma.py:21-27` `ssl._create_default_https_context = ssl._create_unverified_context` | CONFIRMED |
| L4 | Dependency / môi trường | `requirements.txt` thiếu `torch`, `clip`, `mamba_ssm`, `pointnet2_ops`, `plyfile`, `huggingface_hub`; `README.md:91` ghim `torch==2.0.1 … cu118` trong khi paper train trên "NVIDIA RTX 5090" (§4.1); GPU Blackwell nhiều khả năng cần bản build CUDA 12.8 trở lên | CONFIRMED (thiếu dep) / PLAUSIBLE (torch vs 5090) |
| L5 | `eval.py` im lặng khi checkpoint sai; `--modality` là chuỗi tự do | `eval.py:77-84`, `eval.py:42` | CONFIRMED |
| L6 | Preprocessing S3DIS dễ vỡ | `preprocess/collect_s3dis_data.py:97-98` `scene_path.split('/')` sai trên Windows; tên thư mục `blocks_bs1.0_s1.0` (`room2blocks.py:191`) khác `blocks_bs1_s1` (`download_and_prepare_s3dis.py:105`); `:206` `os.system('mklink /J …')` chỉ chạy trên Windows | CONFIRMED |
| L7 | `utils/checkpoint_util.py` monkey-patch `torch.load` toàn cục (`weights_only=False`); hiện không được import | `utils/checkpoint_util.py:7-16` | CONFIRMED |
| L8 | Smoke test dùng mask {1,2}, trái spec 05 và loader thật, nên che lỗi C2 | `tests/test_smoke_episode.py:38-39` | CONFIRMED |
| L9 | Chi tiết nhỏ không có trong paper: fusion weight theo từng class `[B,N+1,2]` so với "w ∈ R²"; SE reduction 4; Dropout đặt sau ReLU; clamp p ∈ [1e-7, 1−1e-7] | `models/eppm.py:255`, `:216-219`; `models/lma.py:116-120`; `models/eppm.py:36` | PLAUSIBLE |

---

## 4. Lỗi trong spec/README/AGENTS (mâu thuẫn hoặc không có trong paper)

| ID | Tài liệu | Nội dung sai | Paper nói |
| :--- | :--- | :--- | :--- |
| S1 | `docs/spec/04_DATA_AND_EPISODES.md:17-18`, `README.md:117-118` | S0 = {ceiling, floor, wall, beam, column, window}; S1 = {door, table, chair, sofa, bookcase, board} | Paper **không liệt kê** class của S0/S1, chỉ nói "two category splits S0 and S1" theo [34]. Code (`dataloaders/s3dis.py:20-21`, `train.py:47-48`) dùng split của AttMPTI {beam, board, bookcase, ceiling, chair, column}. Spec tự đặt ra split này. |
| S2 | `04_…:68,77`, `README.md:65`, `AGENTS.md:27` | Ngưỡng "≥ 50 points" cho support/query | Paper không có. Loader dùng `max(int(N·0.05), 100)` (`dataloaders/s3dis.py:55`). |
| S3 | `01_ARCHITECTURE_SPEC.md:133-140`, `02_TENSOR_MATH_SPEC.md:80-88` | Thêm `F_qs = A_qs F_s`, `A_proto = softmax(ψ(P)φ(F_qs)ᵀ/√72)`, `P_cross = A_proto·F_qs`, và phương án "Alternatively…" | Eq. (14) chỉ có "P_cross = A · ψ(P^{t−1})". Công thức của spec là tự chế, dẫn tới H3. |
| S4 | `05_VERIFICATION_PLAN.md:171` | "L_seg = CrossEntropy(S_final, Y_q; w_cls), w_cls = [0.8, 1.0, 1.0]" | Eq. (27) là "standard cross-entropy", không trọng số. Dẫn tới H2. |
| S5 | `03_MULTIMODAL_SPEC.md:107` vs `05_…:78` vs `02_…:52-53` | Ba định nghĩa MMD khác nhau: `sqrt(max(·, ε))`; MMD² trung bình theo từng class; MMD trên tập `P[1:]` | Eq. (7) là chuẩn RKHS **bình phương**, không có sqrt; Eq. (8) dùng tập `P^fg`. Dẫn tới H5. |
| S6 | `02_…:145-148`, `AGENTS.md:122` | Bắt buộc clamp `p ∈ [1e-7, 1−1e-7]` | Paper chỉ có "ε = 10^−8". Vô hại, nhưng không phải nội dung paper. |
| S7 | `01_…:186-189`, `04_…:87-90`, `05_…:158` | Support/query chỉ có xyz 3 kênh | Paper: "S3DIS contains RGB point clouds"; VIP-Seg/loader dùng 9 kênh. Dẫn tới M5. |
| S8 | `04_…:24`, `README.md:131` | "The 312 validation scenes are pre-partitioned into 36,350 blocks" | Paper: "1513 RGB-D indoor scans … split into 1201 training and 312 validation scenes partitioned into 36350 blocks". Cụm "partitioned into" gắn với toàn bộ dữ liệu; spec đọc sai thành chỉ tập validation. |
| S9 | `04_…:118-129` | mIoU trung bình theo từng episode | Paper không đưa công thức, chỉ nói theo protocol [34]. Dẫn tới H8. |
| S10 | `03_…:5, 52-54` | "Frozen OpenAI CLIP (ViT-B/32 or ViT-B/16) and Whisper ASR", audio được "transcribed" rồi đưa qua CLIP text | Paper không nêu biến thể CLIP; Whisper chỉ xuất hiện trong Fig. 1 ("Audio → Whisper → CLIP"), không có mô tả về transcription. |
| S11 | `03_…:49` | Prompt background "This point cloud represents the background clutter." | Paper chỉ có ví dụ foreground "This point cloud represents the chair." (Fig. 1). Prompt background không được nêu. |
| S12 | `04_…:104-110` | Cột ghi chú vô nghĩa (ví dụ Learning Rate → "Scaled dot-product prototype matching") | Không có trong paper. |
| S13 | `AGENTS.md:84`, `04_…:14`, `README.md:116,202` | `s3dis.py` "(Areas 1-4, 6 train; Area 5 test)" | Paper **có** nói Area 5, nhưng code không làm (H1). AGENTS mô tả sai code. |
| S14 | `AGENTS.md:113-116`, `05_…:48-49` | "Phase 3: Dry-Run Episode Execution on Real S3DIS Dataloader" | `train.py --dry_run` dùng `torch.randn` (C1). |
| S15 | `AGENTS.md:54` | "Never rewrite … evaluation routines inherited from the base VIP-Seg repository" | Repo không có eval của VIP-Seg (`models/vipseg.py` chỉ là stub, `vipseg_learner.py` hỏng); `eval.py` viết lại protocol (H8). |
| S16 | `README.md:3` | "Official PyTorch implementation" | `AGENTS.md:9` ghi "Re-implement and reproduce". Đây không phải code chính thức. |
| S17 | `README.md:163-172` | Table 6 bỏ hàng QGPA (2.79/16.30/56.30) | Paper Table 6 có QGPA. Nhỏ. |

**Các nghi vấn từ lượt trước đã bị bác bỏ:**
- "README Table 6 88.53 vs Table 2 86.53" **không phải** mâu thuẫn. Table 6 là "on S0 split", và 88.53 đúng bằng cột S0 của CascadeProto (Text), 2-way 1-shot, trong Table 2 (Avg là 86.53).
- Miền gate "[0.45, 0.73]" không xuất hiện trong docs, và miền thực tế đo được là **[0.405, 0.731]** (θ=0.5, log tự nhiên).
- θ init 0.5, d=72, τ=α=0.5 trong spec đều **khớp** paper.

---

## 5. Chi tiết paper mà repo bỏ qua hoàn toàn

1. **Modality image và audio** (Table 1, Fig. 1, Eq. (4), các hàng Table 2/3): không có encoder, adapter hay dữ liệu (ảnh/âm thanh theo class). Paper cũng không nói ảnh/âm thanh lấy từ đâu.
2. **Split Area 5 / 1201–312 scenes** (§4.1): không hiện thực (H1).
3. **Batch 4 episode; 30 epoch cho ScanNet** (§4.1): không hiện thực (H7).
4. **Ablation Table 4** (bỏ LMA, Entropy Gate, Cascade, ADRM) và **Table 5** (T = 1..6): không có cờ CLI. `num_stages` có trong `CascadeProto.__init__` nhưng `train.py:140` hard-code 4, và không có cách tắt gate/LMA/ADRM.
5. **Đo độ phức tạp** (Table 6: Params/FLOPs; §4.3: "inference time grows linearly with T at approximately 16ms per stage"): không có script.
6. **"Inference strategy"** được hứa ở §3 (trang 6) nhưng paper không trình bày, và repo cũng không có quy ước inference riêng (M3).
7. **Evaluation trên dữ liệu thật cho 2/3-way, 1/5-shot**: chưa có đường chạy nào (C1).
8. Pipeline preprocessing ScanNet đầy đủ (meta/tsv, hướng dẫn chạy `room2blocks --dataset scannet`) chưa có.

---

## 6. Những điểm đã kiểm chứng là đúng

- **D = 128** (Eq. 2): `VIPSegBackbone(out_dim=128)`. **d = 72** (Eq. 13): `models/eppm.py:95-96`. **T = 4** (§3.5): `EPPMCascade(num_stages=4)`, mỗi stage có θ riêng ("each EPPM applies entropy gating independently").
- **Eq. (10)–(12):** `compute_shannon_entropy` (sigmoid, log tự nhiên, ε=1e-8), `g = σ(2(θ−H))`, θ learnable khởi tạo 0.5 (`models/eppm.py:34-72`).
- **Eq. (15)–(18) nguyên văn**, τ = α = 0.5 (`models/eppm.py:178-193`); xem M2 về tính suy biến.
- **Eq. (19)** softmax 2 nhánh; **Eq. (20)** SE với AvgPool trên chiều class; **w_cls = [0.8, 1, …, 1]** trong EPPM; **Eq. (21)** `LN(W_out·ReLU(P_weighted) + P^{t−1})` (`models/eppm.py:254-274`).
- **Eq. (23)** `L^t = F^q (P^t)ᵀ` không scale (`models/eppm.py:338`).
- **Eq. (24)–(25)** ADRM: `W_g ∈ R^{T×D}` không bias, AvgPool F^q, softmax, tổng có trọng số (`models/adrm.py:42-82`).
- **Eq. (26)** λ = 1.0; loss chỉ áp lên `L_final`, không có deep supervision cho từng stage, đúng như Eq. (27) mô tả (`loss/segmentation_loss.py:105-107`).
- **Eq. (5)** adapter 512→128→128 với LayerNorm, ReLU, Dropout; **generator G là MLP 3 lớp** nhận (E, z); **Eq. (9)** `P^0 = P_point + P_modal` (`models/lma.py:93-204`).
- **Eq. (8)** trọng số 0.1 (bg) / 1.0 (fg); **bandwidth σ ∈ {2,5,10,20,40,80}** (`loss/gmmn_loss.py:22,131-132`); xem H5 về sqrt.
- **Optimizer:** AdamW, lr 1e-3, weight decay 0.1, StepLR(step 10, γ 0.5) (`train.py:109-116,160-161`).
- **Danh sách class của fold** S3DIS/ScanNet khớp AttMPTI [34] (`dataloaders/s3dis.py:20-21`, `dataloaders/scannet.py:21-22`); nhãn query được đánh lại chỉ số 0..N với 0 là background (`dataloaders/loader.py:85-88`), khớp thứ tự `[P_bg; P_fg^(1..N)]` của Eq. (3).
- **Prompt foreground** "This point cloud represents the {c}." khớp Fig. 1.
- `eval.py` mặc định 600 episode, khớp "600 randomly sampled episodes" (nhưng xem C1/H8).
- **Số liệu trong README** Table 2, 3, 6 khớp paper (đã so từng ô, trừ việc bỏ hàng QGPA).
- Test suite: 25/25 pass.

---

## 7. Câu hỏi mở (paper mơ hồ hoặc tự mâu thuẫn)

1. **Eq. (14) không khớp chiều:** `A ∈ R^{Nq×Ns}` nhân `ψ(P^{t−1}) ∈ R^{(N+1)×D'}` là không xác định. `P_cross` được tính chính xác thế nào?
2. **Gate áp lên vector nào?** Eq. (10) nói "a feature vector x ∈ R^D", còn §1 nói "suppress high-entropy background features". Gate áp lên F^s, F^q hay P^{t−1}? θ là scalar hay theo channel?
3. **φ chung hay riêng** cho Q′/S′ (Eq. 13 dùng cùng ký hiệu φ)?
4. **Eq. (7)–(8):** "MMD" là giá trị bình phương (đúng định nghĩa Eq. 7)? `P^fg` là tập N prototype (MMD giữa hai phân phối) hay so khớp từng class? Có detach `P_point` không? Hiện tại gradient L_GMMN chảy ngược vào backbone qua `p_point` (`loss/gmmn_loss.py:144-176`, không detach).
5. **§3.2 mâu thuẫn:** "LMA project CLIP text **and** image embeddings … A GMMN-based distribution matching module then **fuses both sources**", trong khi Fig. 1, §1 và Table 2 nói "single modality". `E_fused` trong Eq. (6) là gì?
6. **Inference:** z lấy ngẫu nhiên, z = 0, hay bỏ P_modal? Eq. (9) chỉ nói "used for training".
7. **Split:** "Areas 1, 2, 3, 4, 6 for training and Area 5 for testing" có thật được dùng không, khi các baseline trong Table 2 (AttMPTI, Seg-PN, VIP-Seg) dùng protocol chỉ tách theo class? ScanNet có tách 1201/312 không?
8. **Đánh giá:** "600 randomly sampled episodes" so với [34] (100 episode × mỗi tổ hợp class); mIoU cộng dồn hay trung bình theo episode? Câu "averaged over S0 and S1" mâu thuẫn với việc bảng báo cáo riêng S0/S1/Avg.
9. **Table 6 / §4.4:** overhead 0.12M params và 0.38G FLOPs cho LMA + 4 EPPM + ADRM. Với 512→128 (Eq. 5, §4.1) riêng adapter đã khoảng 0.082M, và attention Nq×Ns ở d=72 khoảng 0.6G phép nhân mỗi stage. Có subsampling hay chia sẻ tham số không?
10. **Eq. (23)** được gọi là "scaled dot-product matching" nhưng công thức không có hệ số scale hay temperature. Có cosine hoặc temperature không?
11. **Eq. (19):** `w ∈ R²` là toàn cục (sau pooling) hay theo từng class như code?
12. **Epoch** gồm bao nhiêu episode? Có data augmentation không (loader có `pc_augm`, paper không nhắc)?
13. **Audio/Image:** nguồn âm thanh/ảnh cho mỗi class; Whisper dùng để transcribe hay lấy embedding; biến thể CLIP; prompt background.
14. **Feature trước EPPM** có qua ReLU không? Nếu có, diffusion với τ = 0.5 luôn suy biến (M2).
