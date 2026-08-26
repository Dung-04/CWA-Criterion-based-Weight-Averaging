"""
Configuration file for CWA - Object Detection (Ultralytics YOLO + Pascal VOC)

Nhánh này CHỈ làm object detection với các model YOLO (qua Ultralytics API).
Mọi tham số chỉnh ở đây; các giá trị hay dùng đều override được qua CLI (main.py).
"""
import os


class Config:

    # ===================== Model Configuration =====================
    # KHÔNG chốt cứng version YOLO nào làm mặc định — bạn TỰ SET giá trị này
    # (hoặc truyền --model khi chạy). Nhận mọi giá trị mà ultralytics.YOLO()
    # nhận, đổi version = sửa đúng 1 dòng:
    #   - Pretrained weights : "yolov8n.pt" | "yolo11n.pt" | "yolov5nu.pt" | ...
    #   - Custom weights     : "path/to/your_best.pt"
    #   - Train from scratch : "yolov8n.yaml" (hoặc .yaml kiến trúc custom)
    MODEL = "yolov8s.pt"  # <-- ĐẶT MODEL CỦA BẠN Ở ĐÂY, ví dụ: "yolov8n.pt"

    # ===================== Dataset Configuration =====================
    # "VOC.yaml" = Pascal VOC built-in của Ultralytics, TỰ ĐỘNG DOWNLOAD lần đầu
    # (https://docs.ultralytics.com/datasets/detect/voc). Data đã tải sẵn /
    # data custom: trỏ tới file data.yaml của bạn.
    #
    # ⚠️ VOC.yaml GỐC: split `val` và `test` TRÙNG NHAU (đều là VOC2007 test,
    # 4952 ảnh) — KHÔNG có validation set độc lập. CWA cần chọn/average
    # checkpoint theo fitness trên val ĐỘC LẬP với test, nên pipeline tự tách
    # VAL_RATIO từ train làm validation riêng (xem dataset.py):
    #   train (16551 ảnh) → train' (1 - VAL_RATIO) + val' (VAL_RATIO, holdout)
    #   test  = VOC2007 test (4952 ảnh), giữ nguyên làm hold-out báo cáo cuối
    DATA = "VOC.yaml"
    VAL_RATIO = 0.1  # tỉ lệ tách val' từ train (giống VALIDATION_RATIO nhánh classification)
                     # = 0 → dùng nguyên data.yaml gốc: val ≡ test, CWA
                     #   sẽ chọn checkpoint trên chính tập test (leakage) — tránh!

    # ===================== Training Configuration =====================
    EPOCHS = 100
    IMGSZ = 640
    BATCH = 128     # -1 = auto-batch theo VRAM (chỉ áp dụng khi train)
    DEVICE = None     # None = auto (GPU nếu có); "0" | "0,1" | "cpu"
    WORKERS = 8
    # Số worker cho dataloader của các lượt SAU train (model.val của từng
    # strategy + BN recalibration). Tách khỏi WORKERS vì RAM, không vì tốc độ:
    # mỗi worker giữ prefetch_factor(=2) batch trong hàng đợi, tức
    #     workers × 2 × batch × 3 × imgsz² byte
    # ≈ 7 GB với WORKERS=24, batch=128, imgsz=640 cho MỘT dataloader. Lúc train
    # chỉ có 1 loader nên chịu được, nhưng mỗi seed còn chạy 6 lượt val + 5 lượt
    # BN recalibration nối nhau. None = min(WORKERS, 8).
    EVAL_WORKERS = 8
    # Tự hạ WORKERS/EVAL_WORKERS theo số CPU mà SLURM thực sự cấp cho job.
    # Cần vì Ultralytics chặn worker bằng os.cpu_count() (CPU của CẢ NODE) còn
    # PyTorch cảnh báo theo os.sched_getaffinity (CPU của job) → mặc kệ thì log
    # đầy "This DataLoader will create N worker processes in total".
    AUTO_LIMIT_WORKERS = True
    PATIENCE = 10   # Ultralytics early stopping (epoch không cải thiện fitness val)
    PRETRAINED = True
    CACHE = False     # False | "ram" | "disk" — cache dataset
    RESUME = False
    DETERMINISTIC = True  # Ultralytics đặt torch.deterministic + seed reproducible

    # ===================== Optimizer & LR Schedule (Overridden Only) =====================
    OPTIMIZER = "SGD"  # Bộ tối ưu (auto, SGD, Adam, AdamW, RMSprop, ...)
    LR0 = 5e-3       # LR ban đầu (mặc định Ultralytics là 0.01)
    LRF = 0.01        # Hệ số LR cuối cùng (final learning rate factor = lr0 * lrf)
    WARMUP_EPOCHS = 5  # Số epoch warmup (mặc định Ultralytics là 3.0)
    COS_LR = True      # Dùng cosine learning rate decay (mặc định Ultralytics là False)

    # ===================== Loss Function =====================
    # Chọn cls loss cho detection — song song LOSS_FUNCTION của nhánh
    # CWA_TinyImageNet ('cross_entropy' | 'poly_focal').
    # - "bce"  : nn.BCEWithLogitsLoss(reduction="none") — mặc định Ultralytics
    # - "focal": thay bằng FocalBCE (losses.py) — BCE * (1-p_t)^γ * α_factor,
    #            công thức Focal Loss chuẩn, có hook install_cls_loss.
    LOSS_FUNCTION = "focal"
    FOCAL_GAMMA = 1.5  # focusing param — tăng γ dồn học vào hard examples
    FOCAL_ALPHA = 0.25  # balancing param — 0 tắt

    # ===================== Augmentation (Overridden Only) =====================
    MIXUP = 0.15       # Bật nhẹ mixup cho detection (mặc định Ultralytics là 0.0)
    # ⚠ COPY_PASTE CHỈ CÓ TÁC DỤNG VỚI LABEL DẠNG SEGMENTATION.
    # ultralytics.data.augment.CopyPaste.__call__ return luôn khi
    # len(labels["instances"].segments) == 0. Label VOC do Ultralytics convert là
    # bbox-only → giá trị này là NO-OP trên VOC, đặt bao nhiêu cũng không đổi gì.
    # Giữ lại để dùng khi chuyển sang dataset có segment; đặt 0.0 nếu muốn config
    # phản ánh đúng những gì thực sự chạy.
    COPY_PASTE = 0.15

    # ===================== Strategy Configuration =====================
    # Baseline: best.pt — RAW checkpoint có fitness cao nhất trên val'
    # CWA: uniform element-wise average tất cả RAW learnable parameters
    #             của Top-K checkpoint theo raw-model fitness validation.
    #             EMA smoothing được tắt trong train.py để ranking nhất quán.
    USE_CWA = True
    TOP_K_VALUES = [2, 3, 4, 5]
    # Chỉ giữ đúng K checkpoint tốt nhất trên disk: checkpoint mỗi epoch được
    # Ultralytics lưu (save_period=1) rồi TopKCheckpointManager prune NGAY nếu
    # ngoài Top-K — không lưu tất cả epoch (xem train.py)
    KEEP_TOP_K_CHECKPOINTS = 5  # nên = max(TOP_K_VALUES)

    # Baseline lấy từ rank #1 của chính bảng ranking Top-K (raw FP32) thay vì
    # best.pt. Cùng epoch, cùng weights — nhưng best.pt được Ultralytics lưu ở
    # FP16 nên nếu so trực tiếp với bản average FP32 thì hai nhánh lệch
    # precision. True = so sánh apples-to-apples. False = dùng best.pt.
    BASELINE_FROM_RAW_TOPK = True

    # BatchNorm recalibration sau khi average — tương đương update_bn() của
    # torch.optim.swa_utils: sau average, running_mean/running_var giữ nguyên từ
    # ckpt tốt nhất (không average vì đây là population stats) nhưng weights đã
    # đổi → phải LẶP QUA TRAINING DATA ở chế độ forward-only (no_grad, không
    # optimizer) để ước lượng lại BN stats khớp weights mới. Không làm thì mAP
    # của CWA tụt do BN lệch phân phối.
    USE_BN_UPDATE = True
    BN_UPDATE_BATCHES = 100   # số batch forward (giống num_batches=100 repo gốc)
    BN_UPDATE_AUGMENT = True  # True = dataloader mode='train' (đúng phân phối đã
                              # augment mà BN stats gốc được tích lũy trên đó)
    BN_UPDATE_AMP = True      # autocast trên CUDA cho khớp precision lúc train
    # Control quan trọng cho paper: CWA được BN-recalibrate còn Baseline
    # thì không → không thể biết cải thiện đến từ AVERAGING hay từ BN. Bật cờ này
    # để eval thêm dòng "Baseline + BN recal" (baseline đã BN-recalibrate) làm
    # ablation. Tốn thêm 1 lần BN update + 1 lần eval mỗi seed.
    BN_UPDATE_CONTROL = True

    # ===================== Evaluation Configuration =====================
    # Split dùng cho báo cáo cuối (Baseline vs CWA):
    #   "test" (mặc định — VOC2007 test) | "val" (val' holdout) | None (theo data.yaml)
    EVAL_SPLIT = "test"
    CONF = None  # confidence threshold; None = mặc định Ultralytics khi val (0.001)
    IOU = None   # NMS IoU threshold; None = mặc định Ultralytics

    # ===================== Output Configuration =====================
    # Cây thư mục kết quả của `python main.py train --exp-name <NAME>`:
    #
    #   results/detection/<NAME>/
    #   ├── SUMMARY.xlsx            ★ mean ± std của TẤT CẢ seed × strategy
    #   ├── experiment_config.json  snapshot config lúc chạy
    #   ├── charts/                 chart tổng hợp cả experiment (5 hình)
    #   └── seeds/seed_<N>/
    #       ├── results_seed_<N>.xlsx
    #       └── charts/{training, test_<method>}/
    #
    # Không giữ lại bất kỳ checkpoint .pt nào.
    PROJECT = os.path.join("results", "detection")  # thư mục output gốc
    # Truyền qua CLI: --exp-name voc_yolov8s_raw_topk_run01
    # Có EXP_NAME thì pipeline dùng đúng tên này, không ghép timestamp/model.
    EXP_NAME = None
    NAME = None          # prefix legacy khi không truyền EXP_NAME
    EXIST_OK = False
    EXCEL_OUTPUT = None  # chỉ dùng cho `main.py eval/export`; train luôn tự đặt tên
    # Checkpoint chỉ là file tạm để rank/average/eval; Excel/chart được giữ.
    DELETE_CHECKPOINTS_AFTER_RUN = True
    # Dọn run dir mỗi seed xuống còn đúng Excel + charts/
    TIDY_RUN_DIR = True
    KEEP_RESULTS_CSV = False    # nội dung đã nằm nguyên trong sheet PerEpoch
    KEEP_SAMPLE_IMAGES = False  # train_batch*.jpg / val_batch*.jpg (ảnh debug, nặng)
    MAKE_CHARTS = True          # vẽ chart tổng hợp mean ± std ở cuối experiment

    # Random seed: dùng cho cả tách val' (dataset.py) và model.train(seed=...).
    # Hỗ trợ số nguyên đơn lẻ (ví dụ: 42) hoặc danh sách các seed (ví dụ: [42, 100, 2026]).
    RANDOM_SEED = [1, 10, 42, 100, 500]

    # ===================== Edge AI Export (optional) =====================
    # Hook xuất model sau train bằng model.export() — TẮT MẶC ĐỊNH.
    # Bật EXPORT_ENABLED=True (hoặc --export-after-train) khi cần deploy edge.
    EXPORT_ENABLED = False
    EXPORT_FORMAT = "onnx"   # onnx | engine (TensorRT) | openvino | tflite | ...
    EXPORT_HALF = False      # FP16 (hữu ích cho TensorRT/edge)
    EXPORT_IMGSZ = None      # None = dùng IMGSZ phía trên
    EXPORT_DYNAMIC = False
    EXPORT_SIMPLIFY = True
    EXPORT_DEVICE = None     # export TensorRT cần GPU → "0" nếu format engine
    EXTRA_TRAIN_ARGS = {}

    VALID_EVAL_SPLITS = (None, "val", "test", "train")

    @classmethod
    def validate_config(cls, require_model=True):
        """Validate configuration (giữ pattern validate_config của repo gốc)."""
        if require_model and not cls.MODEL:
            raise ValueError(
                "Chưa set model. Đặt Config.MODEL trong config.py "
                "(ví dụ: 'yolov8n.pt', 'yolo11n.pt', 'yolov5nu.pt', .pt/.yaml custom) "
                "hoặc truyền --model khi chạy CLI."
            )

        if not cls.DATA:
            raise ValueError("DATA must not be empty (VOC.yaml hoặc path tới data.yaml custom)")

        if cls.EXP_NAME:
            exp_name = str(cls.EXP_NAME).strip()
            if exp_name in (".", "..") or any(separator in exp_name for separator in ("/", "\\")):
                raise ValueError(
                    "EXP_NAME phải là một tên thư mục đơn, không chứa '/' hoặc '\\'. "
                    "Ví dụ: voc_yolov8s_raw_topk_run01."
                )

        if not 0.0 <= float(cls.VAL_RATIO) < 1.0:
            raise ValueError("VAL_RATIO must be in [0, 1)")

        if int(cls.EPOCHS) <= 0:
            raise ValueError("EPOCHS must be positive")

        if int(cls.IMGSZ) <= 0:
            raise ValueError("IMGSZ must be positive")

        # BATCH = -1 hợp lệ (auto-batch của Ultralytics); chỉ cấm 0
        if int(cls.BATCH) == 0:
            raise ValueError("BATCH must be positive, or -1 for Ultralytics auto-batch")

        if int(cls.WORKERS) < 0:
            raise ValueError("WORKERS must be non-negative")

        if cls.EVAL_WORKERS is not None and int(cls.EVAL_WORKERS) < 0:
            raise ValueError("EVAL_WORKERS must be non-negative, or None to use min(WORKERS, 8)")

        # Cảnh báo RAM HOST (không phải VRAM): mỗi worker của dataloader giữ
        # prefetch_factor(=2) batch ảnh uint8 trong hàng đợi. Đây là thứ khiến
        # job bị OOM killer giết ("Killed" / slurmstepd: oom_kill) chứ không
        # phải CUDA out of memory.
        if int(cls.BATCH) > 0:
            from memory import cpu_quota, training_prefetch_bytes

            workers = int(cls.WORKERS)
            if cls.AUTO_LIMIT_WORKERS:
                workers = min(workers, max(1, cpu_quota() // 2))
            queue_bytes = training_prefetch_bytes(workers, int(cls.BATCH), int(cls.IMGSZ))
            print(
                f"  RAM prefetch (ước lượng): ~{queue_bytes / 1024 ** 3:.1f} GB — "
                f"train loader {workers} worker × batch {cls.BATCH} + "
                f"val loader {workers * 2} worker × batch {cls.BATCH * 2} "
                "(Ultralytics tự nhân đôi cả hai cho val)"
            )
            if queue_bytes > 8 * 1024 ** 3:
                print(
                    "⚠ WARNING: hàng đợi prefetch có thể chiếm "
                    f"~{queue_bytes / 1024 ** 3:.1f} GB RAM host. Xin đủ --mem cho job SLURM, "
                    "hoặc giảm WORKERS / BATCH nếu bị OOM killer."
                )

        if not cls.OPTIMIZER:
            raise ValueError("OPTIMIZER must not be empty")

        if float(cls.WARMUP_EPOCHS) < 0:
            raise ValueError("WARMUP_EPOCHS must be non-negative")

        if not 0.0 <= float(cls.LRF) <= 1.0:
            raise ValueError("LRF must be in [0, 1]")

        for prob in ("MIXUP", "COPY_PASTE"):
            if not 0.0 <= float(getattr(cls, prob)) <= 1.0:
                raise ValueError(f"{prob} must be in [0, 1]")

        if float(cls.COPY_PASTE) > 0 and str(cls.DATA).endswith("VOC.yaml"):
            print(
                f"⚠ WARNING: COPY_PASTE={cls.COPY_PASTE} nhưng label VOC là bbox-only. "
                "Ultralytics chỉ áp dụng copy-paste khi có segmentation mask "
                "(CopyPaste.__call__ return ngay nếu instances.segments rỗng) → "
                "tham số này KHÔNG có tác dụng gì trên VOC."
            )

        if str(cls.LOSS_FUNCTION).lower() not in ("bce", "focal"):
            raise ValueError(
                f"LOSS_FUNCTION={cls.LOSS_FUNCTION!r} không hỗ trợ. "
                "Chọn: 'bce' (Ultralytics default) hoặc 'focal'."
            )
        if float(cls.FOCAL_GAMMA) < 0:
            raise ValueError("FOCAL_GAMMA must be non-negative")
        if not 0.0 <= float(cls.FOCAL_ALPHA) <= 1.0:
            raise ValueError("FOCAL_ALPHA must be in [0, 1]")

        if cls.EVAL_SPLIT not in cls.VALID_EVAL_SPLITS:
            raise ValueError(f"EVAL_SPLIT must be one of {cls.VALID_EVAL_SPLITS}")

        seeds = cls.RANDOM_SEED if isinstance(cls.RANDOM_SEED, (list, tuple)) else [cls.RANDOM_SEED]
        if not seeds:
            raise ValueError("RANDOM_SEED must be an int or a non-empty list of ints")
        if len(set(int(s) for s in seeds)) != len(seeds):
            raise ValueError(f"RANDOM_SEED có seed trùng nhau: {list(seeds)}")

        if cls.USE_CWA:
            if not cls.TOP_K_VALUES or any(int(k) <= 1 for k in cls.TOP_K_VALUES):
                raise ValueError("TOP_K_VALUES must be a non-empty list of ints > 1")
            if int(cls.KEEP_TOP_K_CHECKPOINTS) < max(cls.TOP_K_VALUES):
                raise ValueError(
                    "KEEP_TOP_K_CHECKPOINTS must be >= max(TOP_K_VALUES) "
                    "để đủ checkpoint cho mọi giá trị K"
                )
            if int(cls.EPOCHS) < max(cls.TOP_K_VALUES):
                raise ValueError(
                    f"EPOCHS={cls.EPOCHS} < max(TOP_K_VALUES)={max(cls.TOP_K_VALUES)} "
                    "→ không đủ checkpoint để average"
                )
            if cls.USE_BN_UPDATE and int(cls.BN_UPDATE_BATCHES) <= 0:
                raise ValueError("BN_UPDATE_BATCHES must be positive khi USE_BN_UPDATE=True")
            if float(cls.VAL_RATIO) == 0.0 and str(cls.DATA).endswith("VOC.yaml"):
                print(
                    "⚠ WARNING: USE_CWA=True nhưng VAL_RATIO=0 → không có val "
                    "độc lập, checkpoint sẽ được chọn trên chính tập test (leakage)!"
                )
            if not cls.USE_BN_UPDATE:
                print(
                    "⚠ WARNING: USE_BN_UPDATE=False → sau khi average, BN running stats "
                    "vẫn là của checkpoint tốt nhất trong khi weights đã đổi. "
                    "mAP của CWA nhiều khả năng bị tụt oan."
                )

        if cls.EXPORT_ENABLED and not cls.EXPORT_FORMAT:
            raise ValueError("EXPORT_ENABLED=True requires EXPORT_FORMAT (e.g. onnx, engine)")

        print("[OK] Config validated successfully")
        print(f"  Model : {cls.MODEL or '(chưa set — bắt buộc khi train)'}")
        print(f"  Data  : {cls.DATA} | VAL_RATIO: {cls.VAL_RATIO} (val' tách từ train)")
        print(f"  Epochs: {cls.EPOCHS} | imgsz: {cls.IMGSZ} | batch: {cls.BATCH}")
        from memory import cpu_quota

        train_workers = int(cls.WORKERS)
        eval_workers = (int(cls.EVAL_WORKERS) if cls.EVAL_WORKERS is not None
                        else min(int(cls.WORKERS), 8))
        limited = ""
        if cls.AUTO_LIMIT_WORKERS:
            quota = cpu_quota()
            capped_train = min(train_workers, max(1, quota // 2))
            capped_eval = min(eval_workers, quota)
            if (capped_train, capped_eval) != (train_workers, eval_workers):
                limited = (f"  (đã hạ từ {train_workers}/{eval_workers} "
                           f"theo {quota} CPU job được cấp)")
            train_workers, eval_workers = capped_train, capped_eval
        print(
            f"  Workers: {train_workers} khi train (Ultralytics dùng "
            f"{train_workers * 2} cho val loader) | {eval_workers} khi val/BN recal{limited}"
        )
        print(f"  Seeds : {list(seeds)}")
        print(f"  Task  : Object Detection")
        print(f"  Experiment name: {cls.EXP_NAME or '(auto timestamp)'}")
        print(f"  Loss  : {cls.LOSS_FUNCTION}"
              + (f" (γ={cls.FOCAL_GAMMA}, α={cls.FOCAL_ALPHA})" if cls.LOSS_FUNCTION == 'focal' else ""))
        print(f"  CWA: {'ON — Top-K ' + str(cls.TOP_K_VALUES) if cls.USE_CWA else 'OFF'}")
        if cls.USE_CWA:
            print("    averaging  : uniform element-wise mean của learnable params (KHÔNG EMA)")
            print(
                "    BN recal   : "
                + (f"ON — {cls.BN_UPDATE_BATCHES} batch trên split train, forward-only"
                   if cls.USE_BN_UPDATE else "OFF")
            )
            print(f"    BN control : {'ON (thêm dòng Baseline + BN recal)' if cls.BN_UPDATE_CONTROL else 'OFF'}")
            print(f"    Baseline : {'rank #1 raw FP32' if cls.BASELINE_FROM_RAW_TOPK else 'best.pt (FP16)'}")
        print(
            "  Checkpoints: "
            + ("temporary → delete after evaluation" if cls.DELETE_CHECKPOINTS_AFTER_RUN else "keep")
        )
        print(f"  Eval split: {cls.EVAL_SPLIT or 'mặc định theo data.yaml'}")
