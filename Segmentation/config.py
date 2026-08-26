"""
Configuration file for CWA - Instance Segmentation (Ultralytics YOLO + Carparts)

Nhánh này CHỈ làm instance segmentation với các model YOLO (qua Ultralytics API).
Mọi tham số chỉnh ở đây; các giá trị hay dùng đều override được qua CLI (main.py).
"""
import os


class Config:

    # ===================== Model Configuration =====================
    # KHÔNG chốt cứng version YOLO nào làm mặc định — bạn TỰ SET giá trị này
    # (hoặc truyền --model khi chạy). Nhận mọi giá trị mà ultralytics.YOLO()
    # nhận, đổi version = sửa đúng 1 dòng:
    #   - Pretrained weights : "yolov8s-seg.pt" | "yolo11s-seg.pt" (SEGMENTATION models)
    #   - Custom weights     : "path/to/your_best.pt"
    #   - Train from scratch : "yolov8s-seg.yaml" (hoặc .yaml kiến trúc custom)
    # ⚠️ PHẢI dùng segmentation model (-seg) cho instance segmentation!
    MODEL = "yolov8s-seg.pt"  # <-- ĐẶT MODEL CỦA BẠN Ở ĐÂY, VD: "yolov8n-seg.pt"

    # ===================== Dataset Configuration =====================
    # Dùng YAML được pin ngay trong project để không vô tình lấy YAML cũ
    # đóng gói trong ultralytics==8.3.152.
    # TỰ ĐỘNG DOWNLOAD lần đầu (~133 MB)
    # (https://docs.ultralytics.com/datasets/segment/carparts-seg)
    # Data custom: thay đường dẫn này bằng file data.yaml tương ứng.
    #
    # ✓ Carparts hiện tại: train (3156), val (401), test (276) ảnh
    # ✓ Đã có validation set độc lập → không cần tách val' từ train
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carparts-seg.yaml")
    VAL_RATIO = 0  # Carparts đã có val độc lập → không cần tách
                   # = 0 → dùng nguyên data.yaml gốc (train, val, test rõ ràng)

    # ===================== Training Configuration =====================
    EPOCHS = 100
    IMGSZ = 640
    BATCH = 128     # -1 = auto-batch theo VRAM (chỉ áp dụng khi train)
    DEVICE = None     # None = auto (GPU nếu có); "0" | "0,1" | "cpu"
    # Số dataloader worker. LƯU Ý: Ultralytics dùng `workers * 2` cho val
    # dataloader lúc train (models/yolo/detect/train.py) → set 16 thì val chạy
    # 32 worker và dễ bị "DataLoader worker killed" / hết /dev/shm.
    # CAP_VAL_WORKERS=True buộc val loader dùng đúng WORKERS, không nhân đôi.
    WORKERS = 16
    CAP_VAL_WORKERS = True
    PATIENCE = 10   # Ultralytics early stopping (epoch không cải thiện fitness val)
    PRETRAINED = True
    CACHE = False     # False | "ram" | "disk" — cache dataset
    RESUME = False
    DETERMINISTIC = True  # Ultralytics đặt torch.deterministic + seed reproducible

    # ===================== Optimizer & LR Schedule (Overridden Only) =====================
    # CHỈ override những giá trị bên dưới. `momentum` (0.937), `weight_decay`
    # (5e-4, được Ultralytics scale theo batch), `warmup_momentum` (0.8),
    # `warmup_bias_lr` (0.1), `nbs` (64) GIỮ NGUYÊN mặc định của Ultralytics —
    # build_train_args() không truyền chúng nên trainer tự dùng default.
    OPTIMIZER = "SGD"  # Bộ tối ưu (auto, SGD, Adam, AdamW, RMSprop, ...)
    LR0 = 5e-3       # LR ban đầu (mặc định Ultralytics là 0.01)
    LRF = 0.01        # Hệ số LR cuối cùng (final learning rate factor = lr0 * lrf)
    WARMUP_EPOCHS = 5  # Số epoch warmup (mặc định Ultralytics là 3.0)
    COS_LR = True      # Dùng cosine learning rate decay (mặc định Ultralytics là False)

    # ===================== Loss Function =====================
    # Segmentation loss gồm box + mask + classification + DFL.
    # "bce" = LOSS MẶC ĐỊNH CỦA ULTRALYTICS (v8SegmentationLoss nguyên bản) —
    # dùng cái này cho task segmentation. "focal" chỉ thay BCE ở nhánh
    # classification, không thay box/mask/DFL loss.
    LOSS_FUNCTION = "bce"
    FOCAL_GAMMA = 1.5
    FOCAL_ALPHA = 0.25

    # ===================== Augmentation (Overridden Only) =====================
    MIXUP = 0.0            # mixup không tương thích well với instance segmentation
    COPY_PASTE = 0.1       # copy-paste hữu ích cho segmentation (đặc biệt parts)

    # ===================== Strategy Configuration =====================
    # Baseline: best.pt — RAW checkpoint có fitness cao nhất trên val'
    # CWA: uniform element-wise average RAW weights của Top-K checkpoint
    #             có raw-model fitness cao nhất. train.py tắt EMA để toàn bộ
    #             validation, early stopping và checkpoint selection nhất quán.
    USE_CWA = True
    TOP_K_VALUES = [2, 3, 4, 5]
    # Chỉ giữ đúng K checkpoint tốt nhất trên disk: checkpoint mỗi epoch được
    # Ultralytics lưu (save_period=1) rồi TopKCheckpointManager prune NGAY nếu
    # ngoài Top-K — không lưu tất cả epoch (xem train.py)
    KEEP_TOP_K_CHECKPOINTS = 5  # nên = max(TOP_K_VALUES)

    # Baseline K=1: eval TRỰC TIẾP raw FP32 checkpoint hạng 1 (không average).
    # Cần thiết cho paper vì best.pt của Ultralytics bị strip_optimizer ép về
    # FP16, trong khi CWA average ở FP32 → K=1 raw là điểm so sánh
    # cùng precision, đồng thời là điểm đầu của đường cong mAP theo K.
    EVAL_TOP1_BASELINE = True

    # BatchNorm update sau khi average — bắt chước update_bn() của repo gốc:
    # sau average, running_mean/running_var giữ nguyên từ ckpt tốt nhất (không
    # average vì đây là population stats) nhưng weights đổi → phải chạy
    # forward pass trên train để re-estimate BN stats khớp weights mới,
    # nếu không mAP của CWA sẽ tụt do BN lệch phân phối.
    USE_BN_UPDATE = True
    BN_UPDATE_BATCHES = 100  # giống num_batches=100 của repo gốc
    # Ultralytics tắt mosaic/mixup/cutmix/copy_paste trong `close_mosaic` epoch
    # cuối (mặc định 10). BN phải được ước lượng lại trên ĐÚNG phân phối ảnh của
    # giai đoạn mà Top-K checkpoint được train ra.
    #   "auto" (khuyến nghị) = bám theo epoch của checkpoint hạng 1
    #   True  = luôn tắt mosaic/mixup/cutmix/copy_paste khi update BN
    #   False = luôn dùng full train augmentation
    BN_UPDATE_CLOSE_MOSAIC = "auto"
    # Mixed precision: dùng cho CẢ training (Ultralytics `amp`) lẫn forward pass
    # của BN update. Trên H100 autocast tự chọn bfloat16 (không cần GradScaler
    # tuning) — để True.
    AMP = True

    # ===================== Evaluation Configuration =====================
    # Split dùng cho báo cáo cuối (Baseline vs CWA):
    #   "test" (mặc định — hold-out test set) | "val" (validation set)
    EVAL_SPLIT = "test"
    CONF = None  # confidence threshold; None = mặc định Ultralytics khi val (0.001)
    IOU = None   # NMS IoU threshold; None = mặc định Ultralytics

    # ===================== Output Configuration =====================
    PROJECT = os.path.join("results", "segmentation")  # thư mục output gốc (thay đổi từ detection)
    # Truyền qua CLI: --exp-name carparts_yolov8s_raw_topk
    # Khi có EXP_NAME, pipeline dùng đúng tên này và KHÔNG ghép timestamp/model.
    EXP_NAME = None
    NAME = None          # backward-compatible khi không truyền EXP_NAME
    EXIST_OK = False
    EXCEL_OUTPUT = None  # None = tên mặc định trong run dir (chỉ dùng cho CLI eval/export)
    # Checkpoint chỉ là file tạm: dùng để rank/average/eval rồi xóa sau mỗi seed.
    # CSV, plot, config snapshot và Excel metrics vẫn được giữ.
    DELETE_CHECKPOINTS_AFTER_RUN = True

    # --- Dọn dẹp artefact để thư mục kết quả chỉ còn Excel + chart ---
    # Ultralytics dump ảnh mẫu (train_batch*.jpg, val_batch*.jpg, labels*.jpg)
    # nặng hàng chục MB và không phải "chart" → mặc định xóa.
    KEEP_SAMPLE_IMAGES = False
    # Giữ plot của TỪNG lần model.val() theo strategy (PR/F1/confusion matrix
    # riêng cho best.pt, Top-2 avg, ...). Mặc định tắt cho gọn; các chart tổng
    # hợp trong summary/charts đã đủ để theo dõi.
    SAVE_EVAL_PLOTS = False
    # Sinh chart tổng hợp (mean±std theo strategy, đường cong mAP theo K, ...)
    MAKE_CHARTS = True

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
                "(ví dụ: 'yolov8s-seg.pt', 'yolo11s-seg.pt', .pt/.yaml segmentation custom) "
                "hoặc truyền --model khi chạy CLI."
            )

        if not cls.DATA:
            raise ValueError("DATA must not be empty (carparts-seg.yaml hoặc data.yaml custom)")

        if cls.EXP_NAME:
            exp_name = str(cls.EXP_NAME).strip()
            if exp_name in (".", "..") or any(separator in exp_name for separator in ("/", "\\")):
                raise ValueError(
                    "EXP_NAME phải là một tên thư mục đơn, không chứa '/' hoặc '\\'. "
                    "Ví dụ: carparts_yolov8s_raw_topk."
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

        if not cls.OPTIMIZER:
            raise ValueError("OPTIMIZER must not be empty")

        if float(cls.WARMUP_EPOCHS) < 0:
            raise ValueError("WARMUP_EPOCHS must be non-negative")

        if not 0.0 <= float(cls.LRF) <= 1.0:
            raise ValueError("LRF must be in [0, 1]")

        for prob in ("MIXUP", "COPY_PASTE"):
            if not 0.0 <= float(getattr(cls, prob)) <= 1.0:
                raise ValueError(f"{prob} must be in [0, 1]")

        if str(cls.LOSS_FUNCTION).lower() not in ("bce", "focal"):
            raise ValueError(
                f"LOSS_FUNCTION={cls.LOSS_FUNCTION!r} không hỗ trợ. "
                "Chọn: 'bce' (Ultralytics default) hoặc 'focal' cho nhánh classification."
            )
        if float(cls.FOCAL_GAMMA) < 0:
            raise ValueError("FOCAL_GAMMA must be non-negative")
        if not 0.0 <= float(cls.FOCAL_ALPHA) <= 1.0:
            raise ValueError("FOCAL_ALPHA must be in [0, 1]")
        # Focal chỉ thay classification BCE; box/mask/DFL loss giữ nguyên.

        if cls.EVAL_SPLIT not in cls.VALID_EVAL_SPLITS:
            raise ValueError(f"EVAL_SPLIT must be one of {cls.VALID_EVAL_SPLITS}")

        if cls.USE_CWA:
            if not cls.TOP_K_VALUES or any(int(k) <= 1 for k in cls.TOP_K_VALUES):
                raise ValueError("TOP_K_VALUES must be a non-empty list of ints > 1")
            if int(cls.KEEP_TOP_K_CHECKPOINTS) < max(cls.TOP_K_VALUES):
                raise ValueError(
                    "KEEP_TOP_K_CHECKPOINTS must be >= max(TOP_K_VALUES) "
                    "để đủ checkpoint cho mọi giá trị K"
                )
            if cls.USE_BN_UPDATE:
                if int(cls.BN_UPDATE_BATCHES) <= 0:
                    raise ValueError("BN_UPDATE_BATCHES must be positive when USE_BN_UPDATE=True")
                if not isinstance(cls.BN_UPDATE_CLOSE_MOSAIC, bool) and \
                        str(cls.BN_UPDATE_CLOSE_MOSAIC).lower() != "auto":
                    raise ValueError("BN_UPDATE_CLOSE_MOSAIC must be True, False or 'auto'")
            if float(cls.VAL_RATIO) == 0.0:
                print(
                    "ℹ VAL_RATIO=0: dùng split val gốc để chọn checkpoint. "
                    "Hãy bảo đảm data.yaml có val độc lập với test "
                    "(carparts-seg.yaml mặc định đáp ứng điều này)."
                )

        if cls.EXPORT_ENABLED and not cls.EXPORT_FORMAT:
            raise ValueError("EXPORT_ENABLED=True requires EXPORT_FORMAT (e.g. onnx, engine)")

        print("[OK] Config validated successfully")
        print(f"  Model : {cls.MODEL or '(chưa set — bắt buộc khi train)'}")
        print(f"  Data  : {cls.DATA} | VAL_RATIO: {cls.VAL_RATIO}")
        print(f"  Epochs: {cls.EPOCHS} | imgsz: {cls.IMGSZ} | batch: {cls.BATCH}")
        print(f"  Task  : Instance Segmentation")
        print(f"  Experiment name: {cls.EXP_NAME or '(auto timestamp)'}")
        if cls.USE_CWA:
            print(f"  CWA: ON — Top-K {cls.TOP_K_VALUES} "
                  "(uniform element-wise average của raw weights, KHÔNG EMA)")
            print(f"  Top-1 raw baseline (K=1): {'ON' if cls.EVAL_TOP1_BASELINE else 'OFF'}")
        else:
            print("  CWA: OFF")
        print(
            "  BN update : "
            + (f"ON — {cls.BN_UPDATE_BATCHES} batch train, no-grad forward, "
               f"close_mosaic={cls.BN_UPDATE_CLOSE_MOSAIC}"
               if cls.USE_BN_UPDATE else "OFF")
        )
        print(
            "  Checkpoints: "
            + ("temporary → delete after evaluation" if cls.DELETE_CHECKPOINTS_AFTER_RUN else "keep")
        )
        print(f"  Eval split: {cls.EVAL_SPLIT or 'mặc định theo data.yaml'}")
