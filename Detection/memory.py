"""
Giải phóng RAM giữa các bước của pipeline (train → evaluate → seed kế tiếp).

VÌ SAO CẦN FILE NÀY — nguồn gốc lỗi OOM khi chạy nhiều seed trong 1 process:

1. ``InfiniteDataLoader`` của Ultralytics tạo sẵn ``self.iterator`` ngay trong
   ``__init__`` và giữ nó suốt đời object. Với ``workers=N``, PyTorch fork N
   process con NGAY lúc đó và chúng chỉ chết khi iterator bị hủy. Mỗi worker
   còn ôm ``prefetch_factor`` batch trong hàng đợi — với batch=128, imgsz=640
   là ~150 MB/batch, tức hàng GB cho MỖI dataloader còn sống.

2. ``DetMetrics`` mà ``model.val()`` trả về giữ ``on_plot`` = bound method của
   validator → validator → ``validator.dataloader`` → đúng cái iterator ở (1).
   Pipeline lại nhét DetMetrics vào kết quả của từng seed và giữ tới cuối
   experiment ⇒ dataloader của MỌI lượt eval đều không bao giờ được thu hồi.

3. ``trainer`` giữ cả ``train_loader`` lẫn ``test_loader``; nếu object YOLO của
   seed trước còn sống thì hai loader đó cũng còn nguyên worker.

Một seed chạy 6 lượt ``model.val()`` + 5 lượt BN recalibration; nhân với
WORKERS=24 là hơn trăm process con vẫn sống sau seed đầu tiên. Seed thứ hai
dựng dataloader mới cho train → hết RAM, SLURM báo ``oom_kill``.

Các hàm dưới đây đóng dataloader/validator/trainer TƯỜNG MINH thay vì phó mặc
cho garbage collector.
"""
import gc
import multiprocessing
import os


def cpu_quota():
    """
    Số CPU THỰC SỰ được cấp cho process này, không phải của cả node.

    Quan trọng trên SLURM: ``build_dataloader`` của Ultralytics chặn số worker
    bằng ``os.cpu_count()`` — tổng CPU của máy — nên trên node 96 CPU mà job chỉ
    xin 16 thì nó vẫn cho phép 32 worker, và PyTorch (dùng ``sched_getaffinity``)
    lại cảnh báo "This DataLoader will create 32 worker processes in total".

    Lấy min của mọi nguồn tin để không bao giờ ước lượng quá tay.
    """
    candidates = []
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(var, "")
        if value.isdigit() and int(value) > 0:
            candidates.append(int(value))
    total = os.cpu_count()
    if total:
        candidates.append(total)
    return max(1, min(candidates)) if candidates else 1


def dataloader_queue_bytes(workers, batch, imgsz, prefetch_factor=2):
    """
    RAM hàng đợi prefetch của MỘT dataloader: mỗi worker giữ sẵn
    ``prefetch_factor`` batch ảnh uint8 (batch × 3 × imgsz²).
    """
    return int(workers) * int(prefetch_factor) * int(batch) * 3 * int(imgsz) ** 2


def training_prefetch_bytes(workers, batch, imgsz):
    """
    RAM hàng đợi prefetch lúc train — Ultralytics giữ HAI loader song song
    (``BaseTrainer._setup_train``):

        train_loader : workers      worker, batch
        test_loader  : workers × 2  worker, batch × 2

    Cả hai hệ số ×2 đều nằm trong Ultralytics: ``DetectionTrainer.get_dataloader``
    dùng ``self.args.workers * 2`` cho mode='val', còn ``_setup_train`` gọi nó với
    ``batch_size * 2``. Nên val loader tốn gấp ~4 lần train loader.
    """
    return (dataloader_queue_bytes(workers, batch, imgsz)
            + dataloader_queue_bytes(workers * 2, batch * 2, imgsz))


def shutdown_dataloader(loader):
    """
    Đóng ngay worker process của một DataLoader.

    ``InfiniteDataLoader`` giữ iterator ở ``.iterator``, DataLoader thường giữ
    ở ``._iterator``. Gọi ``_shutdown_workers()`` để process con thoát ngay
    thay vì chờ ``__del__``; sau đó bỏ tham chiếu để dataset/queue được thu hồi.
    """
    if loader is None:
        return
    for attr in ("iterator", "_iterator"):
        iterator = getattr(loader, attr, None)
        if iterator is None:
            continue
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # noqa: BLE001 - dọn dẹp best-effort, không được che lỗi chính
                pass
        try:
            setattr(loader, attr, None)
        except Exception:  # noqa: BLE001
            pass
    for attr in ("dataset", "batch_sampler", "sampler"):
        try:
            object.__setattr__(loader, attr, None)
        except Exception:  # noqa: BLE001
            pass


def validator_of(metrics):
    """
    Lấy validator đang bị ``DetMetrics`` giữ qua bound method ``on_plot``.

    Dùng khi không bắt được validator bằng callback ``on_val_end``.
    """
    on_plot = getattr(metrics, "on_plot", None)
    return getattr(on_plot, "__self__", None)


def release_validator(validator):
    """Đóng dataloader của validator rồi bỏ mọi tham chiếu nặng của nó."""
    if validator is None:
        return
    shutdown_dataloader(getattr(validator, "dataloader", None))
    for attr in ("dataloader", "model", "stats", "jdict", "confusion_matrix", "plots", "lb"):
        if hasattr(validator, attr):
            try:
                setattr(validator, attr, None)
            except Exception:  # noqa: BLE001
                pass


def release_trainer(trainer):
    """
    Đóng train_loader/test_loader + validator của trainer và bỏ tham chiếu model.

    Gọi sau khi một seed đã train + evaluate xong, TRƯỚC khi seed kế tiếp dựng
    dataloader mới — nếu không, worker của seed trước vẫn chiếm RAM.
    """
    if trainer is None:
        return
    for attr in ("train_loader", "test_loader"):
        shutdown_dataloader(getattr(trainer, attr, None))
    release_validator(getattr(trainer, "validator", None))
    for attr in (
        "train_loader", "test_loader", "validator", "model", "ema", "optimizer",
        "scaler", "trainset", "testset", "data", "metrics", "loss_items", "lf",
    ):
        if hasattr(trainer, attr):
            try:
                setattr(trainer, attr, None)
            except Exception:  # noqa: BLE001
                pass


def release_yolo(model):
    """Bỏ trainer/predictor/validator mà một object ``ultralytics.YOLO`` đang giữ."""
    if model is None:
        return
    release_trainer(getattr(model, "trainer", None))
    release_validator(getattr(model, "validator", None))
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        shutdown_dataloader(getattr(predictor, "dataset", None))
    for attr in ("trainer", "predictor", "validator", "metrics"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
            except Exception:  # noqa: BLE001
                pass


def free_memory():
    """gc.collect() + trả VRAM cache về driver. Rẻ, gọi sau mỗi bước nặng."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch có thể chưa import được ở nhánh CLI nhẹ
        pass


def rss_bytes():
    """RSS của process hiện tại (byte); None nếu không đọc được (non-Linux)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import resource

        # Linux: ru_maxrss tính bằng KB (là PEAK, không phải hiện tại).
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:  # noqa: BLE001
        return None


def worker_process_count():
    """
    Số process con còn sống do multiprocessing tạo — chính là dataloader worker.

    Sau khi dọn đúng cách, con số này phải quay về ~0 giữa các seed. Nếu nó
    tăng dần thì còn dataloader bị giữ lại ở đâu đó.
    """
    try:
        return len(multiprocessing.active_children())
    except Exception:  # noqa: BLE001
        return -1


def memory_report():
    """Chuỗi ngắn 'RSS x.x GB | n worker' để log giữa các seed."""
    rss = rss_bytes()
    rss_text = "RSS n/a" if rss is None else f"RSS {rss / (1024 ** 3):.1f} GB"
    workers = worker_process_count()
    worker_text = "worker n/a" if workers < 0 else f"{workers} worker process"
    return f"{rss_text} | {worker_text}"
