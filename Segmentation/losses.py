"""
Loss functions cho YOLO — cho phép swap giữa BCE (mặc định của Ultralytics)
và Focal loss cho phần classification.

``v8SegmentationLoss`` kế thừa classification BCE của detection loss, nên focal
có thể thay đúng nhánh classification mà không thay box/mask/DFL loss.

Match convention `losses.py` của nhánh CWA_TinyImageNet (nơi expose
LOSS_FUNCTION = 'cross_entropy' | 'poly_focal').

Sử dụng FocalLoss API từ ultralytics.utils.loss thay vì implement thủ công.
"""
from ultralytics.utils.loss import FocalLoss
import torch.nn.functional as F
from config import Config


class FocalBCE(FocalLoss):
    """
    Focal BCE loss giữ shape output (bs, num_anchors, nc) — drop-in replacement
    cho ``nn.BCEWithLogitsLoss(reduction="none")`` mà ``v8DetectionLoss.__call__``
    sử dụng (line: ``self.bce(pred_scores, target_scores).sum() / target_scores_sum``).

    Kế thừa từ ``ultralytics.utils.loss.FocalLoss`` nhưng override ``forward()``
    để trả element-wise loss thay vì scalar (FocalLoss gốc trả
    ``loss.mean(1).sum()`` → không thể thay ``self.bce`` được).

    Args:
        gamma: focusing parameter — tăng γ → dồn học vào hard examples.
        alpha: balancing parameter — 0 tắt (dùng α cân giữa fg/bg như paper).
    """

    def forward(self, pred, label):
        """Compute focal BCE loss, trả element-wise (cùng shape với BCE reduction='none')."""

        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1.0 - label) * (1.0 - pred_prob)
        loss = loss * (1.0 - p_t).pow(self.gamma)
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1.0 - label) * (1.0 - self.alpha)
            loss = loss * alpha_factor
        return loss  # (bs, num_anchors, nc) — cùng shape với BCE(reduction='none')


def _patch_module_criterion(module):
    """Bọc ``init_criterion`` của một BaseModel để thay BCE bằng FocalBCE."""
    original_init_criterion = module.init_criterion

    def patched_init_criterion():
        criterion = original_init_criterion()  # v8SegmentationLoss(self)
        criterion.bce = FocalBCE(Config.FOCAL_GAMMA, Config.FOCAL_ALPHA)
        return criterion

    module.init_criterion = patched_init_criterion
    # criterion có thể đã được tạo lazily trước đó → ép tạo lại ở lần loss() sau.
    if getattr(module, "criterion", None) is not None:
        module.criterion = None


def install_cls_loss(yolo_model):
    """
    Dùng loss theo Config.LOSS_FUNCTION cho nhánh classification.

    - "bce" : không đụng gì (mặc định của Ultralytics).
    - "focal": thay `criterion.bce` bằng `FocalBCE(FOCAL_GAMMA, FOCAL_ALPHA)`.

    Gọi sau `model = YOLO(...)` và TRƯỚC `model.train()`.

    QUAN TRỌNG — vì sao phải dùng callback:
    ``Model.train()`` của Ultralytics KHÔNG train trên object model hiện tại;
    nó tạo model mới bằng ``trainer.get_model(weights=..., cfg=self.model.yaml)``
    rồi gán ``self.trainer.model``. Nếu chỉ patch ``yolo_model.model`` tại đây
    thì bản vá bị vứt bỏ ngay khi train bắt đầu và focal loss âm thầm không có
    tác dụng. Do đó patch được đăng ký thêm ở callback ``on_train_start``, lúc
    ``trainer.model`` đã là model thật sự được train và criterion vẫn chưa được
    khởi tạo (``BaseModel.loss`` tạo criterion lazily ở batch đầu tiên).
    """
    task = getattr(yolo_model.model, "task", None) or getattr(yolo_model, "task", None)

    choice = str(Config.LOSS_FUNCTION).lower()
    if choice == "bce":
        return
    if choice not in ("focal",):
        raise ValueError(
            f"Config.LOSS_FUNCTION={Config.LOSS_FUNCTION!r} không hỗ trợ. "
            "Chọn: 'bce' (Ultralytics default) hoặc 'focal'."
        )

    # Patch model hiện tại (phục vụ model.val() / predict trực tiếp)...
    _patch_module_criterion(yolo_model.model)

    # ...và model thật sự do trainer dựng lại khi model.train() chạy.
    def _on_train_start(trainer):
        from train import unwrap_ultralytics_model

        _patch_module_criterion(unwrap_ultralytics_model(trainer.model))
        print(f"  [Loss] FocalBCE đã được gắn vào trainer.model (task={task or 'YOLO'})")

    yolo_model.add_callback("on_train_start", _on_train_start)
    print(
        f"  [Loss] {task or 'YOLO'} classification: FocalBCE (gamma={Config.FOCAL_GAMMA}, "
        f"alpha={Config.FOCAL_ALPHA}) thay cho BCE mặc định"
    )
