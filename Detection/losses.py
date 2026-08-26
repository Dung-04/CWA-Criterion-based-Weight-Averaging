"""
Loss functions cho YOLO detection — cho phép swap giữa BCE (mặc định của
Ultralytics) và Focal loss cho phần classification.

Match convention `losses.py` của nhánh CWA_TinyImageNet (nơi expose
LOSS_FUNCTION = 'cross_entropy' | 'poly_focal').

Sử dụng FocalLoss API từ ultralytics.utils.loss thay vì implement thủ công.
"""
import torch.nn.functional as F
from ultralytics.utils.loss import FocalLoss

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


def _unwrap(module):
    """Bỏ lớp bọc DDP/DataParallel/torch.compile để lấy DetectionModel thật."""
    from train import unwrap_ultralytics_model  # import trễ: tránh vòng import

    return unwrap_ultralytics_model(module)


def _install_focal_criterion(module):
    """
    Dựng v8DetectionLoss rồi swap ``.bce`` sang FocalBCE, gán vào
    ``module.criterion``.

    Gán thẳng ``criterion`` chứ KHÔNG monkeypatch ``init_criterion`` trên
    instance, vì hai lý do:

    - ``BaseModel.loss()`` chỉ gọi ``init_criterion()`` khi ``self.criterion``
      còn None, nên đặt sẵn criterion là đủ để mọi lượt forward dùng đúng loss.
    - Gắn hàm/bound-method lên instance sẽ làm HỎNG pickle: ``torch.save`` lưu
      bound method dưới dạng ``(obj, tên_hàm)`` rồi lúc load gọi
      ``getattr(obj, tên_hàm)`` → AttributeError, tức checkpoint không mở lại
      được. ``criterion`` là attribute dữ liệu bình thường, chính Ultralytics
      cũng dùng (``strip_optimizer`` set ``model.criterion = None``).

    Phải gọi khi model ĐÃ ở đúng device: ``v8DetectionLoss.__init__`` chụp lại
    ``next(model.parameters()).device``.
    """
    module = _unwrap(module)
    criterion = module.init_criterion()  # v8DetectionLoss(module) chuẩn Ultralytics
    criterion.bce = FocalBCE(Config.FOCAL_GAMMA, Config.FOCAL_ALPHA)
    module.criterion = criterion
    return criterion


def install_cls_loss(yolo_model):
    """
    Cài cls loss theo Config.LOSS_FUNCTION. Gọi sau `model = YOLO(...)` và
    TRƯỚC `model.train()`.

    - "bce"  : không đụng gì (mặc định của Ultralytics).
    - "focal": thay `criterion.bce` bằng `FocalBCE(FOCAL_GAMMA, FOCAL_ALPHA)`.

    QUAN TRỌNG — vì sao phải dùng callback chứ không đụng thẳng
    ``yolo_model.model``: ``Model.train()`` KHÔNG train trên module đang nằm
    trong wrapper. Nó dựng một DetectionModel MỚI::

        self.trainer.model = self.trainer.get_model(weights=..., cfg=self.model.yaml)

    (ultralytics/engine/model.py). Mọi thứ đã cài lên ``yolo_model.model`` trước
    đó đều bị vứt đi, và training âm thầm chạy bằng BCE mặc định trong khi log
    vẫn báo đã cài FocalBCE.

    Loss vì vậy được cài ở ``on_train_start`` — thời điểm ``_setup_train`` đã
    xong nên model đã nằm đúng device (criterion chụp device lúc khởi tạo).
    ``trainer.ema.ema`` (module dùng để validate) cũng được cài để cột val loss
    trong results.csv cùng thang đo với train loss.
    """
    choice = str(Config.LOSS_FUNCTION).lower()
    if choice == "bce":
        return
    if choice not in ("focal",):
        raise ValueError(
            f"Config.LOSS_FUNCTION={Config.LOSS_FUNCTION!r} không hỗ trợ. "
            "Chọn: 'bce' (Ultralytics default) hoặc 'focal'."
        )

    def _install_on_trainer(trainer):
        module = getattr(trainer, "model", None)
        if module is None:
            raise RuntimeError("trainer.model chưa tồn tại — không cài được FocalBCE.")
        _install_focal_criterion(module)
        ema = getattr(trainer, "ema", None)
        if getattr(ema, "ema", None) is not None:
            _install_focal_criterion(ema.ema)

    yolo_model.add_callback("on_train_start", _install_on_trainer)
    print(
        f"  [Loss] Cls loss: FocalBCE (gamma={Config.FOCAL_GAMMA}, "
        f"alpha={Config.FOCAL_ALPHA}) thay cho BCE mặc định"
    )


def assert_cls_loss_installed(trainer):
    """
    Callback kiểm chứng: xác nhận loss ĐANG DÙNG đúng như Config yêu cầu.

    Đăng ký SAU install_cls_loss trên cùng hook ``on_train_start`` (callback
    chạy theo thứ tự đăng ký). Nếu một bản Ultralytics sau này đổi cách dựng
    model và làm hỏng cơ chế trên, lỗi sẽ nổ ngay đây thay vì âm thầm train cả
    experiment bằng BCE.
    """
    if str(Config.LOSS_FUNCTION).lower() != "focal":
        return
    criterion = getattr(_unwrap(trainer.model), "criterion", None)
    if criterion is None or not isinstance(getattr(criterion, "bce", None), FocalBCE):
        actual = type(getattr(criterion, "bce", None)).__name__ if criterion else "None"
        raise RuntimeError(
            f"Config.LOSS_FUNCTION='focal' nhưng loss thực tế khi train là {actual}. "
            "Cơ chế cài FocalBCE đã hỏng (nhiều khả năng do đổi phiên bản "
            "Ultralytics) — dừng để tránh chạy cả experiment với loss sai."
        )
    print(f"  [Loss] ✓ Xác nhận cls loss khi train: {type(criterion.bce).__name__}")
