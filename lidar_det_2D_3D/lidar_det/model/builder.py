from functools import partial
import torch.nn.functional as F

from .nets.minkunet import MinkUNet
from .nets.minkunet_pillar import MinkPillarUNet
from .nets.minkresnet import MinkResNet
from .nets.dr_spaam import DrSpaam

from .losses import (
    SymmetricBCELoss,
    SelfPacedLearningLoss,
    PartiallyHuberisedBCELoss,
)

__all__ = [
    "get_model",
    "MinkUNetDetector",
    "MinkPillarUNetDetector",
    "MinkResNetDetector",
]


def get_model(model_cfg, inference_only=False):
    if model_cfg["type"] == "MinkUNet":
        net = MinkUNetDetector(**model_cfg["kwargs"])
    elif model_cfg["type"] == "MinkPillarUNet":
        net = MinkPillarUNetDetector(**model_cfg["kwargs"])
    elif model_cfg["type"] == "MinkResNet":
        net = MinkResNetDetector(**model_cfg["kwargs"])
    elif model_cfg["type"] == "DrSPAAM":
        net = DrSPAAMDetector(model_cfg)
    else:
        raise RuntimeError(f"Unknown model '{model_cfg['type']}'")
    
    if (not inference_only):
        from .model_fn import plot_p_r_curves, plot_multiple_p_r_curves
        net.plot_p_r_curves = plot_p_r_curves
        net.plot_multiple_p_r_curves = plot_multiple_p_r_curves

    if (not inference_only) and (model_cfg["type"] != "DrSPAAM"):
        from .model_fn import model_fn, model_eval_fn, model_eval_collate_fn, error_fn

        net.model_fn = partial(
            model_fn,
            target_mode=model_cfg["target_mode"],
            disentangled_loss=model_cfg["disentangled_loss"],
        )
        net.model_eval_fn = partial(model_eval_fn, nuscenes=model_cfg["nuscenes"])
        net.model_eval_collate_fn = partial(
            model_eval_collate_fn, nuscenes=model_cfg["nuscenes"]
        )
        net.error_fn = error_fn

    elif (not inference_only) and (model_cfg["type"] == "DrSPAAM"):
        from .dr_spaam_fn import _model_fn, _model_eval_fn, _model_eval_collate_fn, _model_fn_mixup

        net.model_eval_fn = _model_eval_fn
        net.model_eval_collate_fn = _model_eval_collate_fn
        net.model_fn = partial(
            _model_fn, max_num_pts=1000, cls_loss_weight=1.0 - net.mixup_w
        )
        net.model_fn_mixup = partial(
            _model_fn_mixup, max_num_pts=1000, cls_loss_weight=net.mixup_w
        )


    return net


class MinkUNetDetector(MinkUNet):
    def __init__(
        self,
        num_anchors,
        num_ori_bins,
        cr=1.0,
        run_up=True,
        fpn=False,
        num_classes=1,
        input_dim=3,
    ):
        out_dim = _get_num_output_channels(num_ori_bins, num_anchors, num_classes)
        super().__init__(cr=cr, run_up=run_up, num_classes=out_dim, input_dim=input_dim)
        self._na = num_anchors
        self._no = num_ori_bins
        self._nc = num_classes

    @property
    def num_anchors(self):
        return self._na

    @property
    def num_classes(self):
        return self._nc
    
class DrSPAAMDetector(DrSpaam):
    def __init__(
        self,
        model_cfg,
    ):
        if model_cfg["cls_loss_2D"]["type"] == 0:
            cls_loss = F.binary_cross_entropy_with_logits

        elif model_cfg["cls_loss_2D"]["type"] == 1:
            if "kwargs" in model_cfg["cls_loss_2D"]:
                cls_loss = SymmetricBCELoss(**model_cfg["cls_loss_2D"]["kwargs"])
            else:
                cls_loss = SymmetricBCELoss()
        elif model_cfg["cls_loss_2D"]["type"] == 2:
            if "kwargs" in model_cfg["cls_loss_2D"]:
                cls_loss = PartiallyHuberisedBCELoss(**model_cfg["cls_loss_2D"]["kwargs"])
            else:
                cls_loss = PartiallyHuberisedBCELoss()
        else:
            raise NotImplementedError
        
        if model_cfg["self_paced_2D"]:
            cls_loss = SelfPacedLearningLoss(cls_loss)
        super().__init__(
            **model_cfg["kwargs_2D"],
            cls_loss=cls_loss,
            mixup_alpha=model_cfg["mixup_alpha_2D"],
            mixup_w=model_cfg["mixup_w_2D"],
            use_box=True,
            )



class MinkPillarUNetDetector(MinkPillarUNet):
    def __init__(
        self,
        num_anchors,
        num_ori_bins,
        cr=1.0,
        run_up=True,
        fpn=False,
        num_classes=1,
        input_dim=3,
    ):
        out_dim = _get_num_output_channels(num_ori_bins, num_anchors, num_classes)
        super().__init__(cr=cr, run_up=run_up, num_classes=out_dim, input_dim=input_dim)
        self._na = num_anchors
        self._no = num_ori_bins
        self._nc = num_classes

    @property
    def num_anchors(self):
        return self._na

    @property
    def num_classes(self):
        return self._nc


class MinkResNetDetector(MinkResNet):
    def __init__(
        self,
        num_anchors,
        num_ori_bins,
        cr=1.0,
        run_up=True,
        fpn=False,
        num_classes=1,
        input_dim=3,
    ):
        out_dim = _get_num_output_channels(num_ori_bins, num_anchors, num_classes)
        super().__init__(cr=cr, num_classes=out_dim, fpn=fpn, input_dim=input_dim)
        self._na = num_anchors
        self._no = num_ori_bins
        self._nc = num_classes

    @property
    def num_anchors(self):
        return self._na

    @property
    def num_classes(self):
        return self._nc


def _get_num_output_channels(num_ori_bins, num_anchors, num_classes):
    if num_ori_bins > 1:
        # out_dim = num_anchors * (num_ori_bins + 8)
        out_dim = num_anchors * (2 * num_ori_bins + 7)
    else:
        out_dim = num_anchors * 8

    out_dim *= num_classes

    return out_dim
