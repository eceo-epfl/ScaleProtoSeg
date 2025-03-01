"""
Pytorch Lightning Module for training prototype segmentation model from https://github.com/gmum/proto-segmentation
with some logging modifications.
"""

import os
from collections import defaultdict
from typing import Dict, Optional

import gin
import numpy as np
import torch
import torch.nn.functional as F
from segmentation.model.model import PPNet
from pytorch_lightning import LightningModule

from deeplab_pytorch.libs.utils import PolynomialLR
from helpers import list_of_distances
from segmentation.data.dataset import resize_label
from segmentation.utils import get_params
from settings import log
from train_and_test import joint, last_only, warm_only


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def reset_metrics() -> Dict:
    return {"n_correct": 0, "n_batches": 0, "n_patches": 0, "cross_entropy": 0, "kld_loss": 0, "loss": 0}


# noinspection PyAbstractClass
@gin.configurable(denylist=["model_dir", "ppnet", "training_phase", "max_steps"])
class PatchClassificationModule(LightningModule):
    def __init__(
        self,
        model_dir: str,
        ppnet: PPNet,
        training_phase: int,
        max_steps: Optional[int] = None,
        poly_lr_power: float = gin.REQUIRED,
        loss_weight_crs_ent: float = gin.REQUIRED,
        loss_weight_l1: float = gin.REQUIRED,
        loss_weight_kld: float = 0.0,
        joint_optimizer_lr_features: float = gin.REQUIRED,
        joint_optimizer_lr_add_on_layers: float = gin.REQUIRED,
        joint_optimizer_lr_prototype_vectors: float = gin.REQUIRED,
        joint_optimizer_weight_decay: float = gin.REQUIRED,
        warm_optimizer_lr_add_on_layers: float = gin.REQUIRED,
        warm_optimizer_lr_prototype_vectors: float = gin.REQUIRED,
        warm_optimizer_weight_decay: float = gin.REQUIRED,
        last_layer_optimizer_lr: float = gin.REQUIRED,
        ignore_void_class: bool = False,
        iter_size: int = 1,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.prototypes_dir = os.path.join(model_dir, "prototypes")
        self.checkpoints_dir = os.path.join(model_dir, "checkpoints")
        self.ppnet = ppnet
        self.training_phase = training_phase
        self.max_steps = max_steps
        self.poly_lr_power = poly_lr_power
        self.loss_weight_crs_ent = loss_weight_crs_ent
        self.loss_weight_l1 = loss_weight_l1
        self.joint_optimizer_lr_features = joint_optimizer_lr_features
        self.joint_optimizer_lr_add_on_layers = joint_optimizer_lr_add_on_layers
        self.joint_optimizer_lr_prototype_vectors = joint_optimizer_lr_prototype_vectors
        self.joint_optimizer_weight_decay = joint_optimizer_weight_decay
        self.warm_optimizer_lr_add_on_layers = warm_optimizer_lr_add_on_layers
        self.warm_optimizer_lr_prototype_vectors = warm_optimizer_lr_prototype_vectors
        self.warm_optimizer_weight_decay = warm_optimizer_weight_decay
        self.last_layer_optimizer_lr = last_layer_optimizer_lr
        self.ignore_void_class = ignore_void_class
        self.iter_size = iter_size
        self.loss_weight_kld = loss_weight_kld

        os.makedirs(self.prototypes_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # initialize variables for computing metrics
        self.metrics = {}
        for split_key in ["train", "val", "test", "train_last_layer"]:
            self.metrics[split_key] = reset_metrics()

        # initialize configure_optimizers()
        self.optimizer_defaults = None
        self.start_step = None

        # we use optimizers manually
        self.automatic_optimization = False
        self.best_acc = 0.0

        if self.training_phase == 0:
            warm_only(model=self.ppnet, log=log)
            log(f"WARM-UP TRAINING START. ({self.max_steps} steps)")
        elif self.training_phase == 1:
            joint(model=self.ppnet, log=log)
            log(f"JOINT TRAINING START. ({self.max_steps} steps)")
        else:
            last_only(model=self.ppnet, log=log)
            log("LAST LAYER TRAINING START.")

        self.ppnet.prototype_class_identity = self.ppnet.prototype_class_identity.cuda()
        self.lr_scheduler = None
        self.iter_steps = 0
        self.batch_metrics = defaultdict(list)

    def forward(self, x):
        return self.ppnet(x)

    def _step(self, split_key: str, batch):
        optimizer = self.optimizers()
        if split_key == "train" and self.iter_steps == 0:
            optimizer.zero_grad()

        if self.start_step is None:
            self.start_step = self.trainer.global_step
            log(f"INITIAL STEP: {self.global_step}")

        self.ppnet.features.base.freeze_bn()
        prototype_class_identity = self.ppnet.prototype_class_identity.to(self.device)

        metrics = self.metrics[split_key]

        image, mcs_target = batch

        image = image.to(self.device).to(torch.float32)
        mcs_target = mcs_target.cpu().detach().numpy().astype(np.float32)

        mcs_model_outputs = self.ppnet.forward(image, return_activations=False)
        if not isinstance(mcs_model_outputs, list):
            mcs_model_outputs = [mcs_model_outputs]

        mcs_loss, mcs_cross_entropy, mcs_kld_loss, mcs_cls_act_loss = 0.0, 0.0, 0.0, 0.0
        for output, patch_activations in mcs_model_outputs:
            target = []
            for sample_target in mcs_target:
                target.append(resize_label(sample_target, size=(output.shape[2], output.shape[1])).to(self.device))
            target = torch.stack(target, dim=0)

            # we flatten target/output - classification is done per patch
            output = output.reshape(-1, output.shape[-1])
            target_img = target.reshape(target.shape[0], -1)  # (batch_size, img_size)
            target = target.flatten()

            patch_activations = patch_activations.permute(0, 2, 3, 1)
            patch_activations_img = patch_activations.reshape(
                patch_activations.shape[0], -1, patch_activations.shape[-1]
            )  # (batch_size, img_size, num_proto)

            if self.ignore_void_class:
                # do not predict label for void class (0)
                target_not_void = (target != 0).nonzero().squeeze()
                target = target[target_not_void] - 1
                output = output[target_not_void]

            cross_entropy = torch.nn.functional.cross_entropy(
                output,
                target.long(),
            )

            # calculate KLD over class pixels between prototypes from same class
            kld_loss = []
            for img_i in range(len(target_img)):
                for cls_i in torch.unique(target_img[img_i]).cpu().detach().numpy():
                    if cls_i < 0 or cls_i >= self.ppnet.prototype_class_identity.shape[1]:
                        continue
                    cls_protos = (
                        torch.nonzero(self.ppnet.prototype_class_identity[:, cls_i]).flatten().cpu().detach().numpy()
                    )
                    if len(cls_protos) == 0:
                        continue

                    cls_mask = target_img[img_i] == cls_i

                    log_cls_activations = [
                        torch.masked_select(patch_activations_img[img_i, :, i], cls_mask) for i in cls_protos
                    ]

                    log_cls_activations = [torch.nn.functional.log_softmax(act, dim=0) for act in log_cls_activations]

                    for i in range(len(cls_protos)):
                        if len(cls_protos) < 2 or len(log_cls_activations[0]) < 2:
                            # no distribution over given class
                            continue

                        log_p1_scores = log_cls_activations[i]
                        for j in range(i + 1, len(cls_protos)):
                            log_p2_scores = log_cls_activations[j]

                            # add kld1 and kld2 to make 'symmetrical kld'
                            kld1 = torch.nn.functional.kl_div(
                                log_p1_scores, log_p2_scores, log_target=True, reduction="sum"
                            )
                            kld2 = torch.nn.functional.kl_div(
                                log_p2_scores, log_p1_scores, log_target=True, reduction="sum"
                            )
                            kld = (kld1 + kld2) / 2.0
                            kld_loss.append(kld)

            if len(kld_loss) > 0:
                kld_loss = torch.stack(kld_loss)
                # to make 'loss' (lower == better) take exponent of the negative (maximum value is 1.0, for KLD == 0.0)
                kld_loss = torch.exp(-kld_loss)
                kld_loss = torch.mean(kld_loss)
            else:
                kld_loss = torch.tensor(0.0)

            output_class = torch.argmax(output, dim=-1)
            is_correct = output_class == target

            if hasattr(self.ppnet, "nearest_proto_only") and self.ppnet.nearest_proto_only:
                l1_mask = 1 - torch.eye(self.ppnet.num_classes, device=self.device)
            else:
                l1_mask = 1 - torch.t(prototype_class_identity)

            l1 = (self.ppnet.last_layer.weight * l1_mask).norm(p=1)

            loss = self.loss_weight_crs_ent * cross_entropy + self.loss_weight_kld * kld_loss + self.loss_weight_l1 * l1

            mcs_loss += loss / len(mcs_model_outputs)
            mcs_cross_entropy += cross_entropy / len(mcs_model_outputs)
            mcs_kld_loss += kld_loss / len(mcs_model_outputs)
            metrics["n_correct"] += torch.sum(is_correct)
            metrics["n_patches"] += output.shape[0]

        self.batch_metrics["loss"].append(mcs_loss.item())
        self.batch_metrics["cross_entropy"].append(mcs_cross_entropy.item())
        self.batch_metrics["kld_loss"].append(mcs_kld_loss.item())
        self.iter_steps += 1

        if split_key == "train":
            self.manual_backward(mcs_loss / self.iter_size)

            if self.iter_steps == self.iter_size:
                self.iter_steps = 0
                optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            lr = get_lr(optimizer)
            self.log("lr", lr)

        elif self.iter_steps == self.iter_size:
            self.iter_steps = 0

        if self.iter_steps == 0:
            for key, values in self.batch_metrics.items():
                mean_value = float(np.mean(self.batch_metrics[key]))
                metrics[key] += mean_value
                if key == "loss":
                    self.log("train_loss_step", mean_value, prog_bar=True)
                # print(key, mean_value)
            # print()
            metrics["n_batches"] += 1

            self.batch_metrics = defaultdict(list)

    def training_step(self, batch, batch_idx):
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch)

    def on_train_epoch_start(self):
        # reset metrics
        for split_key in self.metrics.keys():
            self.metrics[split_key] = reset_metrics()

        # Freeze the pre-trained batch norm
        self.ppnet.features.base.freeze_bn()

    def on_validation_epoch_end(self):
        val_acc = (self.metrics["val"]["n_correct"] / self.metrics["val"]["n_patches"]).item()

        self.log("training_stage", float(self.training_phase))

        if self.training_phase == 0:
            stage_key = "warmup"
        elif self.training_phase == 1:
            stage_key = "nopush"
        else:
            stage_key = "push"

        torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f"{stage_key}_last.pth"))

        if val_acc > self.best_acc:
            log(f"Saving best model, accuracy: " + str(val_acc))
            self.best_acc = val_acc
            torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f"{stage_key}_best.pth"))

    def _epoch_end(self, split_key: str):
        metrics = self.metrics[split_key]
        if len(self.batch_metrics) > 0:
            for key, values in self.batch_metrics.items():
                mean_value = float(np.mean(self.batch_metrics[key]))
                metrics[key] += mean_value
            metrics["n_batches"] += 1

        n_batches = metrics["n_batches"]

        self.batch_metrics = defaultdict(list)

        for key in ["loss", "cross_entropy", "kld_loss"]:
            self.log(f"{split_key}_{key}", metrics[key] / n_batches)

        self.log(f"{split_key}_accuracy", metrics["n_correct"] / metrics["n_patches"])
        self.log("l1", self.ppnet.last_layer.weight.norm(p=1).item())
        if hasattr(self.ppnet, "nearest_proto_only") and self.ppnet.nearest_proto_only:
            self.log("gumbel_tau", self.ppnet.gumbel_tau)

    def training_epoch_end(self, step_outputs):
        return self._epoch_end("train")

    def validation_epoch_end(self, step_outputs):
        p = self.ppnet.prototype_vectors.view(self.ppnet.prototype_vectors.shape[0], -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        self.log("avg_dist_proto", p_avg_pair_dist.item())

        return self._epoch_end("val")

    def test_epoch_end(self, step_outputs):
        return self._epoch_end("test")

    def configure_optimizers(self):
        if self.training_phase == 0:  # warmup
            aspp_params = [
                self.ppnet.features.base.aspp.c0.weight,
                self.ppnet.features.base.aspp.c0.bias,
                self.ppnet.features.base.aspp.c1.weight,
                self.ppnet.features.base.aspp.c1.bias,
                self.ppnet.features.base.aspp.c2.weight,
                self.ppnet.features.base.aspp.c2.bias,
                self.ppnet.features.base.aspp.c3.weight,
                self.ppnet.features.base.aspp.c3.bias,
            ]
            optimizer_specs = [
                {
                    "params": list(self.ppnet.add_on_layers.parameters()) + aspp_params,
                    "lr": self.warm_optimizer_lr_add_on_layers,
                    "weight_decay": self.warm_optimizer_weight_decay,
                },
                {"params": self.ppnet.prototype_vectors, "lr": self.warm_optimizer_lr_prototype_vectors},
            ]
        elif self.training_phase == 1:  # joint
            optimizer_specs = [
                {
                    "params": get_params(self.ppnet.features, key="1x"),
                    "lr": self.joint_optimizer_lr_features,
                    "weight_decay": self.joint_optimizer_weight_decay,
                },
                {
                    "params": get_params(self.ppnet.features, key="10x"),
                    "lr": 10 * self.joint_optimizer_lr_features,
                    "weight_decay": self.joint_optimizer_weight_decay,
                },
                {
                    "params": get_params(self.ppnet.features, key="20x"),
                    "lr": 10 * self.joint_optimizer_lr_features,
                    "weight_decay": self.joint_optimizer_weight_decay,
                },
                {
                    "params": self.ppnet.add_on_layers.parameters(),
                    "lr": self.joint_optimizer_lr_add_on_layers,
                    "weight_decay": self.joint_optimizer_weight_decay,
                },
                {"params": self.ppnet.prototype_vectors, "lr": self.joint_optimizer_lr_prototype_vectors},
            ]
        else:  # last layer
            optimizer_specs = [{"params": self.ppnet.last_layer.parameters(), "lr": self.last_layer_optimizer_lr}]

        optimizer = torch.optim.Adam(optimizer_specs)

        if self.training_phase == 1:
            self.lr_scheduler = PolynomialLR(
                optimizer=optimizer, step_size=1, iter_max=self.max_steps // self.iter_size, power=self.poly_lr_power
            )

        return optimizer
