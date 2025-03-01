"""
Pytorch Lightning Module for training ScaleProtoSeg segmentation model: Model Grouping Phase
"""

import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import gin
import numpy as np
import torch
from pytorch_lightning import LightningModule

from deeplab_pytorch.libs.utils import PolynomialLR
from helpers import list_of_distances
from segmentation.model.model_multiscale_group import PPNetMultiScale
from segmentation.data.dataset import resize_label
from segmentation.model.loss import (
    CrossEntropyGroup,
    EntropyGroup,
    EntropySpatLoss,
    KLDLossGroup,
    NormLoss,
    PixelWiseCrossEntropyLoss,
    ScaleMax,
)
from segmentation.utils import freezing_batch_norm, get_params, projection_simplex_sort
from settings import log
from train_and_test import group_joint, group_last_only, group_only


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def reset_metrics() -> Dict:
    return {
        "n_correct": 0,
        "n_batches": 0,
        "n_patches": 0,
        "cross_entropy": 0,
        "kld_loss": 0,
        "loss": 0,
        "spat_ent_loss": 0,
        "norm_loss": 0,
        "cross_entropy_group": 0,
        "scale_max_loss": 0,
        "group_ent_loss": 0,
    }


@gin.configurable(denylist=["model_dir", "ppnet", "training_phase", "max_steps"])
class PatchClassificationModuleMultiScale(LightningModule):
    def __init__(
        self,
        model_dir: str,
        ppnet: PPNetMultiScale,
        training_phase: int,
        max_steps: Optional[int] = None,
        poly_lr_power: float = gin.REQUIRED,
        loss_weight_crs_ent: float = gin.REQUIRED,
        loss_weight_l1: float = gin.REQUIRED,
        loss_weight_kld: float = 0.0,
        loss_weight_spatial_entropy: float = 0.0,
        loss_weight_norm: float = 0.0,
        loss_weight_crs_ent_group: float = 0.0,
        loss_weight_scale_max: float = 0.0,
        loss_weight_group_ent: float = 0.0,
        joint_optimizer_lr_features: float = gin.REQUIRED,
        joint_optimizer_lr_add_on_layers: float = gin.REQUIRED,
        joint_optimizer_lr_prototype_vectors: float = gin.REQUIRED,
        joint_optimizer_lr_group_projection: float = gin.REQUIRED,
        joint_optimizer_weight_decay: float = gin.REQUIRED,
        warm_optimizer_lr_group_projection: float = gin.REQUIRED,
        last_layer_optimizer_lr: float = gin.REQUIRED,
        ignore_void_class: bool = False,
        iter_size: int = 1,
        warmup_iters: int = 1000,
        warmup_ratio: float = 0.1,
        joint_no_proto: bool = False,
        joint_last: bool = False,
        freeze_type: str = "all",
    ):
        """Initialize the PatchClassificationModuleMultiScale

        Args:
            model_dir (str): Path to the model directory to save the model and checkpoints.
            ppnet (PPNetMultiScale): Pytorch model to train.
            training_phase (int): Training phase to target specific parts of the model.
            max_steps (int, optional): _description_. Defaults to None.
            poly_lr_power (float, optional): Polynomial power for the learning rate. Defaults to gin.REQUIRED.
            loss_weight_crs_ent (float, optional): Weight of the cross entropy loss in the total loss. Defaults to gin.REQUIRED.
            loss_weight_l1 (float, optional): Weight of the L1 loss in the total loss. Defaults to gin.REQUIRED.
            loss_weight_kld (float, optional): Weight of the KLD loss in the total loss. Defaults to 0.0.
            loss_weight_spatial_entropy (float, optional): Weight of the entropy loss on spatial prototype activations in the total loss. Defaults to 0.0.
            loss_weight_norm (float, optional): Weight of the prototype norm loss in the total loss. Defaults to 0.0.
            loss_weight_crs_ent_group (float, optional): Weight of the cross entropy between group loss in the total loss. Defaults to 0.0.
            loss_weight_scale_max (float, optional): Weight of the loss that maximize per scale prototype activations. Defaults to 0.0.
            loss_weight_group_ent (float, optional): Weight of the entropy loss on the prototype weight in each group. Defaults to 0.0.
            joint_optimizer_lr_features (float, optional): Joint phase learning rate for image encoder features. Defaults to gin.REQUIRED.
            joint_optimizer_lr_add_on_layers (float, optional): Joint phase learning rate for the add on layers. Defaults to gin.REQUIRED.
            joint_optimizer_lr_prototype_vectors (float, optional): Joint phase learning rate for the prototypes. Defaults to gin.REQUIRED.
            joint_optimizer_lr_group_projection (float, optional): Joint phase learning rate for the group projection. Defaults to gin.REQUIRED.
            joint_optimizer_weight_decay (float, optional): Joint phase optimizer weight decay. Defaults to gin.REQUIRED.
            warm_optimizer_lr_group_projection (float, optional): Warm-up phase learning rate for the group projection. Defaults to gin.REQUIRED.
            last_layer_optimizer_lr (float, optional): Finetuning phase last layer learning rate. Defaults to gin.REQUIRED.
            ignore_void_class (bool, optional): Flag to ignore the void class at -1. Defaults to False.
            iter_size (int, optional): Number of accumulation steps for a batch. Defaults to 1.
            warmup_iters (int, optional): DEPRECATED. Defaults to 1000.
            warmup_ratio (float, optional): DEPRECATED. Defaults to 0.1.
            joint_no_proto (bool, optional): Flag to specify if the prototypes should be trained to KEEP False. Defaults to False.
            joint_last (bool, optional): Flag to specify to only train group projection and last layer. Defaults to False.
            freeze_type (str, optional): Freezing type for the batch normalization. Defaults to "all".
        """

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
        self.loss_weight_spatial_entropy = loss_weight_spatial_entropy
        self.loss_weight_norm = loss_weight_norm
        self.loss_weight_crs_ent_group = loss_weight_crs_ent_group
        self.loss_weight_scale_max = loss_weight_scale_max
        self.loss_weight_group_ent = loss_weight_group_ent
        self.joint_optimizer_lr_features = joint_optimizer_lr_features
        self.joint_optimizer_lr_add_on_layers = joint_optimizer_lr_add_on_layers
        self.joint_optimizer_lr_prototype_vectors = joint_optimizer_lr_prototype_vectors
        self.joint_optimizer_lr_group_projection = joint_optimizer_lr_group_projection
        self.joint_optimizer_weight_decay = joint_optimizer_weight_decay
        self.warm_optimizer_lr_group_projection = warm_optimizer_lr_group_projection
        self.last_layer_optimizer_lr = last_layer_optimizer_lr
        self.iter_size = iter_size
        self.loss_weight_kld = loss_weight_kld
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.joint_no_proto = joint_no_proto
        self.joint_last = joint_last
        self.freeze_type = freeze_type

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

        # Set-up the weights depending on the training phase
        if self.training_phase == 0:
            group_only(model=self.ppnet, log=log)
            log(f"WARM-UP GROUP ONLY TRAINING START. ({self.max_steps} steps)")
        elif self.training_phase == 1:
            group_joint(model=self.ppnet, joint_no_proto=joint_no_proto, joint_last=joint_last, log=log)
            log(f"JOINT GROUP ONLY TRAINING START. ({self.max_steps} steps)")
        else:
            group_last_only(model=self.ppnet, log=log)
            log("LAST LAYER GROUP ONLY TRAINING START.")

        self.ppnet.prototype_class_identity = self.ppnet.prototype_class_identity.cuda()
        self.ppnet.group_class_identity = self.ppnet.group_class_identity.cuda()  # TODO: Might not be necessary
        self.lr_scheduler = None
        self.iter_steps = 0
        self.batch_metrics = defaultdict(list)

        # Set-up the loss functions
        self.cross_entropy_func = PixelWiseCrossEntropyLoss(
            ignore_index=-1 if ignore_void_class else None, return_correct=True
        )
        self.kld_loss_func = KLDLossGroup(
            prototype_class_identity=self.ppnet.prototype_class_identity,
            group_class_identity=self.ppnet.group_class_identity,
            num_groups=self.ppnet.num_groups,
        )
        self.ent_loss_func = EntropySpatLoss(prototype_class_identity=self.ppnet.prototype_class_identity)
        self.norm_loss_func = NormLoss(prototype_class_identity=self.ppnet.prototype_class_identity, norm_type="l1")
        self.cross_entropy_func_group = CrossEntropyGroup(self.ppnet)
        self.scale_max_loss_func = ScaleMax(self.ppnet)
        self.group_ent_loss_func = EntropyGroup(self.ppnet)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.ppnet(x)

    def _step(self, split_key: str, batch: Tuple[torch.Tensor, torch.Tensor]):
        """Step function for training, validation and test steps"""

        # Initialize the optimizer at the first step
        optimizer = self.optimizers()
        if split_key == "train" and self.iter_steps == 0:
            optimizer.zero_grad()

        # Log the initial step
        if self.start_step is None:
            self.start_step = self.trainer.global_step
            log(f"INITIAL STEP: {self.global_step}")

        # Freeze the pre-trained batch norm
        freezing_batch_norm(self)

        group_class_identity = self.ppnet.group_class_identity.to(self.device)

        metrics = self.metrics[split_key]

        # Load the batch
        image, mcs_target = batch
        image = image.to(self.device).to(torch.float32)
        mcs_target = mcs_target.cpu().detach().numpy().astype(np.float32)

        # Forward pass
        mcs_model_outputs = self.ppnet.forward(image, return_activations=True, return_distances=True)

        if not isinstance(mcs_model_outputs, list):
            mcs_model_outputs = [mcs_model_outputs]

        # Compute the losses
        mcs_loss, mcs_cross_entropy, mcs_kld_loss, mcs_spat_ent_loss, mcs_norm_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        mcs_cross_entropy_group, mcs_scale_max_loss, mcs_group_ent_loss = 0.0, 0.0, 0.0
        for outputs in mcs_model_outputs:

            output = outputs[0]
            dist_act = outputs[1:]
            target = []

            if len(dist_act) > 1:
                prototype_distances = dist_act[0]
                prototype_activations = dist_act[1]

                list_group_activations = self.ppnet.compute_group(prototype_activations)

                H, W = prototype_distances.shape[-2], prototype_distances.shape[-1]
                prototype_activations = prototype_activations.view(
                    prototype_distances.shape[0], self.ppnet.num_prototypes, -1
                )

            else:
                prototype_distances = dist_act[0]

            for sample_target in mcs_target:
                target.append(resize_label(sample_target, size=(output.shape[2], output.shape[1])).to(self.device))
            target = torch.stack(target, dim=0)  # TODO: This is useless if resize properly implemented.

            # calculate cross entropy loss
            cross_entropy, correct = self.cross_entropy_func(predicted_logits=output, target_labels=target)

            # calculate KLD over class pixels between groups from same class
            kld_loss = self.kld_loss_func(list_group_activation=list_group_activations, target_labels=target)

            # calculate spatial entropy loss
            if self.loss_weight_spatial_entropy > 0:
                spat_ent_loss = self.ent_loss_func(prototype_activations, target)
            else:
                spat_ent_loss = torch.tensor(0.0)

            # calculate norm loss
            if self.loss_weight_norm > 0:
                norm_loss = self.norm_loss_func(prototype_activations, target)
            else:
                norm_loss = torch.tensor(0.0)

            # claculate model loss: group cross-entropy, maximimum prtototype activationns, group entropy
            crs_ent_group = self.cross_entropy_func_group()
            scale_max_loss = self.scale_max_loss_func()
            group_ent_loss = self.group_ent_loss_func()

            # calculate L1 loss
            if hasattr(self.ppnet, "nearest_proto_only") and self.ppnet.nearest_proto_only:
                raise NotImplementedError
            else:
                l1_mask = 1 - torch.t(group_class_identity)

            l1 = (self.ppnet.last_layer_group.weight * l1_mask).norm(p=1)

            # calculate total loss
            loss = (
                self.loss_weight_crs_ent * cross_entropy
                + self.loss_weight_kld * kld_loss
                + self.loss_weight_l1 * l1
                + self.loss_weight_spatial_entropy * spat_ent_loss
                + self.loss_weight_norm * norm_loss
                + self.loss_weight_crs_ent_group * crs_ent_group
                + self.loss_weight_scale_max * scale_max_loss
                + self.loss_weight_group_ent * group_ent_loss
            )

            # accumulate the losses on multiscale outputs
            mcs_loss += loss / len(mcs_model_outputs)
            mcs_cross_entropy += cross_entropy / len(mcs_model_outputs)
            mcs_kld_loss += kld_loss / len(mcs_model_outputs)
            mcs_spat_ent_loss += spat_ent_loss / len(mcs_model_outputs)
            mcs_norm_loss += norm_loss / len(mcs_model_outputs)
            mcs_cross_entropy_group += crs_ent_group / len(mcs_model_outputs)
            mcs_scale_max_loss += scale_max_loss / len(mcs_model_outputs)
            mcs_group_ent_loss += group_ent_loss / len(mcs_model_outputs)

            metrics["n_correct"] += torch.sum(correct)
            metrics["n_patches"] += correct.shape[0]

        # accumulate the losses over the batch
        self.batch_metrics["loss"].append(mcs_loss.item())
        self.batch_metrics["cross_entropy"].append(mcs_cross_entropy.item())
        self.batch_metrics["kld_loss"].append(mcs_kld_loss.item())
        self.batch_metrics["spat_ent_loss"].append(mcs_spat_ent_loss.item())
        self.batch_metrics["norm_loss"].append(mcs_norm_loss.item())
        self.batch_metrics["cross_entropy_group"].append(mcs_cross_entropy_group.item())
        self.batch_metrics["scale_max_loss"].append(mcs_scale_max_loss.item())
        self.batch_metrics["group_ent_loss"].append(mcs_group_ent_loss.item())
        self.iter_steps += 1

        # Backward pass
        if split_key == "train":
            self.manual_backward(mcs_loss / self.iter_size)

            if self.ppnet.incorrect_strength == 0 and self.training_phase == 1:
                self.ppnet.last_layer_group.weight.grad *= torch.t(group_class_identity)

            if self.iter_steps == self.iter_size:
                self.iter_steps = 0
                optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                for group_proj in self.ppnet.group_projection:
                    group_proj.weight.data = projection_simplex_sort(group_proj.weight.data)

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

            metrics["n_batches"] += 1

            self.batch_metrics = defaultdict(list)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args):
        self._step("train", batch)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args):
        self._step("val", batch)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args):
        self._step("test", batch)

    def on_train_epoch_start(self):
        # reset metrics
        for split_key in self.metrics.keys():
            self.metrics[split_key] = reset_metrics()

        # Freeze the pre-trained batch norm
        freezing_batch_norm(self)

    def on_validation_epoch_end(self):
        """On validation epoch end function to save model"""

        val_acc = (self.metrics["val"]["n_correct"] / self.metrics["val"]["n_patches"]).item()

        log(f"Validation Accuracy at on_validation_epoch_end: {val_acc}")
        log("Best Accuracy at this time: " + str(self.best_acc))

        self.log("training_stage", float(self.training_phase))

        if self.training_phase == 0:
            stage_key = "warmup-group"
        elif self.training_phase == 1:
            stage_key = "nopush-group"
        else:
            stage_key = "push-group"

        # Save the model
        torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f"{stage_key}_last.pth"))

        if val_acc > self.best_acc:
            log(f"Saving best model, accuracy: " + str(val_acc))
            self.best_acc = val_acc
            torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f"{stage_key}_best.pth"))

    def _epoch_end(self, split_key: str):
        """Epoch end function for training, validation and test steps"""
        metrics = self.metrics[split_key]
        if len(self.batch_metrics) > 0:
            for key, values in self.batch_metrics.items():
                mean_value = float(np.mean(self.batch_metrics[key]))
                metrics[key] += mean_value
            metrics["n_batches"] += 1

        n_batches = metrics["n_batches"]

        self.batch_metrics = defaultdict(list)

        # Log the metrics and losses
        for key in [
            "loss",
            "cross_entropy",
            "kld_loss",
            "spat_ent_loss",
            "norm_loss",
            "cross_entropy_group",
            "scale_max_loss",
            "group_ent_loss",
        ]:
            self.log(f"{split_key}_{key}", metrics[key] / n_batches)

        acc = metrics["n_correct"] / metrics["n_patches"]
        log(f"{split_key}_accuracy that will be logged {acc}")
        self.log(f"{split_key}_accuracy", metrics["n_correct"] / metrics["n_patches"])
        self.log("l1", self.ppnet.last_layer_group.weight.norm(p=1).item())
        if hasattr(self.ppnet, "nearest_proto_only") and self.ppnet.nearest_proto_only:
            raise NotImplementedError

    def training_epoch_end(self, *args):
        return self._epoch_end("train")

    def validation_epoch_end(self, *args):
        p = self.ppnet.prototype_vectors.view(self.ppnet.num_prototypes, -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = 0.0
            for i in range(self.ppnet.num_scales):
                p_scale = p[self.ppnet.scale_num_prototypes[i][0] : self.ppnet.scale_num_prototypes[i][1]]
                p_avg_pair_dist += torch.mean(list_of_distances(p_scale, p_scale)) / self.ppnet.num_scales
        self.log("avg_dist_proto", p_avg_pair_dist.item())

        return self._epoch_end("val")

    def test_epoch_end(self, *args):
        return self._epoch_end("test")

    def configure_optimizers(self):
        """Set-up the optimizers for the training phases"""
        if self.training_phase == 0:  # warmup
            optimizer_specs = [
                {
                    "params": self.ppnet.group_projection.parameters(),
                    "lr": self.warm_optimizer_lr_group_projection,
                },
            ]
        elif self.training_phase == 1:  # joint

            if self.joint_last:
                optimizer_specs = [
                    {
                        "params": self.ppnet.group_projection.parameters(),
                        "lr": self.joint_optimizer_lr_group_projection,
                    },
                    {"params": self.ppnet.last_layer_group.parameters(), "lr": self.last_layer_optimizer_lr},
                ]

            elif not self.joint_no_proto:
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
                        "params": (
                            list(self.ppnet.add_on_layers.parameters()) + list(self.ppnet.scale_head.parameters())
                            if self.ppnet.scale_head is not None
                            else list(self.ppnet.add_on_layers.parameters())
                        ),
                        "lr": self.joint_optimizer_lr_add_on_layers,
                        "weight_decay": self.joint_optimizer_weight_decay,
                    },
                    {"params": self.ppnet.prototype_vectors, "lr": self.joint_optimizer_lr_prototype_vectors},
                    {
                        "params": self.ppnet.group_projection.parameters(),
                        "lr": self.joint_optimizer_lr_group_projection,
                    },
                ]

            else:  # joint
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
                        "params": (
                            list(self.ppnet.add_on_layers.parameters()) + list(self.ppnet.scale_head.parameters())
                            if self.ppnet.scale_head is not None
                            else list(self.ppnet.add_on_layers.parameters())
                        ),
                        "lr": self.joint_optimizer_lr_add_on_layers,
                        "weight_decay": self.joint_optimizer_weight_decay,
                    },
                    {
                        "params": self.ppnet.group_projection.parameters(),
                        "lr": self.joint_optimizer_lr_group_projection,
                    },
                ]

        else:  # last layer
            optimizer_specs = [{"params": self.ppnet.last_layer_group.parameters(), "lr": self.last_layer_optimizer_lr}]

        optimizer = torch.optim.Adam(optimizer_specs)

        if self.training_phase == 1:
            self.lr_scheduler = PolynomialLR(
                optimizer=optimizer, step_size=1, iter_max=self.max_steps // self.iter_size, power=self.poly_lr_power
            )

        return optimizer
