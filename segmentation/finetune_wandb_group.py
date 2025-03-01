"""
Code for the group mechanism in ScaleProtoSeg
"""

import os
import shutil
from typing import Optional, Union

import argh
import gin
import torch
from dotenv import load_dotenv
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

import wandb
from segmentation.config import get_operative_config_json
from segmentation.data.data_module import PatchClassificationDataModule
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.model.model_multiscale_group import construct_PPNet_Group
from segmentation.model.module_multiscale_group_train import (
    PatchClassificationModuleMultiScale,
)
from segmentation.push_multiscale_optimization import push_prototypes_multiscale
from settings import log

Trainer = gin.external_configurable(Trainer)
load_dotenv()


@gin.configurable(denylist=["config_path", "experiment_name"])
def train(
    config_path: str,
    experiment_name: str,
    wandb_experiment: Optional[str] = None,
    start_checkpoint: str = gin.REQUIRED,
    random_seed: int = gin.REQUIRED,
    early_stopping_patience_last_layer: int = gin.REQUIRED,
    warmup_steps: int = gin.REQUIRED,
    warmup_batch_size: int = gin.REQUIRED,
    joint_steps: int = gin.REQUIRED,
    joint_batch_size: int = gin.REQUIRED,
    finetune_steps: int = gin.REQUIRED,
    push_proto: bool = gin.REQUIRED,
    val_check_interval: Union[int, float] = 1.0,  # For COCO to check validation at higher frequency
):
    """Training script for ScaleProtoSeg grouping mechanism.

    Args:
        config_path (str): Path to the training config.
        experiment_name (str): Name of the experiment.
        wandb_experiment (Optional[str], optional): Name of the wandb experiment to re-start. Defaults to None.
        start_checkpoint (str, optional): Model checkpoint path to resume training (from prototype learning). Defaults to gin.REQUIRED.
        random_seed (int, optional): Random seed for the training. Defaults to gin.REQUIRED.
        early_stopping_patience_last_layer (int, optional):  Number of validation checks for early stopping. Defaults to gin.REQUIRED.
        warmup_steps (int, optional): Number of steps for the warm-up phase. Defaults to gin.REQUIRED.
        warmup_batch_size (int, optional): Batch size for the warm-up phase. Defaults to gin.REQUIRED.
        joint_steps (int, optional): Number of steps for the joint phase. Defaults to gin.REQUIRED.
        joint_batch_size (int, optional): Batch size for the joint phase. Defaults to gin.REQUIRED.
        finetune_steps (int, optional): Number of steps for the finetuning phase. Defaults to gin.REQUIRED.
        push_proto (bool, optional): Flag to specify it is necessary to push the prototypes. Defaults to gin.REQUIRED.
        val_check_interval (Union[int, float], optional): Interval to check the validation sets, 1. means every epoch. Defaults to 1.0.
    """

    seed_everything(random_seed)

    # Initialize model with previous key attributes for the prototype layer.
    results_dir = os.path.join(os.environ["RESULTS_DIR"], experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    log(f'Starting experiment in "{results_dir}" from config {config_path}')

    log(f"Loading checkpoint from {start_checkpoint}")
    old_ppnet = torch.load(start_checkpoint)
    ppnet = construct_PPNet_Group()
    mis_keys, un_keys = ppnet.load_state_dict(old_ppnet.state_dict(), strict=False)
    log(f"Missing keys: {mis_keys}")
    log(f"Unexpected keys: {un_keys}")
    ppnet.scale_num_prototypes = old_ppnet.scale_num_prototypes
    ppnet.prototype_class_identity = old_ppnet.prototype_class_identity

    ppnet._initialize_groups()
    ppnet._initialize_weights()

    for cls_i in range(ppnet.num_classes):
        if ppnet.prototype_class_identity[:, cls_i].sum() == 0:
            print(f"Class {cls_i} has no prototypes")

    ppnet.cuda()

    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(os.path.join(logs_dir, "tb"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "csv"), exist_ok=True)

    tb_logger = TensorBoardLogger(logs_dir, name="tb")
    csv_logger = CSVLogger(logs_dir, name="csv")
    loggers = [tb_logger, csv_logger]

    json_gin_config = get_operative_config_json()

    tb_logger.log_hyperparams(json_gin_config)
    csv_logger.log_hyperparams(json_gin_config)

    use_wandb = bool(int(os.environ["USE_WANDB"]))
    if use_wandb:
        if wandb_experiment is not None:
            log(f"Using wandb experiment {wandb_experiment}")
            wandb_run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                entity=os.environ["WANDB_USER"],
                name=wandb_experiment,
            )
            wandb_logger = WandbLogger(
                id=wandb_run.id,
            )
        else:
            wandb_logger = WandbLogger(
                project=os.environ["WANDB_PROJECT"],
                entity=os.environ["WANDB_USER"],
                tags=[config_path, "segmentation", "protopnet"],
                name=experiment_name,
            )
            loggers.append(wandb_logger)

        wandb_logger.log_hyperparams(json_gin_config)

    shutil.copy(f"segmentation/configs/{config_path}.gin", os.path.join(results_dir, "config.gin"))

    current_epoch = -1
    global_step = -1

    # Warm-up phase for the group projection
    if warmup_steps > 0:

        log("WARM-UP TRAINING")
        data_module = PatchClassificationDataModule(batch_size=warmup_batch_size)
        module = PatchClassificationModuleMultiScale(
            model_dir=results_dir,
            ppnet=ppnet,
            training_phase=0,
            max_steps=warmup_steps,
        )
        trainer = Trainer(
            logger=loggers,
            checkpoint_callback=None,
            enable_progress_bar=True,
            min_steps=1,
            max_steps=warmup_steps,
            val_check_interval=val_check_interval,
        )
        print(warmup_steps)
        print("In warmup")
        trainer.fit(model=module, datamodule=data_module)
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step

    # Join phase for both last layer and group projection
    if joint_steps > 0:

        log(f"JOINT TRAINING")
        module = PatchClassificationModuleMultiScale(
            model_dir=results_dir,
            ppnet=ppnet,
            training_phase=1,
            max_steps=joint_steps,
        )
        data_module = PatchClassificationDataModule(batch_size=joint_batch_size)
        trainer = Trainer(
            logger=loggers,
            checkpoint_callback=None,
            enable_progress_bar=True,
            min_steps=1,
            max_steps=joint_steps + warmup_steps,
            val_check_interval=val_check_interval,
        )
        trainer.fit_loop.current_epoch = current_epoch + 1
        trainer.fit_loop.global_step = global_step + 1
        trainer.fit(model=module, datamodule=data_module)
    else:
        trainer = None  # For the last fine-tuning step

    # Pushing step if necessary (NOT USED)
    if push_proto:

        log(f"PUSH PROTOTYPES")
        module = PatchClassificationModuleMultiScale(
            model_dir=results_dir,
            ppnet=ppnet,
            training_phase=1,
            max_steps=joint_steps,
        )
        module.eval()
        torch.set_grad_enabled(False)

        push_dataset = PatchClassificationDataset(split_key="train", is_eval=True, push_prototypes=True)

        push_prototypes_multiscale(
            push_dataset,
            prototype_network_parallel=ppnet,
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=module.prototypes_dir,
            prototype_img_filename_prefix="prototype-img",
            prototype_self_act_filename_prefix="prototype-self-act",
            proto_bound_boxes_filename_prefix="bb",
            save_prototype_class_identity=True,
            pascal=not push_dataset.only_19_from_cityscapes,
            log=log,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )

        torch.save(obj=ppnet, f=os.path.join(results_dir, f"checkpoints/push_last.pth"))
        torch.save(obj=ppnet, f=os.path.join(results_dir, f"checkpoints/push_best.pth"))

        ppnet = torch.load(os.path.join(results_dir, f"checkpoints/push_last.pth"))

    # Finetuning-phase (MOT used)
    if finetune_steps > 0:
        ppnet = ppnet.cuda()
        log("LAST LAYER FINE-TUNING")
        torch.set_grad_enabled(True)
        callbacks = [EarlyStopping(monitor="val_accuracy", patience=early_stopping_patience_last_layer, mode="max")]
        data_module = PatchClassificationDataModule(batch_size=warmup_batch_size)
        module = PatchClassificationModuleMultiScale(
            model_dir=results_dir,
            ppnet=ppnet,
            training_phase=2,
            max_steps=finetune_steps,
        )
        current_epoch = trainer.current_epoch if trainer is not None else 0  # TODO: Would put -1 here
        global_step = trainer.global_step if trainer is not None else 0
        trainer = Trainer(
            logger=loggers,
            callbacks=callbacks,
            checkpoint_callback=None,
            enable_progress_bar=True,
            max_steps=finetune_steps + joint_steps + warmup_steps,
            val_check_interval=val_check_interval,
        )
        trainer.fit_loop.current_epoch = current_epoch + 1
        trainer.fit_loop.global_step = global_step + 1
        trainer.fit(model=module, datamodule=data_module)

    # Save the final model for COCO
    torch.save(obj=module.ppnet, f=os.path.join(module.checkpoints_dir, f"final-group.pth"))


def load_config_and_train(
    config_path: str,
    experiment_name: str,
):
    gin.parse_config_file(f"segmentation/configs/{config_path}.gin")
    train(
        config_path=config_path,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    argh.dispatch_command(load_config_and_train)
