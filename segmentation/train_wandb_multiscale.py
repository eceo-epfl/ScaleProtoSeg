"""
Code for the prototype learning in ScaleProtoSeg
"""

import os
import shutil
from typing import Optional, Union

import argh
import gin
import torch
import torchvision
from dotenv import load_dotenv
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

import wandb
from segmentation.config import get_operative_config_json
from segmentation.data.data_module import PatchClassificationDataModule
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.model.deeplab_features import (
    torchvision_resnet_weight_key_to_deeplab2,
)
from segmentation.model.model_multiscale import construct_PPNet
from segmentation.model.module_multiscale import PatchClassificationModuleMultiScale
# Change to from segmentation.em.module_multiscale import PatchClassificationModuleMultiScale for EM dataset
from segmentation.push_multiscale_optimization import push_prototypes_multiscale
from settings import log

Trainer = gin.external_configurable(Trainer)
load_dotenv()


@gin.configurable(denylist=["config_path", "experiment_name", "wandb_experiment", "pruned"])
def train(
    config_path: str,
    experiment_name: str,
    wandb_experiment: Optional[str] = None,
    pruned: bool = False,
    start_checkpoint: str = gin.REQUIRED,
    random_seed: int = gin.REQUIRED,
    early_stopping_patience_last_layer: int = gin.REQUIRED,
    warmup_steps: int = gin.REQUIRED,
    joint_steps: int = gin.REQUIRED,
    finetune_steps: int = gin.REQUIRED,
    warmup_batch_size: int = gin.REQUIRED,
    joint_batch_size: int = gin.REQUIRED,
    data_type: str = gin.REQUIRED,
    load_coco: bool = False,
    load_coco_v2: bool = False,
    is_resnet: bool = True,
    val_check_interval: Union[int, float] = 1.0,  # For COCO to check validation at higher frequency
):
    """Training script for ScaleProtoSeg prototype learning.

    Args:
        config_path (str): Path to the training config.
        experiment_name (str): Name of the experiment.
        wandb_experiment (Optional[str], optional): Name of the wandb experiment to re-start. Defaults to None.
        pruned (bool, optional): Flag if fine-tuning a pruned model. Defaults to False.
        start_checkpoint (str, optional): Model checkpoint path to resume training. Defaults to gin.REQUIRED.
        random_seed (int, optional): Random seed for the training. Defaults to gin.REQUIRED.
        early_stopping_patience_last_layer (int, optional): Number of validation checks for early stopping. Defaults to gin.REQUIRED.
        warmup_steps (int, optional): Number of steps for the warm-up phase. Defaults to gin.REQUIRED.
        joint_steps (int, optional): Number of steps for the joint phase. Defaults to gin.REQUIRED.
        finetune_steps (int, optional): Number of steps for the finetuning phase. Defaults to gin.REQUIRED.
        warmup_batch_size (int, optional): Batch size for the warm-up phase. Defaults to gin.REQUIRED.
        joint_batch_size (int, optional): Batch size for the joint phase. Defaults to gin.REQUIRED.
        data_type (str, optional): Data type for the pushing mechanism. Defaults to gin.REQUIRED.
        load_coco (bool, optional): Flag to load coco weights from caffe. Defaults to False.
        load_coco_v2 (bool, optional): Flag to load coco weights from pytorch. Defaults to False.
        is_resnet (bool, optional): Flag to specify if the architecture uses a ResNet image encoder. Defaults to True.
        val_check_interval (Union[int, float], optional): Interval to check the validation sets, 1. means every epoch. Defaults to 1.0.
    """

    seed_everything(random_seed)

    results_dir = os.path.join(os.environ["RESULTS_DIR"], experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    log(f'Starting experiment in "{results_dir}" from config {config_path}')

    last_checkpoint = os.path.join(results_dir, "checkpoints", "nopush_best.pth")

    if start_checkpoint:
        log(f"Loading checkpoint from {start_checkpoint}")
        ppnet = torch.load(start_checkpoint)
        pre_loaded = True
    elif wandb_experiment is not None and os.path.exists(last_checkpoint):
        log(f"Loading last model from {last_checkpoint}")
        ppnet = torch.load(last_checkpoint)
        pre_loaded = True
    else:
        pre_loaded = False
        ppnet = construct_PPNet()

    if not pre_loaded and is_resnet:
        if load_coco:
            log("Loading COCO pretrained weights")
            state_dict = torch.load(
                "deeplab_pytorch/data/models/coco/deeplabv1_resnet101/" "caffemodel/deeplabv1_resnet101-coco.pth"
            )
            load_result = ppnet.features.base.load_state_dict(state_dict, strict=False)
            log(f"Loaded {len(state_dict)} weights from pretrained COCO")

            assert len(load_result.missing_keys) == 8  # ASPP layer (has different shape)
            assert len(load_result.unexpected_keys) == 2  # final FC for COCO

        elif load_coco_v2:
            log("Loading COCO v2 pretrained weights")
            state_dict = torch.load(
                "deeplab_pytorch/data/models/coco/deeplabv2_resnet101/"
                "pytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth"
            )

            # Filter out parameters with mismatched shapes
            filtered_state_dict = {}
            for name, param in state_dict.items():
                if name in ppnet.features.state_dict() and param.shape == ppnet.features.state_dict()[name].shape:
                    filtered_state_dict[name] = param
                else:
                    log(
                        f"Skipping parameter {name} due to shape mismatch: {param.shape} vs {ppnet.features.state_dict()[name].shape}"
                    )

            load_result = ppnet.features.load_state_dict(filtered_state_dict, strict=False)
            log(f"Loaded {len(filtered_state_dict)} weights from pretrained COCO v2")

            # Logging missing and unknown keys for validation.
            log(str(load_result.missing_keys))
            log(str(load_result.unexpected_keys))

        else:
            # load weights from Resnet pretrained on ImageNet
            resnet_state_dict = torchvision.models.resnet101(pretrained=True).state_dict()
            new_state_dict = {}
            for k, v in resnet_state_dict.items():
                new_key = torchvision_resnet_weight_key_to_deeplab2(k)
                if new_key is not None:
                    new_state_dict[new_key] = v

            load_result = ppnet.features.base.load_state_dict(new_state_dict, strict=False)
            log(f"Loaded {len(new_state_dict)} weights from pretrained ResNet101")

            assert len(load_result.unexpected_keys) == 0

        log(str(load_result))

    elif not pre_loaded and not is_resnet:
        log("No loading, using VGG 16 | U-Net with default kaiming initialization")

    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(os.path.join(logs_dir, "tb"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "csv"), exist_ok=True)

    tb_logger = TensorBoardLogger(logs_dir, name="tb")
    csv_logger = CSVLogger(logs_dir, name="csv")
    loggers = [tb_logger, csv_logger]

    json_gin_config = get_operative_config_json()

    tb_logger.log_hyperparams(json_gin_config)
    csv_logger.log_hyperparams(json_gin_config)

    # Post-pruning the finetuning of the last layer is done differently
    if not pruned:
        use_wandb = bool(int(os.environ["USE_WANDB"]))
        if use_wandb:
            if wandb_experiment is not None:
                wandb_run = wandb.init(
                    project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_USER"], name=wandb_experiment
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

        # Warm-up phase
        if warmup_steps > 0:
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
            trainer.fit(model=module, datamodule=data_module)
            current_epoch = trainer.current_epoch
            global_step = trainer.global_step
        else:
            current_epoch = -1
            global_step = -1

        last_checkpoint = os.path.join(results_dir, "checkpoints/warmup_last.pth")
        if os.path.exists(last_checkpoint):
            log(f"Loading model after warmup from {last_checkpoint}")
            ppnet = torch.load(last_checkpoint)
            ppnet = ppnet.cuda()

        module = PatchClassificationModuleMultiScale(
            model_dir=results_dir, ppnet=ppnet, training_phase=1, max_steps=joint_steps
        )

        # Joint phase
        if joint_steps > 0:
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
            trainer = None

        # Pushing mechanism
        log("SAVING PROTOTYPES")
        last_checkpoint = os.path.join(results_dir, "checkpoints/nopush_last.pth")
        if os.path.exists(last_checkpoint):
            log(f"Loading model after joint training from {last_checkpoint}")
            ppnet = torch.load(last_checkpoint)
            ppnet = ppnet.cuda()

        ppnet = ppnet.cuda()
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
            data_type=data_type,
            log=log,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )

        torch.save(obj=ppnet, f=os.path.join(results_dir, f"checkpoints/push_last.pth"))
        torch.save(obj=ppnet, f=os.path.join(results_dir, f"checkpoints/push_best.pth"))

        ppnet = torch.load(os.path.join(results_dir, f"checkpoints/push_last.pth"))
        ppnet = ppnet.cuda()

    # Pruning use-case
    else:
        best_checkpoint = os.path.join(results_dir, "pruned/pruned.pth")
        log(f"Loading pruned model from {best_checkpoint}")
        ppnet = torch.load(best_checkpoint)
        ppnet = ppnet.cuda()
        trainer = None

        use_wandb = bool(int(os.environ["USE_WANDB"]))
        if use_wandb:
            wandb_logger = WandbLogger(
                project=os.environ["WANDB_PROJECT"],
                entity=os.environ["WANDB_USER"],
                tags=[config_path, "patch_classification", "protopnet", "pruned"],
                name=experiment_name,
            )
            loggers.append(wandb_logger)
            wandb_logger.log_hyperparams(json_gin_config)

    # Finetuning phase
    log("LAST LAYER FINE-TUNING")
    torch.set_grad_enabled(True)
    callbacks = [EarlyStopping(monitor="val_accuracy", patience=early_stopping_patience_last_layer, mode="max")]
    data_module = PatchClassificationDataModule(batch_size=warmup_batch_size)
    module = PatchClassificationModuleMultiScale(
        model_dir=os.path.join(results_dir, "pruned") if pruned else results_dir,
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
        max_steps=finetune_steps + joint_steps + warmup_steps if not pruned else finetune_steps,
        val_check_interval=val_check_interval,
    )
    trainer.fit_loop.current_epoch = current_epoch + 1
    trainer.fit_loop.global_step = global_step + 1
    trainer.fit(model=module, datamodule=data_module)

    # Save the final model for COCO
    torch.save(obj=module.ppnet, f=os.path.join(module.checkpoints_dir, f"push_final.pth"))


def load_config_and_train(
    config_path: str,
    experiment_name: str,
    wandb_experiment: Optional[str] = None,
    pruned: bool = False,
):
    gin.parse_config_file(f"segmentation/configs/{config_path}.gin")
    train(
        config_path=config_path,
        experiment_name=experiment_name,
        pruned=pruned,
        wandb_experiment=wandb_experiment,
    )


if __name__ == "__main__":
    argh.dispatch_command(load_config_and_train)
