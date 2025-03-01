"""
Original code for the training from https://github.com/gmum/proto-segmentation.

NO Modifications.
"""
import os
import shutil
from typing import Optional

import argh
import torch
import neptune.new as neptune
import torchvision
from pytorch_lightning import Trainer, seed_everything
import gin
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, CSVLogger

from segmentation.data.data_module import PatchClassificationDataModule
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.model.module import PatchClassificationModule
from segmentation.config import get_operative_config_json
from segmentation.model.model import construct_PPNet
from segmentation.push import push_prototypes
from settings import log
from segmentation.model.deeplab_features import torchvision_resnet_weight_key_to_deeplab2


Trainer = gin.external_configurable(Trainer)


@gin.configurable(denylist=['config_path', 'experiment_name', 'neptune_experiment', 'pruned'])
def train(
        config_path: str,
        experiment_name: str,
        neptune_experiment: Optional[str] = None,
        pruned: bool = False,
        start_checkpoint: str = '',
        random_seed: int = gin.REQUIRED,
        early_stopping_patience_last_layer: int = gin.REQUIRED,
        warmup_steps: int = gin.REQUIRED,
        joint_steps: int = gin.REQUIRED,
        finetune_steps: int = gin.REQUIRED,
        warmup_batch_size: int = gin.REQUIRED,
        joint_batch_size: int = gin.REQUIRED,
        load_coco: bool = False
):
    seed_everything(random_seed)

    results_dir = os.path.join(os.environ['RESULTS_DIR'], experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    log(f'Starting experiment in "{results_dir}" from config {config_path}')

    last_checkpoint = os.path.join(results_dir, 'checkpoints', 'nopush_best.pth')

    if start_checkpoint:
        log(f'Loading checkpoint from {start_checkpoint}')
        ppnet = torch.load(start_checkpoint)
        pre_loaded = True
    elif neptune_experiment is not None and os.path.exists(last_checkpoint):
        log(f'Loading last model from {last_checkpoint}')
        ppnet = torch.load(last_checkpoint)
        pre_loaded = True
    else:
        pre_loaded = False
        ppnet = construct_PPNet()

    if not pre_loaded:
        if load_coco:
            log('Loading COCO pretrained weights')
            state_dict = torch.load('deeplab_pytorch/data/models/coco/deeplabv1_resnet101/'
                                    'caffemodel/deeplabv1_resnet101-coco.pth')
            load_result = ppnet.features.base.load_state_dict(state_dict, strict=False)
            log(f'Loaded {len(state_dict)} weights from pretrained COCO')

            assert len(load_result.missing_keys) == 8  # ASPP layer (has different shape)
            assert len(load_result.unexpected_keys) == 2  # final FC for COCO
        else:
            # load weights from Resnet pretrained on ImageNet
            resnet_state_dict = torchvision.models.resnet101(pretrained=True).state_dict()
            new_state_dict = {}
            for k, v in resnet_state_dict.items():
                new_key = torchvision_resnet_weight_key_to_deeplab2(k)
                if new_key is not None:
                    new_state_dict[new_key] = v

            load_result = ppnet.features.base.load_state_dict(new_state_dict, strict=False)
            log(f'Loaded {len(new_state_dict)} weights from pretrained ResNet101')

            assert len(load_result.missing_keys) == 8  # ASPP layer (has different shape)
            assert len(load_result.unexpected_keys) == 0

        log(str(load_result))

    logs_dir = os.path.join(results_dir, 'logs')
    os.makedirs(os.path.join(logs_dir, 'tb'), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, 'csv'), exist_ok=True)

    tb_logger = TensorBoardLogger(logs_dir, name='tb')
    csv_logger = CSVLogger(logs_dir, name='csv')
    loggers = [tb_logger, csv_logger]

    json_gin_config = get_operative_config_json()

    tb_logger.log_hyperparams(json_gin_config)
    csv_logger.log_hyperparams(json_gin_config)

    if not pruned:
        use_neptune = bool(int(os.environ['USE_NEPTUNE']))
        if use_neptune:
            if neptune_experiment is not None:
                neptune_run = neptune.init(
                    project=os.environ['NEPTUNE_PROJECT'],
                    run=neptune_experiment
                )
                neptune_logger = NeptuneLogger(
                    run=neptune_run
                )
            else:
                neptune_logger = NeptuneLogger(
                    project=os.environ['NEPTUNE_PROJECT'],
                    tags=[config_path, 'segmentation', 'protopnet'],
                    name=experiment_name
                )
                loggers.append(neptune_logger)

            neptune_run = neptune_logger.run
            neptune_run['config_file'].upload(f'segmentation/configs/{config_path}.gin')
            neptune_run['config'] = json_gin_config

        shutil.copy(f'segmentation/configs/{config_path}.gin', os.path.join(results_dir, 'config.gin'))

        if warmup_steps > 0:
            data_module = PatchClassificationDataModule(batch_size=warmup_batch_size)
            module = PatchClassificationModule(
                model_dir=results_dir,
                ppnet=ppnet,
                training_phase=0,
                max_steps=warmup_steps,
            )
            trainer = Trainer(logger=loggers, checkpoint_callback=None, enable_progress_bar=False,
                              min_steps=1, max_steps=warmup_steps)
            trainer.fit(model=module, datamodule=data_module)
            current_epoch = trainer.current_epoch
        else:
            current_epoch = -1

        last_checkpoint = os.path.join(results_dir, 'checkpoints/warmup_last.pth')
        if os.path.exists(last_checkpoint):
            log(f'Loading model after warmup from {last_checkpoint}')
            ppnet = torch.load(last_checkpoint)
            ppnet = ppnet.cuda()

        data_module = PatchClassificationDataModule(batch_size=joint_batch_size)
        module = PatchClassificationModule(
            model_dir=results_dir,
            ppnet=ppnet,
            training_phase=1,
            max_steps=joint_steps
        )
        trainer = Trainer(logger=loggers, checkpoint_callback=None, enable_progress_bar=False,
                          min_steps=1, max_steps=joint_steps)
        trainer.fit_loop.current_epoch = current_epoch + 1
        trainer.fit(model=module, datamodule=data_module)

        log('SAVING PROTOTYPES')
        ppnet = ppnet.cuda()
        module.eval()
        torch.set_grad_enabled(False)

        push_dataset = PatchClassificationDataset(
            split_key='train',
            is_eval=True,
            push_prototypes=True
        )

        push_prototypes(
            push_dataset,
            prototype_network_parallel=ppnet,
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=module.prototypes_dir,
            prototype_img_filename_prefix='prototype-img',
            prototype_self_act_filename_prefix='prototype-self-act',
            proto_bound_boxes_filename_prefix='bb',
            save_prototype_class_identity=True,
            pascal=not push_dataset.only_19_from_cityscapes,
            log=log
        )

        torch.save(obj=ppnet, f=os.path.join(results_dir, f'checkpoints/push_last.pth'))
        torch.save(obj=ppnet, f=os.path.join(results_dir, f'checkpoints/push_best.pth'))

        ppnet = torch.load(os.path.join(results_dir, f'checkpoints/push_last.pth'))
        ppnet = ppnet.cuda()
    else:
        best_checkpoint = os.path.join(results_dir, 'pruned/pruned.pth')
        log(f'Loading pruned model from {best_checkpoint}')
        ppnet = torch.load(best_checkpoint)
        ppnet = ppnet.cuda()
        trainer = None

        use_neptune = bool(int(os.environ['USE_NEPTUNE']))
        if use_neptune:
            neptune_logger = NeptuneLogger(
                project=os.environ['NEPTUNE_PROJECT'],
                tags=[config_path, 'patch_classification', 'protopnet', 'pruned'],
                name=f'{experiment_name}_pruned' if pruned else experiment_name
            )
            loggers.append(neptune_logger)

            neptune_run = neptune_logger.run
            neptune_run['config_file'].upload(f'segmentation/configs/{config_path}.gin')
            neptune_run['config'] = json_gin_config

    log('LAST LAYER FINE-TUNING')
    torch.set_grad_enabled(True)
    callbacks = [
        EarlyStopping(monitor='val/accuracy', patience=early_stopping_patience_last_layer, mode='max')
    ]
    data_module = PatchClassificationDataModule(batch_size=warmup_batch_size)
    module = PatchClassificationModule(
        model_dir=os.path.join(results_dir, 'pruned') if pruned else results_dir,
        ppnet=ppnet,
        training_phase=2,
        max_steps=finetune_steps,
    )
    current_epoch = trainer.current_epoch if trainer is not None else 0
    trainer = Trainer(logger=loggers, callbacks=callbacks, checkpoint_callback=None,
                      enable_progress_bar=False, max_steps=finetune_steps)
    trainer.fit_loop.current_epoch = current_epoch + 1
    trainer.fit(model=module, datamodule=data_module)


def load_config_and_train(
        config_path: str,
        experiment_name: str,
        neptune_experiment: Optional[str] = None,
        pruned: bool = False,
        start_checkpoint: str = ''
):
    gin.parse_config_file(f'segmentation/configs/{config_path}.gin')
    train(
        config_path=config_path,
        experiment_name=experiment_name,
        pruned=pruned,
        neptune_experiment=neptune_experiment,
        start_checkpoint=start_checkpoint
    )


if __name__ == '__main__':
    argh.dispatch_command(load_config_and_train)
