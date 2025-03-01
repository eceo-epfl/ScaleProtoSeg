import os

import argh
import gin
import torch
import torch.utils.data

import prune
import save
from preprocess import preprocess
from segmentation.data.data_module import PatchClassificationDataModule
from log import create_logger
from segmentation.data.dataset import PatchClassificationDataset


def run_pruning(config_name: str, experiment_name: str, k: int = 6, prune_threshold: int = 3):
    gin.parse_config_file(f'segmentation/configs/{config_name}.gin', skip_unknown=True)
    gin.parse_config_file(os.path.join(os.environ['RESULTS_DIR'], experiment_name, 'config.gin'),
                          skip_unknown=True)

    model_path = os.path.join(os.environ['RESULTS_DIR'], experiment_name, 'checkpoints/push_last.pth')
    output_dir = os.path.join(os.environ['RESULTS_DIR'], experiment_name, 'pruned')

    os.makedirs(output_dir, exist_ok=True)

    log, logclose = create_logger(log_filename=os.path.join(output_dir, 'prune.log'))

    ppnet = torch.load(model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    # load the data
    # TODO use configurable value for model_image_size here
    data_module = PatchClassificationDataModule(batch_size=1)

    # TODO: implement test here for segmentation
    # test_loader = data_module.val_dataloader(batch_size=1)

    # push set: needed for pruning because it is unnormalized
    train_push_loader = data_module.train_push_dataloader(batch_size=1)

    push_dataset = PatchClassificationDataset(
        split_key='train',
        is_eval=True,
        push_prototypes=True
    )

    def preprocess_push_input(x):
        return preprocess(x, mean=train_push_loader.dataset.mean, std=train_push_loader.dataset.std)

    # log('test set size: {0}'.format(len(test_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))

    # prune prototypes
    log('prune')
    with torch.no_grad():
        # accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        # class_specific=class_specific, log=log)
        # log(f"Accuracy before pruning: {accu}")

        prune.prune_prototypes(dataset=push_dataset,
                               prototype_network_parallel=ppnet_multi,
                               k=k,
                               prune_threshold=prune_threshold,
                               preprocess_input_function=preprocess_push_input,  # normalize
                               original_model_dir=output_dir,
                               epoch_number=0,
                               # model_name=None,
                               log=log,
                               copy_prototype_imgs=False, )
        # accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        # class_specific=class_specific, log=log)
        # log(f"Accuracy after pruning: {accu}")

    save.save_model_w_condition(model=ppnet, model_dir=output_dir,
                                model_name='pruned',
                                accu=1.0,
                                target_accu=0.0, log=log)
    logclose()


if __name__ == '__main__':
    argh.dispatch_command(run_pruning)
