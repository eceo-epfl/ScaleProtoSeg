"""Scripts to zero out prototypes with low weight in all groups"""

import os

import argh
import torch

from settings import log


def threshold_save(model_name: str, training_phase: str, threshold: float):
    """Callable to zero out unused prototypes based on input threshold."""

    threshold = float(threshold)
    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)

    if training_phase == "pruned":
        checkpoint_path = os.path.join(model_path, "pruned/checkpoints/push_last.pth")
    elif training_phase == "final-group":
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}.pth")
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

    log(f"Loading model from {checkpoint_path}")
    ppnet = torch.load(checkpoint_path)

    for i in range(len(ppnet.group_projection)):
        ppnet.group_projection[i].weight.data[ppnet.group_projection[i].weight.data < threshold] = 0

    log(f"Saving model to {os.path.dirname(checkpoint_path)}")
    torch.save(ppnet, os.path.join(os.path.dirname(checkpoint_path), f"th-{threshold}-{training_phase}_last.pth"))


if __name__ == "__main__":
    argh.dispatch_command(threshold_save)
