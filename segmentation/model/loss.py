"""Loss functions for semantic segmentation."""

from typing import Dict, List, Tuple, Union

import torch
from torch import nn


class PixelWiseCrossEntropyLoss(nn.Module):
    """Pixel-wise cross-entropy loss for semantic segmentation."""

    def __init__(self, ignore_index: int = 255, return_correct: bool = False) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.return_correct = return_correct
        self.ignore_index = ignore_index

    def forward(
        self, predicted_logits: torch.Tensor, target_labels: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Forward call of the loss.

        Args:
            predicted_logits (torch.Tensor): Input predicted logits
            target_labels (torch.Tensor): Input target labels as Ids

        Returns:
            torch.Tensor: Output loss
        """
        # Flatten the predicted logits and target labels
        predicted_logits = predicted_logits.view(-1, predicted_logits.size(-1))
        target_labels = target_labels.view(-1) - 1

        # Calculate the cross-entropy loss
        loss = self.loss(predicted_logits, target_labels)

        if self.return_correct:

            # Calculate the number of correct predictions
            predicted_labels = torch.argmax(predicted_logits, dim=-1)
            correct = predicted_labels == target_labels

            # Ignore Index
            mask = (target_labels != self.ignore_index).nonzero().squeeze()
            correct = correct[mask]

            return loss, correct

        else:
            return loss


class KLDLoss(nn.Module):
    """Kullback-Leibler divergence loss for semantic segmentation via multi-scale prototytes."""

    def __init__(
        self, prototype_class_identity: torch.Tensor, num_scales: int, scale_num_prototypes: Dict[int, Tuple[int, int]]
    ) -> None:
        super().__init__()
        self.prototype_class_identity = prototype_class_identity
        self.num_scales = num_scales
        self.scale_num_prototypes = scale_num_prototypes

    def forward(self, prototype_distances: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """Forward call of the loss.

        Args:
            prototype_activations (torch.Tensor): Input prototype activations
            target_labels (torch.Tensor): Input target labels as Ids

        Returns:
            torch.Tensor: Output loss
        """

        target_labels = target_labels.view(target_labels.shape[0], -1) - 1

        prototype_distances = prototype_distances.permute(0, 2, 3, 1)
        prototype_distances = prototype_distances.view(
            prototype_distances.shape[0], -1, prototype_distances.shape[-1]
        )  # (batch_size, img_size, num_proto)

        # calculate KLD over class pixels between prototypes from same class
        kld_loss = []
        for img_i in range(target_labels.shape[0]):
            for cls_i in torch.unique(target_labels[img_i]).cpu().detach().numpy():
                if cls_i < 0 or cls_i >= self.prototype_class_identity.shape[1]:
                    continue  # Filter the undefined class

                cls_protos = torch.nonzero(self.prototype_class_identity[:, cls_i]).flatten().cpu().detach().numpy()

                if len(cls_protos) == 0:
                    continue

                cls_mask = target_labels[img_i] == cls_i

                for scale in range(self.num_scales):

                    cls_protos_scale = [
                        cls_proto
                        for cls_proto in cls_protos
                        if cls_proto >= self.scale_num_prototypes[scale][0]
                        and cls_proto < self.scale_num_prototypes[scale][1]
                    ]

                    log_cls_activations = [
                        torch.masked_select(prototype_distances[img_i, :, i], cls_mask) for i in cls_protos_scale
                    ]
                    log_cls_activations = [torch.nn.functional.log_softmax(act, dim=0) for act in log_cls_activations]

                    if len(cls_protos_scale) < 2:
                        # no distribution over given class
                        continue

                    for j in range(len(cls_protos_scale)):

                        if len(log_cls_activations[j]) < 2:
                            # no distribution over given class
                            continue

                        log_p1_scores = log_cls_activations[j]
                        for k in range(j + 1, len(cls_protos_scale)):

                            if len(log_cls_activations[k]) < 2:
                                # no distribution over given class
                                continue

                            log_p2_scores = log_cls_activations[k]

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
            kld_loss = torch.exp(-kld_loss)
            kld_loss = torch.mean(kld_loss)
        else:
            kld_loss = torch.tensor(0.0)

        return kld_loss


class EntropySpatLoss(nn.Module):
    """Entropy of prototype activation over the 2D feature maps."""

    def __init__(self, prototype_class_identity: torch.Tensor) -> None:
        super().__init__()
        self.prototype_class_identity = prototype_class_identity

    def forward(self, prototype_activations: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """Forward call of the loss.

        Args:
            prototype_activations (torch.Tensor): Input prototype activations
            target_labels (torch.Tensor): Input target labels as Ids

        Returns:
            torch.Tensor: Output loss
        """

        target_labels = target_labels.view(target_labels.shape[0], -1) - 1
        prototype_activations = prototype_activations.view(
            target_labels.shape[0], -1, self.prototype_class_identity.shape[0]
        )

        entropy_loss = []
        # calculate entropy over class pixels across pixel position for each prototypes
        for img_i in range(target_labels.shape[0]):
            for cls_i in torch.unique(target_labels[img_i]).cpu().detach().numpy():
                if cls_i < 0 or cls_i >= self.prototype_class_identity.shape[1]:
                    continue  # Filter the undefined class

                cls_protos = torch.nonzero(self.prototype_class_identity[:, cls_i]).flatten().cpu().detach().numpy()

                if len(cls_protos) == 0:
                    continue

                cls_mask = target_labels[img_i] == cls_i
                if cls_mask.sum() < 2:
                    continue

                cls_activations = [
                    torch.masked_select(prototype_activations[img_i, :, i], cls_mask) for i in cls_protos
                ]

                log_cls_activations = [torch.nn.functional.log_softmax(act) for act in cls_activations]
                log_norm = torch.log(torch.sum(cls_mask)).detach()

                prob_cls_activations = [torch.nn.functional.softmax(act) for act in cls_activations]
                entropy_cls_activations = [
                    torch.sum(-prob_cls_activation * log_prob_cls_activation) / log_norm
                    for prob_cls_activation, log_prob_cls_activation in zip(prob_cls_activations, log_cls_activations)
                ]
                entropy_cls_activations = torch.stack(entropy_cls_activations, dim=0)
                entropy_cls_activations = torch.mean(entropy_cls_activations, dim=0)

                entropy_loss.append(entropy_cls_activations)

        if len(entropy_loss) > 0:
            entropy_loss = torch.stack(entropy_loss)
            entropy_loss = torch.mean(entropy_loss)
        else:
            entropy_loss = torch.tensor(0.0)

        return entropy_loss


class EntropySamplLoss(nn.Module):
    """Entropy of prototype activations over the set of prototypes for a class."""

    def __init__(
        self, prototype_class_identity: torch.Tensor, num_scales: int, scale_num_prototypes: Dict[int, Tuple[int, int]]
    ) -> None:
        super().__init__()
        self.prototype_class_identity = prototype_class_identity
        self.num_scales = num_scales
        self.scale_num_prototypes = scale_num_prototypes

    def forward(self, prototype_activations: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """Forward call of the loss.

        Args:
            prototype_activations (torch.Tensor): Input prototype activations
            target_labels (torch.Tensor): Input target labels as Ids

        Returns:
            torch.Tensor: Output loss
        """

        target_labels = target_labels.view(target_labels.shape[0], -1) - 1
        prototype_activations = prototype_activations.view(
            target_labels.shape[0], -1, self.prototype_class_identity.shape[0]
        )

        entropy_loss = []
        # calculate KLD over class pixels between prototypes from same class
        for img_i in range(target_labels.shape[0]):
            for cls_i in torch.unique(target_labels[img_i]).cpu().detach().numpy():
                if cls_i < 0 or cls_i >= self.prototype_class_identity.shape[1]:
                    continue  # Filter the undefined class

                cls_protos = torch.nonzero(self.prototype_class_identity[:, cls_i]).flatten().cpu().detach().numpy()

                if len(cls_protos) == 0:
                    continue

                cls_mask = target_labels[img_i] == cls_i

                for scale in range(self.num_scales):

                    cls_protos_scale = [
                        cls_proto
                        for cls_proto in cls_protos
                        if cls_proto >= self.scale_num_prototypes[scale][0]
                        and cls_proto < self.scale_num_prototypes[scale][1]
                    ]

                    cls_activations = torch.stack(
                        [torch.masked_select(prototype_activations[img_i, :, i], cls_mask) for i in cls_protos_scale],
                        dim=-1,
                    )

                    log_cls_activations = torch.nn.functional.log_softmax(cls_activations, dim=-1)
                    log_norm = torch.log(torch.tensor(cls_activations.shape[-1])).detach()

                    prob_cls_activations = torch.nn.functional.softmax(cls_activations, dim=-1)
                    entropy_cls_activations = torch.sum(-prob_cls_activations * log_cls_activations, dim=-1) / log_norm
                    entropy_cls_activations = torch.mean(entropy_cls_activations, dim=0)

                    entropy_loss.append(entropy_cls_activations)

        if len(entropy_loss) > 0:
            entropy_loss = torch.stack(entropy_loss)
            entropy_loss = torch.mean(entropy_loss)
        else:
            entropy_loss = torch.tensor(0.0)

        return entropy_loss


class NormLoss(nn.Module):
    """Average of the norm of the prototype activations for a class over the pixels assigned to the class."""

    def __init__(self, prototype_class_identity: torch.Tensor, norm_type: str) -> None:
        super().__init__()
        self.prototype_class_identity = prototype_class_identity
        self.norm_type = norm_type

    def forward(self, prototype_activations: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """Forward call of the loss.

        Args:
            prototype_activations (torch.Tensor): Input prototype activations
            target_labels (torch.Tensor): Input target labels as Ids

        Returns:
            torch.Tensor: Output loss
        """

        target_labels = target_labels.view(target_labels.shape[0], -1) - 1
        prototype_activations = prototype_activations.view(
            target_labels.shape[0], -1, self.prototype_class_identity.shape[0]
        )

        norm_loss = []

        for img_i in range(target_labels.shape[0]):
            for cls_i in torch.unique(target_labels[img_i]).cpu().detach().numpy():
                if cls_i < 0 or cls_i >= self.prototype_class_identity.shape[1]:
                    continue  # Filter the undefined class

                cls_protos = torch.nonzero(self.prototype_class_identity[:, cls_i]).flatten().cpu().detach().numpy()

                if len(cls_protos) == 0:
                    continue

                cls_mask = target_labels[img_i] == cls_i

                if cls_mask.sum() == 0:
                    continue

                cls_activations = [
                    torch.masked_select(prototype_activations[img_i, :, i], cls_mask) for i in cls_protos
                ]

                if self.norm_type == "l1":
                    norm_value = [torch.norm(act, p=1) / act.shape[0] for act in cls_activations]
                elif self.norm_type == "linf":
                    norm_value = [torch.norm(act, p=float("inf")) for act in cls_activations]

                norm_value = torch.stack(norm_value, dim=0)
                norm_value = torch.mean(norm_value, dim=0)

                norm_loss.append(norm_value)

        if len(norm_loss) > 0:
            norm_loss = torch.stack(norm_loss)
            norm_loss = torch.mean(norm_loss)
        else:
            norm_loss = torch.tensor(0.0)

        return norm_loss


class ScaleMax(nn.Module):
    """Loss that maximize per scale prototype activations for all groups across all classes."""

    def __init__(self, ppnet: nn.Module) -> None:
        super().__init__()
        self.ppnet = ppnet

    def forward(self) -> torch.Tensor:
        """Forward call of the loss. Loss directly computed on model."""

        tot_max = []
        for cls_i in range(self.ppnet.num_classes):
            if self.ppnet.prototype_class_identity[:, cls_i].sum() == 0:
                continue
            id_proj = self.ppnet.group_class_identity[:, cls_i].argmax().item() // self.ppnet.num_groups
            matrix_weight = self.ppnet.group_projection[id_proj].weight
            prev_scale = 0
            for scale in range(self.ppnet.num_scales):

                cls_proto_scale = (
                    torch.nonzero(
                        self.ppnet.prototype_class_identity[
                            self.ppnet.scale_num_prototypes[scale][0] : self.ppnet.scale_num_prototypes[scale][1], cls_i
                        ]
                    )
                    .flatten()
                    .cpu()
                    .detach()
                    .numpy()
                )

                if len(cls_proto_scale) == 0:
                    continue

                matrix_weight_scale = matrix_weight[:, prev_scale : prev_scale + len(cls_proto_scale)]
                max_matrix_weight_scale = torch.max(matrix_weight_scale, dim=1).values
                max_matrix_weight_scale = torch.mean(max_matrix_weight_scale)
                tot_max.append(max_matrix_weight_scale)

                prev_scale += len(cls_proto_scale)

        tot_max = torch.stack(tot_max)
        tot_max = torch.mean(tot_max)

        return -tot_max


class EntropyGroup(nn.Module):
    """Entropy of prototype weights for a given group assigned to a class."""

    def __init__(self, ppnet, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.ppnet = ppnet
        self.epsilon = epsilon

    def forward(self) -> torch.Tensor:
        """Forward call of the loss. Loss directly computed on model."""

        tot_entropy_loss = []
        for cls_i in range(self.ppnet.num_classes):
            if self.ppnet.prototype_class_identity[:, cls_i].sum() == 0:
                continue
            id_proj = self.ppnet.group_class_identity[:, cls_i].argmax().item() // self.ppnet.num_groups
            matrix_weight = self.ppnet.group_projection[id_proj].weight

            for group in range(self.ppnet.num_groups):
                group_matrix_weight = matrix_weight[group]
                entropy_loss = -torch.sum(
                    group_matrix_weight * torch.log(group_matrix_weight + self.epsilon)
                ) / torch.log(torch.tensor(group_matrix_weight.shape[0]))
                tot_entropy_loss.append(entropy_loss)

        tot_entropy_loss = torch.stack(tot_entropy_loss)
        tot_entropy_loss = torch.mean(tot_entropy_loss)

        return tot_entropy_loss


class CrossEntropyGroup(nn.Module):
    """Maximize the cross-entropy between prototype weights assigned to group from the same class."""

    def __init__(self, ppnet, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.ppnet = ppnet
        self.epsilon = epsilon

    def forward(self) -> torch.Tensor:
        """Forward call of the loss. Loss directly computed on model."""

        tot_entropy_loss = []
        for cls_i in range(self.ppnet.num_classes):

            if self.ppnet.prototype_class_identity[:, cls_i].sum() == 0:
                continue

            id_proj = self.ppnet.group_class_identity[:, cls_i].argmax().item() // self.ppnet.num_groups
            matrix_weight = self.ppnet.group_projection[id_proj].weight

            for i in range(self.ppnet.num_groups):
                for j in range(self.ppnet.num_groups):
                    if i == j:
                        continue
                    group_matrix_weight_1 = matrix_weight[i]
                    group_matrix_weight_2 = matrix_weight[j]

                    entropy_loss = -torch.sum(
                        group_matrix_weight_1 * torch.log(torch.clamp(group_matrix_weight_2, self.epsilon))
                    )
                    tot_entropy_loss.append(entropy_loss)

        tot_entropy_loss = torch.stack(tot_entropy_loss)
        tot_entropy_loss = torch.mean(tot_entropy_loss)

        return -tot_entropy_loss


class KLDLossGroup(nn.Module):
    """Kullback-Leibler divergence loss over the group activations for group assigned to the same class."""

    def __init__(
        self, prototype_class_identity: torch.Tensor, group_class_identity: torch.Tensor, num_groups: int
    ) -> None:
        super().__init__()
        self.prototype_class_identity = prototype_class_identity
        self.group_class_identity = group_class_identity
        self.num_groups = num_groups

    def forward(self, list_group_activation: List[torch.Tensor], target_labels: torch.Tensor) -> torch.Tensor:
        """Forward call of the loss.

        Args:
            prototype_activations (torch.Tensor): Input prototype activations
            target_labels (torch.Tensor): Input target labels as Ids

        Returns:
            torch.Tensor: Output loss
        """

        target_labels = target_labels.view(target_labels.shape[0], -1) - 1

        # calculate KLD over class pixels between groups from same class
        kld_loss = []
        for img_i in range(target_labels.shape[0]):
            for cls_i in torch.unique(target_labels[img_i]).cpu().detach().numpy():
                if cls_i < 0 or cls_i >= self.prototype_class_identity.shape[1]:
                    continue  # Filter the undefined class

                if self.prototype_class_identity[:, cls_i].sum() == 0:
                    continue

                id_proj = self.group_class_identity[:, cls_i].argmax().item() // self.num_groups

                group_activation = list_group_activation[id_proj]
                group_activation = group_activation.view(target_labels.shape[0], -1, self.num_groups)
                cls_mask = target_labels[img_i] == cls_i

                log_group_activations = [
                    torch.masked_select(group_activation[img_i, :, i], cls_mask) for i in range(self.num_groups)
                ]

                log_group_activations = [torch.nn.functional.log_softmax(act, dim=0) for act in log_group_activations]

                for j in range(self.num_groups):

                    if len(log_group_activations[j]) < 2:
                        # no distribution over given class
                        continue

                    log_p1_scores = log_group_activations[j]
                    for k in range(j + 1, self.num_groups):

                        if len(log_group_activations[k]) < 2:
                            # no distribution over given class
                            continue

                        log_p2_scores = log_group_activations[k]

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
            kld_loss = torch.exp(-kld_loss)
            kld_loss = torch.mean(kld_loss)
        else:
            kld_loss = torch.tensor(0.0)

        return kld_loss
