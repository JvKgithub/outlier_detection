from typing import Dict, Union

import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    r"""
    Implements the Normalized Temperature-scaled Cross-Entropy loss (https://paperswithcode.com/method/nt-xent) used for CSI training. Expects that
    the first half of the pair contain the first image of the image pair and the second half the second image of the image pair.

    Args:
        in_temperature (float, optional)
            The temperature scaling factor for the loss. Defaults to 0.5.
        in_eps (float, optional)
            A small epsilon value for numerical stability. Defaults to 1e-8.
        in_clip_max (int, optional)
            Clamps the maximum value to prevent overflow to infinity when applying torch.exp. Defaults to 80.

    """

    def __init__(self, in_temperature=0.5, in_eps=1e-8, in_clip_max=80):
        super().__init__()
        self.temperature = in_temperature
        self.eps = in_eps
        self.clip_max = in_clip_max  # Will prevent overflow to inf for torch.exp for float32

    def forward(self, in_embeddings: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the NT-Xent out_loss given the outputs tensor.

        Args:
            in_embeddings (torch.Tensor)
                Tensor representing the embeddings or representations.

        Returns:
            torch.Tensor: The computed NT-Xent loss.
        """
        # Normalize to fairly measure similarity through dot product
        in_embeddings = in_embeddings / (in_embeddings.norm(dim=1, keepdim=True) + self.eps)
        sim_matrix = torch.mm(in_embeddings, in_embeddings.t())
        # Apply temperature
        sim_matrix = torch.clamp(sim_matrix / self.temperature, max=self.clip_max)  # Clamp to prevent overflow to inf for torch.exp
        # Remove diagonal, since self identity carries no information
        eye = torch.eye(sim_matrix.size(0)).to(sim_matrix.device)
        sim_matrix = torch.exp(sim_matrix) * (1 - eye)
        # Convert similarity score to log-probability
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix / (denom + self.eps) + self.eps)

        # The final loss are the log-probabilities of the image pairs
        num_original_images = sim_matrix.size(0) // 2
        out_loss = (torch.sum(sim_matrix[:num_original_images, num_original_images:].diag() +
                              sim_matrix[num_original_images:, :num_original_images].diag())
                    / (2 * num_original_images))

        return out_loss


class OdLossTotal(nn.Module):
    r"""
    A custom loss module that computes a weighted combination of NT-Xent loss,
    CrossEntropyLoss for shift classification, general classification, and a norm penalty.

    Args:
        in_duplication_factor (int)
            The factor by which the data is duplicated.
        in_loss_weights (dict)
            Dictionary containing the weights for the different loss components:
                - 'weight_sim': Weight for the NT-Xent similarity loss. Default is 0.0.
                - 'weight_shift': Weight for the shift classification loss. Default is 0.0.
                - 'weight_cls': Weight for the general classification loss. Default is 0.0.
                - 'weight_norm': Weight for the norm penalty. Default is 0.0.
        in_lr_finder (bool)
            Flag to indicate whether the loss is used for the torch_lr_finder module which needs one output tensor of the total loss.

    Attributes:
        duplication_factor (int)
            Stores the factor by which the data is duplicated.
        loss_weights (dict)
            Stores the weights for the different loss components.
        loss_criterion_ntxent (NTXentLoss)
            Criterion for the NT-Xent similarity loss.
        loss_criterion_xent (CrossEntropyLoss)
            Criterion for the shift and general classification losses.
        active_losses (list)
            List containing the active loss names based on their non-zero weights.
    """

    def __init__(self, in_duplication_factor: int, in_loss_weights: Dict, in_lr_finder: bool = False, in_eps: float = 1e-5):
        super().__init__()
        self.loss_weights = in_loss_weights
        self.duplication_factor = in_duplication_factor
        self.eps = in_eps
        self.lr_finder = in_lr_finder

        # Loss criterions
        self.loss_criterion_ntxent = NTXentLoss()
        self.loss_criterion_xent = nn.CrossEntropyLoss()

        # Determine which losses will be active and print them
        self.active_losses = ['loss_total']
        if self.loss_weights['weight_sim'] > 0:
            self.active_losses.append('loss_sim')
        if self.loss_weights['weight_shift'] > 0:
            self.active_losses.append('loss_shift')
        if self.loss_weights['weight_cls'] > 0:
            self.active_losses.append('loss_cls')
        if self.loss_weights['weight_norm'] > 0:
            self.active_losses.append('loss_norm')

    def forward(self, in_model_predictions: Dict, in_labels_shift: torch.Tensor) -> Union[Dict, torch.Tensor]:
        r"""
        Compute the combined loss given the model predictions.

        Args:
            in_model_predictions (dict)
                Dictionary containing model predictions for different tasks:
                    - 'sim': Tensor representing the embeddings or representations for similarity task.
                    - 'shift': Tensor for shift classification task.
                    - 'cls': Tensor for out-of-distribution classification task.
            in_labels_shift (torch.Tensor)
                Ground truth labels for shift classification.

        Returns:
            dict: Dictionary containing computed losses:
                - 'loss_total': The weighted combined loss.
                - 'loss_sim': Loss for the similarity task. Present if weight_sim > 0.
                - 'loss_shift': Loss for the shift classification task. Present if weight_shift > 0.
                - 'loss_cls': Loss for the out-of-distribution classification task. Present if weight_cls > 0.
                - 'loss_norm': Norm penalty loss. Present if weight_norm > 0.
        """
        out_losses = {'loss_total': torch.tensor(0.0, device=in_labels_shift.device)}

        if self.loss_weights['weight_sim'] > 0:
            loss_sim = self.loss_criterion_ntxent(in_model_predictions['sim'])
            out_losses['loss_total'] += loss_sim * self.loss_weights['weight_sim']
            out_losses['loss_sim'] = loss_sim

        if self.loss_weights['weight_shift'] > 0:
            loss_shift = self.loss_criterion_xent(in_model_predictions['shift'], in_labels_shift)
            out_losses['loss_total'] += loss_shift * self.loss_weights['weight_shift']
            out_losses['loss_shift'] = loss_shift

        if self.loss_weights['weight_cls'] > 0:
            # Infer the labels_cls. Only not shifted instances are used, labels therefore are all zeros.
            batch_size_total = in_model_predictions['cls'].size(0)
            batch_size_unshifted = batch_size_total // (self.duplication_factor * 2)
            # There are two minibatches in the complete batch
            labels_cls = torch.zeros(batch_size_unshifted * 2, dtype=torch.long, device=in_model_predictions['cls'].device)
            # Unshifted instances are located at the start of the batch and at the start of the second half of the batch
            predictions_first_half = in_model_predictions['cls'][:batch_size_unshifted]
            predictions_second_half = in_model_predictions['cls'][batch_size_total // 2: batch_size_total // 2 + batch_size_unshifted]
            combined_predictions = torch.cat((predictions_first_half, predictions_second_half), dim=0)

            loss_cls = self.loss_criterion_xent(combined_predictions, labels_cls)
            out_losses['loss_total'] += loss_cls * self.loss_weights['weight_cls']
            out_losses['loss_cls'] = loss_cls

        if self.loss_weights['weight_norm'] > 0:
            loss_inv_norm = (1.0 / (torch.norm(in_model_predictions['sim'], dim=1) + self.eps)).mean()
            out_losses['loss_total'] += loss_inv_norm * self.loss_weights['weight_norm']
            out_losses['loss_norm'] = loss_inv_norm

        if self.lr_finder:
            return out_losses['loss_total']

        return out_losses
