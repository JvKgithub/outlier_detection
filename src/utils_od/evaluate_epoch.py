from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_epoch(in_model_handler, in_test_loader: DataLoader, in_device: str, in_return_score: bool = False, in_batch_stats: bool = False) \
        -> Dict:
    """
    Evaluates the loaded model on the provided data.

    Args:
        in_model_handler:
            Model handler which contains the model to be evaluated.
        in_test_loader:
            The test data loader.
        in_device (str):
            Device where the model will run, e.g., 'cuda' or 'cpu'.
        in_return_score (bool):
            If True, return AUC along with prediction scores and labels. Otherwise, only return AUC.
        in_batch_stats (bool):
            If True, also compute statistics for the batch means. This should be enabled for .wav training data, not for images.

    Returns:
        A dictionary containing the AUC and optionally prediction scores and labels, for both individual samples and batch means.
    """
    in_model_handler.model.eval()

    test_predictions_sample = []
    test_predictions_batch_mean = [] if in_batch_stats else None
    labels_sample = []
    labels_batch_mean = [] if in_batch_stats else None

    with torch.no_grad():
        for images, labels in tqdm(in_test_loader, desc="Testing"):
            img_batch = images.to(in_device)
            labels = labels.to(in_device)  # Keep labels on the same device

            if in_batch_stats:
                # If batch statistics calculation is enabled
                labels = labels.squeeze(0)
                img_batch = img_batch.squeeze(0)  # Squeeze if we're treating the entire batch as one item

            predictions = in_model_handler.model(img_batch, in_sim=True, in_shift=False, in_cls=False)

            # AUC is calculated on the basis of the norm of the similarity scores
            sim_norm = predictions['sim'].norm(dim=1)
            test_predictions_sample.append(sim_norm)
            labels_sample.append(labels)

            if in_batch_stats:
                batch_mean_score = torch.mean(sim_norm)
                test_predictions_batch_mean.append(batch_mean_score)
                batch_mean_label = labels[0] if labels.ndim > 0 else labels
                labels_batch_mean.append(batch_mean_label)

    # Convert lists to tensors for AUC computation
    test_predictions_sample = torch.cat(test_predictions_sample)
    labels_sample = torch.cat(labels_sample)

    # Compute AUC on the device
    out_auc_sample = compute_auc(-test_predictions_sample, labels_sample).item()

    out_test_results = {'auc': out_auc_sample}

    if in_return_score:
        # Return prediction scores and labels to calculate testset statistics, only done in evaluation not during training
        out_test_results['pred_scr'] = test_predictions_sample.cpu().numpy()
        out_test_results['labels'] = labels_sample.cpu().numpy()

    if in_batch_stats:
        # Average prediction score over minibatches (ie .wav audio files) and calculate minibatch AUC
        test_predictions_batch_mean = torch.stack(test_predictions_batch_mean)
        labels_batch_mean = torch.stack(labels_batch_mean)

        out_auc_batch_mean = compute_auc(-test_predictions_batch_mean, labels_batch_mean).item()
        out_test_results['auc_batch'] = out_auc_batch_mean
        if in_return_score:
            # Return prediction scores and labels to calculate testset statistics, only done in evaluation not during training
            out_test_results['pred_scr_batch'] = test_predictions_batch_mean.cpu().numpy()
            out_test_results['labels_batch'] = labels_batch_mean.cpu().numpy()

    return out_test_results


def compute_auc(in_predictions: torch.Tensor, in_labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) using PyTorch.

    Assumes the labels are binary (0 or 1). The function sorts the predictions and computes the AUC
    based on the ordering of the predictions and their corresponding labels.

    Args:
        in_predictions (torch.Tensor):
            The predictions tensor, typically the scores or probabilities output by a model.
        in_labels (torch.Tensor):
            The true binary labels tensor (0 or 1) corresponding to the predictions.

    Returns:
        out_auc (torch.Tensor):
            A scalar tensor representing the AUC score.

    Note:
        This function is designed to be used with binary classification tasks. Ensure labels are
        either 0 or 1. For multi-class tasks or labels other than 0 and 1, modifications might be needed.
    """
    # Sort predictions and corresponding labels
    sorted_indices = torch.argsort(in_predictions, descending=True)
    sorted_labels = in_labels[sorted_indices]

    # Compute the cumulative sum of the sorted labels
    cum_positive = torch.cumsum(sorted_labels, dim=0)
    num_positive = cum_positive[-1]
    num_negative = in_labels.size(0) - num_positive

    # Calculate AUC
    out_auc = (cum_positive.sum() - (num_positive * (num_positive + 1)) / 2) / (num_positive * num_negative)
    return out_auc
