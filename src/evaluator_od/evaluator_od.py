import os
import json

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose
import torchaudio
import matplotlib.pyplot as plt

from src.utils_od.evaluate_epoch import evaluate_epoch


class EvaluatorOD:

    def __init__(self, in_model_handler, in_device: str, in_log_dir: str):
        """
        Orchestrates the training, validation, and testing processes for the outlier detection model.

        Args:
        in_model_handler
            Model handler which contains the model to be evaluated.
        in_device (str)
            Device where the model will run, e.g., 'cuda' or 'cpu'.
        in_logdir (str)
            Path to folder to save evaluation results
        """
        self.model_handler = in_model_handler
        self.device = in_device
        self.log_dir = in_log_dir

        self.dataset_statistics = None
        self.data_setup = None
        self.test_loader = None

    def evaluate_dataset(self, in_data_setup, in_batch_stats: bool = False):
        """
        Evaluates the loaded model on the provided data setup.

        Args:
        in_data_setup
            The data setup containing the test loader.
        """
        self.data_setup = in_data_setup
        test_loader = self.data_setup.get_testloader()
        results_dict = evaluate_epoch(self.model_handler, test_loader, self.device, in_return_score=True, in_batch_stats=in_batch_stats)

        # Calculate and save sample statistics
        scores_class_0_sample = results_dict['pred_scr'][results_dict['labels'] == 0]
        scores_class_1_sample = results_dict['pred_scr'][results_dict['labels'] == 1]

        sample_statistics = {
            'mean_id_sample': float(np.mean(scores_class_0_sample)),
            'std_id_sample': float(np.std(scores_class_0_sample)),
            'mean_ood_sample': float(np.mean(scores_class_1_sample)),
            'std_ood_sample': float(np.std(scores_class_1_sample)),
            'auc_sample': float(results_dict['auc']),
        }

        # Calculate and save wav statistics
        scores_class_0_batch = results_dict['pred_scr_batch'][results_dict['labels_batch'] == 0]
        scores_class_1_batch = results_dict['pred_scr_batch'][results_dict['labels_batch'] == 1]

        batch_statistics = {
            'mean_id_batch': float(np.mean(scores_class_0_batch)),
            'std_id_batch': float(np.std(scores_class_0_batch)),
            'mean_ood_batch': float(np.mean(scores_class_1_batch)),
            'std_ood_batch': float(np.std(scores_class_1_batch)),
            'auc_batch': float(results_dict['auc_batch']),
        }

        self.dataset_statistics = {
            'sample': sample_statistics,
            'batch': batch_statistics
        }

        # Print statistics
        print(f"AUC Sample: {sample_statistics['auc_sample']:.2f}, "
              f"AUC Batch: {batch_statistics['auc_batch']:.2f}")
        print(f"Sample In-distribution scores: {sample_statistics['mean_id_sample']:.2f}+-"
              f"{sample_statistics['std_id_sample']:.2f}, "
              f"Sample Out-of-distribution scores: {sample_statistics['mean_ood_sample']:.2f}+-"
              f"{sample_statistics['std_ood_sample']:.2f}")
        print(f"Batch In-distribution scores: {batch_statistics['mean_id_batch']:.2f}+-"
              f"{batch_statistics['std_id_batch']:.2f}, "
              f"Batch Out-of-distribution scores: {batch_statistics['mean_ood_batch']:.2f}+-"
              f"{batch_statistics['std_ood_batch']:.2f}")

        self._save_histogram(scores_class_0_batch, scores_class_1_batch)

        # Save the combined statistics to a JSON file
        file_path = os.path.join(self.log_dir, 'evaluation_dataset.json')
        with open(file_path, 'w', encoding="utf-8") as json_file:
            json.dump(self.dataset_statistics, json_file, indent=4)

    def evaluate_image(self, in_path_image: str, in_transform: Compose):
        """
        Evaluates the model on a single image.

        Args:
        in_path_image (str)
            Path to the image to be evaluated.
        in_transform (Compose)
            List of torch transforms used for preprocessing image.
        """
        image = Image.open(in_path_image).convert("RGB")

        # Pre-process the image
        image_tensor = in_transform(image).unsqueeze(0).to(self.device)

        # Evaluate sample
        self.model_handler.model.eval()
        with torch.no_grad():
            predictions = self.model_handler.model(image_tensor, in_sim=True, in_shift=False, in_cls=False)
            # The prediction score is calculated on the basis of the norm of the similarity scores
            prediction_score = predictions['sim'].norm(dim=1).cpu().detach().numpy()

        # Print and save the results
        output_string = f"Image: {in_path_image}\nPrediction score: {prediction_score}"
        print(output_string)
        file_path = os.path.join(self.log_dir, 'evaluation_image.txt')
        with open(file_path, 'a', encoding="utf-8") as text_file:
            text_file.write(output_string + '\n')

    def evaluate_audio_wav(self, in_path_audio_file: str, in_transforms: Compose, in_sample_len):
        waveform, _ = torchaudio.load(in_path_audio_file)

        # Calculate the total number of complete samples from the waveform
        num_samples = waveform.shape[-1] // in_sample_len

        # Discard the last possible shorter sample from the beginning
        start_idx = waveform.shape[-1] % in_sample_len
        waveform = waveform[:, start_idx:start_idx + num_samples * in_sample_len]

        # Reshape waveform to (num_samples, sample_len)
        waveform = waveform.reshape(-1, in_sample_len)

        # Initialize an empty list to hold prediction scores
        prediction_scores = []

        # Evaluate the model for each sample
        self.model_handler.model.eval()
        with torch.no_grad():
            for current_sample in waveform:
                # Get the current sample, add a channel dimension, and unsqueeze to add a batch dimension
                current_sample = current_sample.unsqueeze(0).unsqueeze(0)  # Now shape: [1, 1, sample_len]

                # Pre-process the sample
                image_tensor = in_transforms(current_sample).to(self.device)

                # Repeat the input along the color-channel dimension to simulate 3 color channels
                image_tensor = image_tensor.repeat(1, 3, 1, 1)  # Now shape: [1, 3, height, width]

                # Evaluate the sample
                predictions = self.model_handler.model(image_tensor, in_sim=True, in_shift=False, in_cls=False)

                # Extract the similarity score and calculate its norm
                prediction_score = predictions['sim'].norm(dim=1).item()
                prediction_scores.append(prediction_score)

        # Calculate the mean of the prediction scores
        mean_prediction_score = sum(prediction_scores) / len(prediction_scores) if prediction_scores else 0

        # Print and save the results
        output_string = f"Audio file: {in_path_audio_file}\nPrediction score: {mean_prediction_score}"
        print(output_string)
        file_path = os.path.join(self.log_dir, 'evaluation_audio.txt')
        with open(file_path, 'a', encoding="utf-8") as text_file:
            text_file.write(output_string + '\n')

    def _save_histogram(self, in_id_samples: np.ndarray, in_ood_samples: np.ndarray):
        """
        Plots and saves a histogram of inlier and outlier scores.

        Args:
        in_id_samples (np.ndarray)
            Array of in-distribution scores.
        in_ood_samples (np.ndarray)
            Array of out-of-distribution scores.
        """
        plt.figure(figsize=(12, 7))

        # Improved binning for better visual clarity.
        # We compute the range based on the combined data but bin separately to retain the original behavior.
        bins = np.linspace(min(np.min(in_id_samples), np.min(in_ood_samples)),
                           max(np.max(in_id_samples), np.max(in_ood_samples)),
                           50)

        plt.hist(in_id_samples, bins=bins, alpha=0.6, label='In-distribution', color='#1f77b4')  # A shade of blue
        plt.hist(in_ood_samples, bins=bins, alpha=0.6, label='Out-of-distribution', color='#d62728')  # A shade of red

        plt.xlabel('Score', fontsize=14)
        plt.ylabel('Number of Samples (Log Scale)', fontsize=14)
        plt.title('Distribution of Scores', fontsize=16)
        plt.legend(loc='upper right', fontsize=12)

        # Adding gridlines for better visualization
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Set y-axis to logarithmic scale
        plt.yscale('log')

        # Use a tight layout
        plt.tight_layout()

        # Save the histogram at a higher resolution
        plt.savefig(os.path.join(self.log_dir, 'scores_histogram.png'), dpi=150)
        plt.close()
