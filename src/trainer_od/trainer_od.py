import os
from typing import Dict

import torch
import torch.optim as optim
import numpy as np
from torch_lr_finder import LRFinder
from tqdm import tqdm

from src.losses_od.losses_od import OdLossTotal
from src.logger_od.logger_od import ScalarAverager
from src.utils_od.evaluate_epoch import evaluate_epoch


class TrainerOD:
    SUPPORTED_OPTIMIZER = {'adamw', 'sgd'}
    SUPPORTED_SCHEDULERS = {'cosine'}

    def __init__(self, in_logger, in_data_setup, in_model_handler, in_lr: float, in_optimizer_name: str, in_device: str, in_loss_weights: Dict,
                 in_duplication_factor: int, in_lr_scheduler=None, in_weight_decay=0.0, in_save_checkpoint_step: int = 10, in_eval_test: int = np.inf,
                 in_warmup_epochs: int = 0, in_warmup_reduction: int = 25):
        r"""
        Orchestrates the training, validation, and testing processes for the outlier detection model.

        Args:
            in_logger
                Logger object to track and save the training progress.
            in_data_setup
                Data setup object responsible for data loaders.
            in_model_handler
                Model handler object containing the neural network model.
            in_lr (float)
                Learning rate for the optimizer.
            in_optimizer_name (str)
                Name of the optimizer. Currently supported: 'adamw' and 'sgd'.
            in_device (str)
                The device type ('cuda' or 'cpu').
            in_loss_weights (Dict)
                Dictionary containing weights for each loss component.
            in_duplication_factor (int)
                Factor indicating how much duplication in dataset.
            in_lr_scheduler (str, optional)
                Learning rate scheduler name. Default is None.
            in_weight_decay (float, optional)
                Weight decay (L2 penalty) for the optimizer. Default is 0.0.
            in_save_checkpoint_step (int, optional)
                Number of epochs after which to save checkpoints. Default is 10.
            in_eval_test (int, optional)
                Number of epochs after which the test set is evaluated.
            in_warmup_epochs (int, optional)
                Number of epochs for the warm-up phase. Default is 0.
            in_warmup_reduction (int, optional)
                Factor by which learning rate is reduced during warm-up. Default is 25.

        Attributes:
            - SUPPORTED_OPTIMIZER (set)
                Set of currently supported optimizers.
            - SUPPORTED_SCHEDULERS (set)
                Set of currently supported learning rate schedulers.
        """
        self.logger = in_logger
        self.data_setup = in_data_setup
        self.model_handler = in_model_handler
        self.learn_rate = in_lr
        self.lr_scheduler = in_lr_scheduler
        self.device = in_device
        self.loss_weights = in_loss_weights
        self.save_checkpoint_step = in_save_checkpoint_step
        self.eval_test_step = in_eval_test
        self.warmup = in_warmup_epochs
        self.warmup_reduction = in_warmup_reduction
        self.duplication_factor = in_duplication_factor
        self.weight_decay = in_weight_decay

        self.best_val_loss = float('inf')
        self.loss_criterion = OdLossTotal(self.duplication_factor, self.loss_weights)
        self.active_losses = self.loss_criterion.active_losses

        self.train_loader, self.val_loader = self.data_setup.get_trainloader()
        self.test_loader = self.data_setup.get_testloader()

        # Setup optimizer and exclude parameters for weight decay
        decay_excludes = ['bias', 'BatchNorm2d.weight', 'BatchNorm2d.bias']
        params_with_decay = []
        params_without_decay = []
        for name_param, param in self.model_handler.model.named_parameters():
            if any(exclude in name_param for exclude in decay_excludes):
                params_without_decay.append(param)
            else:
                params_with_decay.append(param)
        param_groups = [{'params': params_with_decay, 'weight_decay': in_weight_decay},
                        {'params': params_without_decay, 'weight_decay': self.weight_decay}]
        if in_optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(param_groups, lr=self.learn_rate)
        elif in_optimizer_name == 'sgd':
            self.optimizer = optim.SGD(param_groups, lr=self.learn_rate)
        else:
            raise NotImplementedError(
                f"Optimizer {in_optimizer_name} is not supported. Supported optimizers are: {', '.join(TrainerOD.SUPPORTED_OPTIMIZER)}")

        # Initialized here, set in train()
        self.epochs = None
        self.scheduler = None

    def train(self, in_epochs):
        r"""
        Conducts the training loop for a specified number of epochs.

        Args:
            in_epochs (int)
                Number of epochs for training.
        """
        self.epochs = in_epochs
        starting_epoch = 1
        if self.lr_scheduler is not None:
            if self.lr_scheduler == 'cosine':
                total_batches = len(self.train_loader) * self.epochs
                self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.learn_rate, total_steps=total_batches,
                                                               pct_start=self.warmup / self.epochs, div_factor=self.warmup_reduction)
            else:
                raise NotImplementedError(
                    f"Scheduler {self.lr_scheduler} is not supported. Supported schedulers are: {', '.join(TrainerOD.SUPPORTED_SCHEDULERS)}")

        # Adjust starting_epoch in case of resumed training to have logging consistent
        if self.model_handler.checkpoint is not None:
            starting_epoch = self.load_checkpoint() + 1

        # Main training loop
        for epoch in range(starting_epoch, self.epochs):
            self.train_epoch(epoch)

            total_val_loss = self.eval_validation(epoch)

            if epoch % self.eval_test_step == 0:
                # When working not with image but audio data, sample-wise auc and batch-wise auc (one audio file = one batch)
                batch_stats_enable = self.data_setup.dataset != 'cifar10'
                results_dict = evaluate_epoch(in_model_handler=self.model_handler, in_test_loader=self.test_loader, in_device=self.device,
                                              in_batch_stats=batch_stats_enable)
                self.logger.log_scalar('test/auc_sample', results_dict['auc'], epoch)
                self.logger.log_scalar('test/auc_batch', results_dict['auc_batch'], epoch)

            # Save model every x epochs
            if epoch % self.save_checkpoint_step == 0:
                self.save_checkpoint(epoch, "last.pth")

            # Save model if best val loss improved.
            if total_val_loss < self.best_val_loss:
                self.best_val_loss = total_val_loss
                self.save_checkpoint(epoch, "best.pth")

    def train_epoch(self, in_epoch):
        r"""
        Processes a single epoch of training.

        Args:
            in_epoch (int)
                Current epoch number.
        """
        # Initialize loss trackers for logging
        losses_active = {loss: ScalarAverager() for loss in self.active_losses}
        self.model_handler.model.train()

        for images, labels in tqdm(self.train_loader, desc=f"Training epoch {in_epoch}"):
            batch_size = images.size(0)  # Last batch might be less than default batch_size
            images = images.to(self.device)
            labels = labels.to(self.device)
            predictions = self.model_handler.model(images, in_sim=self.loss_weights['weight_sim'] > 0, in_shift=self.loss_weights['weight_shift'] > 0,
                                                   in_cls=self.loss_weights['weight_cls'] > 0)
            # Calculate loss and update weights
            losses_dict = self.loss_criterion(predictions, labels)
            self.optimizer.zero_grad()
            losses_dict['loss_total'].backward()
            self.optimizer.step()
            self.scheduler.step()

            # Collect losses
            for loss_key in losses_active:
                losses_active[loss_key].update(losses_dict[loss_key].item(), batch_size)
        # Log epoch average losses
        for loss_key in losses_active:
            self.logger.log_scalar('train/' + loss_key, losses_active[loss_key].average, in_epoch)
        lr_current = self.optimizer.param_groups[0]['lr']
        self.logger.log_scalar('lr', lr_current, in_epoch)

    def eval_validation(self, in_epoch):
        r"""
        Evaluates the model on the validation set for a single epoch.

        Args:
            in_epoch (int)
                Current epoch number.

        Returns:
            float: Total validation loss for the current epoch.
        """
        self.model_handler.model.eval()
        # Initialize loss trackers for logging
        losses_active = {loss: ScalarAverager() for loss in self.active_losses}

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, f"Validating epoch {in_epoch}"):
                batch_size = images.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model_handler.model(images, in_sim=self.loss_weights['weight_sim'] > 0,
                                                       in_shift=self.loss_weights['weight_shift'] > 0,
                                                       in_cls=self.loss_weights['weight_cls'] > 0)
                losses_dict = self.loss_criterion(predictions, labels)

                # Collect losses
                for loss_key in losses_active:
                    losses_active[loss_key].update(losses_dict[loss_key].item(), batch_size)

        # Log the average losses
        for loss_key in losses_active:
            self.logger.log_scalar('val/' + loss_key, losses_active[loss_key].average, in_epoch)
        return losses_active['loss_total'].average

    def find_lr(self):
        r"""
        Utilizes the LRFinder to find a suitable learning rate for training.
        """
        lr_finder_loss = OdLossTotal(self.duplication_factor, self.loss_weights, in_lr_finder=True)
        lr_finder = LRFinder(self.model_handler.model, self.optimizer, lr_finder_loss, device=self.device)
        lr_finder.range_test(self.train_loader, end_lr=100, num_iter=200)
        lr_finder.plot()
        lr_finder.reset()

    def save_checkpoint(self, in_epoch, in_filename):
        r"""
        Saves a checkpoint of the current model, optimizer, and scheduler states.

        Args:
            in_epoch (int)
                Current epoch number.
            in_filename (str)
                Name of the checkpoint file.
        """
        in_filename = os.path.join(self.logger.log_dir, in_filename)
        checkpoint = {
            'epoch': in_epoch,
            'model_state_dict': self.model_handler.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, in_filename)

    def load_checkpoint(self):
        r"""
        Loads a previously saved checkpoint, except for model weights (assumed to be done already in the model_handler).

        Returns:
            int: Epoch number at which the checkpoint was saved.
        """
        self.optimizer.load_state_dict(self.model_handler.checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(self.model_handler.checkpoint['scheduler_state_dict'])
        self.best_val_loss = self.model_handler.checkpoint['best_val_loss']
        return self.model_handler.checkpoint['epoch']
