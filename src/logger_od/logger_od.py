import os
from typing import Dict
import json

from tensorboardX import SummaryWriter


class LoggerOD:
    r"""Logging utility for Outlier Detection.

    Args:
        in_log_dir (str)
            Path to the logging directory.
        in_log_tensorboard (bool, optional)
            Flag to determine if TensorBoard logging is required. Defaults to True. Later specific logs can individually decide to log to tensorboard
            or not.
        in_log_txt (bool, optional)
            Flag to determine if logging to txt is required. Defaults to True. Later specific logs can individually decide to log to .txt
        in_continue_train (bool, optional)
            Flag to determine whether training is continued, check for empty log dir to avoid overwriting disabled.

    Raises:
        FileExistsError: If the logging directory already exists.
    """

    def __init__(self, in_log_dir, in_log_tensorboard=True, in_log_txt=True, in_continue_train=False):
        self.log_dir = os.path.join('../logs', in_log_dir)
        self.log_tensorboard = in_log_tensorboard
        self.log_txt = in_log_txt

        # Create directory for logs
        if os.path.exists(self.log_dir) and not in_continue_train:
            raise FileExistsError(f"Logging directory {self.log_dir} already exists. To avoid overwriting, choose a new logging directory.")
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize tensorboard writer
        if in_log_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # Open logs.txt in append mode.
        if in_log_txt:
            self.txt_file_path = os.path.join(self.log_dir, 'logs.txt')
            # Open file and leave open for later logging to avoid opening and closing the file on every log
            self.txt_file = open(self.txt_file_path, 'a', encoding="utf-8")  # pylint: disable=consider-using-with

    def log_scalar(self, in_tag, in_value, in_epoch, in_tensorboard=True, in_txt=True, in_print=True):
        r"""Log scalar values. Depending on choice, can be logged to tensorboard, .txt file and the console.

        Args:
            in_tag (str)
                Name of the scalar.
            in_value (float)
                Value of the scalar.
            in_epoch (int)
                Epoch number.
            in_tensorboard (bool, optional)
                Whether to log to TensorBoard. Defaults to True.
            in_txt (bool, optional)
                Whether to log to txt file. Defaults to True.
            in_print (bool, optional)
                Whether to print to stdout. Defaults to True.
        """
        log_message = f"Epoch {in_epoch} {in_tag}: {in_value}"
        if in_print:
            print(log_message)
        # If .txt file was not initialized in __init__, value cannot be logged
        if self.log_txt and in_txt:
            self.txt_file.write(log_message + '\n')
        # If tensorboard writer was not initialized in __init__, value cannot be logged
        if self.log_tensorboard and in_tensorboard:
            self.writer.add_scalar(in_tag, in_value, in_epoch)

    def log_conf_json(self, in_conf_dict: Dict):
        r"""Logs a training configuration file as .json in the logging directory.

        Args:
            in_conf_dict (Dict): Configuration dictionary.
        """
        file_path = os.path.join(self.log_dir, 'conf.json')
        with open(file_path, 'w', encoding="utf-8") as json_file:
            json.dump(in_conf_dict, json_file, indent=4)  # Using indent for prettified JSON

    def close(self):
        """Closes the txt file and TensorBoard writer."""
        # Close the .txt file.
        if self.log_txt:
            self.txt_file.close()

        # Close tensorboard writer.
        if self.log_tensorboard:
            self.writer.close()


class ScalarAverager:
    r"""Utility to compute the average of scalar values. Mostly used to average losses over a whole epoch.

    Attributes:
        sum (float)
            Sum of the scalar values.
        count (int)
            Count of the scalar values.
    """

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, in_value, in_count):
        r"""Update the sum and count of the scalar values.

        Args:
            in_value (float)
                Scalar value.
            in_count (int)
                Count of the scalar value.
        """
        # ...
        self.sum += in_value * in_count
        self.count += in_count

    @property
    def average(self):
        r"""Compute the average of the scalar values.

        Returns:
            float: The average of the scalar values.
        """
        if self.count == 0:
            return float('nan')
        return self.sum / self.count
