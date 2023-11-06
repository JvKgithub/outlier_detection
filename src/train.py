import json
import os

from data_setup_od.data_setup import DataSetupOD
from logger_od.logger_od import LoggerOD
from model_handler.model_handler import ModelHandler
from trainer_od.trainer_od import TrainerOD
from utils_od.utils_od import set_seed, get_normalization_stats, get_transforms


# Load config from JSON
with open(os.path.join('conf', 'train_config.json'), "r", encoding="utf-8") as file_read:
    config = json.load(file_read)

# Set seeds for deterministic run
if config["deterministic_run"]:
    set_seed(config["seed"])

# Construct logging directory if not specified
if config["log_dir"] is None:
    config["log_dir"] = f"{config['dataset_name']}_{config['model_name']}_scale{config['model_scale']}_lr{config['lr']}"

# Get data transformations
dataset_mean, dataset_std = get_normalization_stats(config)
input_img_size = tuple(config["input_img_size"])
transforms_dict = get_transforms(in_transform_type=config['transforms'], in_image_size=input_img_size, in_dataset_mean=dataset_mean,
                                 in_dataset_std=dataset_std, in_sample_rate=config['sample_rate'], in_sample_len=config['sample_len'])

# Setup dataloaders
data_setup_od = DataSetupOD(
    in_dataset=config["dataset_name"],
    in_path_download=config["path_data"],
    in_batchsize=config["batchsize"],
    in_val_size=config["val_size"],
    in_target_class=config["target_class"],
    in_duplication_factor=config["duplication_factor"],
    in_transforms=transforms_dict
)

# Initialize logger
continue_train = config["model_weights"] is not None and config["model_weights"] != "imagenet"
logger = LoggerOD(config["log_dir"], in_continue_train=continue_train)
logger.log_conf_json(config)

# Initialize model
if config['model_weights'] == "imagenet":
    CUSTOM_WEIGHTS = None
    PRETRAINED_IMAGENET = True
else:
    CUSTOM_WEIGHTS = config['model_weights']
    PRETRAINED_IMAGENET = False

model_handler = ModelHandler(
    in_model_name=config["model_name"],
    in_device=config["device"],
    in_model_scale=config["model_scale"],
    in_checkpoint_path=CUSTOM_WEIGHTS,
    in_pretrained_imagenet=PRETRAINED_IMAGENET,
    in_duplication_factor=config["duplication_factor"]
)

# Setup trainer
trainer = TrainerOD(
    in_logger=logger,
    in_data_setup=data_setup_od,
    in_model_handler=model_handler,
    in_lr=config["lr"],
    in_optimizer_name=config["optimizer"],
    in_device=config["device"],
    in_lr_scheduler=config["lr_scheduler"],
    in_weight_decay=config["weight_decay"],
    in_save_checkpoint_step=config["save_checkpoint_epochs"],
    in_eval_test=config["eval_test_step"],
    in_warmup_epochs=config["warmup_epochs"],
    in_loss_weights=config["loss_weights"],
    in_duplication_factor=config["duplication_factor"]
)
trainer.train(config["epochs"])
