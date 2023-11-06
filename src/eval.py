import json
import os

from data_setup_od.data_setup import DataSetupOD
from model_handler.model_handler import ModelHandler
from evaluator_od.evaluator_od import EvaluatorOD
from utils_od.utils_od import get_normalization_stats, get_transforms


# Load config from JSON
with open(os.path.join('conf', 'eval_config.json'), "r", encoding="utf-8") as file_read:
    config = json.load(file_read)

# Initialize model
model_handler = ModelHandler(
    in_model_name=config["model_name"],
    in_device=config["device"],
    in_model_scale=config["model_scale"],
    in_checkpoint_path=config["path_checkpoint"],
    in_duplication_factor=config["duplication_factor"]
)

# Setup evaluator
logdir = os.path.dirname(config["path_checkpoint"])
evaluator = EvaluatorOD(
    in_model_handler=model_handler,
    in_device=config["device"],
    in_log_dir=logdir
)

# Evaluate depending on mode
if config['evaluation_mode'] == 'dataset':  # Evaluate complete dataset
    # Get data transformations
    dataset_mean, dataset_std = get_normalization_stats(config)
    input_img_size = tuple(config["input_img_size"])
    transforms_dict = get_transforms(in_transform_type=config['transforms'], in_image_size=input_img_size, in_dataset_mean=dataset_mean,
                                     in_dataset_std=dataset_std, in_sample_rate=config['sample_rate'], in_sample_len=config['sample_len'])

    batch_stats = config['dataset_name'] != 'cifar10'
    # Setup dataloaders
    data_setup = DataSetupOD(
        in_dataset=config["dataset_name"],
        in_path_download=config["path_data"],
        in_batchsize=config["batchsize"],
        in_target_class=config["target_class"],
        in_transforms=transforms_dict,
        in_sample_len=config["sample_len"]
    )
    evaluator.evaluate_dataset(in_data_setup=data_setup, in_batch_stats=batch_stats)

elif config['evaluation_mode'] == 'image':  # Evaluate single image
    dataset_mean, dataset_std = get_normalization_stats(config)
    input_img_size = tuple(config["input_img_size"])
    transforms = get_transforms(in_transform_type='image', in_dataset_mean=dataset_mean, in_dataset_std=dataset_std,
                                in_image_size=config['input_img_size'])['test_transforms']
    evaluator.evaluate_image(in_path_image=config["path_data"], in_transform=transforms)

elif config['evaluation_mode'] == 'audio_file':  # Evaluate single audio file
    input_img_size = tuple(config["input_img_size"])
    transforms = get_transforms(in_transform_type='wav', in_sample_len=config["sample_len"], in_sample_rate=config["sample_rate"],
                                in_image_size=config['input_img_size'])['test_transforms']
    evaluator.evaluate_audio_wav(in_path_audio_file=config["path_data"], in_transforms=transforms, in_sample_len=config["sample_len"])
