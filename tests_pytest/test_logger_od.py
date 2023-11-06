import os
import json

import shutil
import pytest

from src.logger_od.logger_od import LoggerOD


def test_logger_init():
    """Test if the logger initializes correctly."""
    directory_log = '../logs/test_log_dir'

    try:
        logger = LoggerOD('test_log_dir')
        assert os.path.exists(directory_log)
        logger.close()

    finally:
        # Cleanup
        if os.path.exists(directory_log):
            shutil.rmtree(directory_log)


def test_logger_file_exists_error():
    """Test if FileExistsError is raised when log dir already exists and continue_train is False."""
    directory_log = '../logs/test_log_dir_exist'

    try:
        os.makedirs(directory_log, exist_ok=True)

        with pytest.raises(FileExistsError):
            LoggerOD('test_log_dir_exist')

    finally:
        # Cleanup
        if os.path.exists(directory_log):
            os.rmdir(directory_log)


def test_logger_log_scalar(capsys):
    """Test logging of scalars."""
    log_dir = '../logs/test_log_scalar'
    txt_file_path = os.path.join(log_dir, 'logs.txt')
    try:
        logger = LoggerOD('test_log_scalar')
        logger.log_scalar('train/loss_total', 0.5, 1)
        logger.close()

        # Check if the log message is printed to stdout
        captured = capsys.readouterr()
        assert "Epoch 1 train/loss_total: 0.5" in captured.out

        # Check if the log message is written to 'logs.txt'
        with open(txt_file_path, 'r', encoding='utf-8') as read_file:
            content = read_file.read()
            assert "Epoch 1 train/loss_total: 0.5" in content
    finally:
        shutil.rmtree('../logs/test_log_scalar')


def test_logger_log_conf_json():
    """Test logging of configuration to JSON."""
    conf_dict = {"lr": 0.001, "batch_size": 64}
    logger = LoggerOD('test_log_json')
    dir_path = os.path.join('../logs', 'test_log_json')

    try:
        logger.log_conf_json(conf_dict)

        # Check if the conf is correctly written to 'conf.json'
        with open(os.path.join(dir_path, 'conf.json'), 'r', encoding='utf-8') as file_read:
            content = json.load(file_read)
            assert content == conf_dict

        logger.close()

    finally:
        # Cleanup
        json_path = os.path.join(dir_path, 'conf.json')
        if os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


def test_logger_close():
    """Test if logger's close method works."""
    logger = LoggerOD('test_log_close')
    dir_path = os.path.join('../logs', 'test_log_close')

    try:
        logger.log_scalar('Loss', 0.5, 1)  # This will create logs.txt
        logger.close()

        # After closing, the file should exist but should not be writable
        with pytest.raises(ValueError):
            logger.txt_file.write("This should not be written.")

    finally:
        # Cleanup
        txt_path = os.path.join(dir_path, 'logs.txt')
        if os.path.exists(txt_path):
            os.remove(txt_path)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
