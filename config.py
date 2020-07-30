import json


def load_config(config_path):
    assert isinstance(config_path, str), f"config_path should be a str, got {type(config_path)}"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_model_config(config_path):

    config = load_config(config_path)

    # TODO: add checks here once we finalize runner config format

    return config


def load_runner_config(config_path):

    config = load_config(config_path)

    # TODO: add checks here once we finalize runner config format

    return config
