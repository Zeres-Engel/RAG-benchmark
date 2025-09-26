import json

import yaml


def load_yaml(path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict:
    """Load JSON data from a file."""
    with open(path, "r") as f:
        return json.load(f)


def load_json_string(json_string: str) -> dict:
    """Load JSON data from a string."""
    return json.loads(json_string)
