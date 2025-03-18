import json

def load_config(config_file='config.json'):
    """Loads a config dictionary from JSON."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config
