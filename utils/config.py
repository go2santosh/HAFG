# Import necessary libraries
import os
import json

# Configuration class to hold the configuration parameters with default values
class Config:
    def __init__(self, config_file):
        # Check if a configuration file exists
        self.config_file = config_file
        if os.path.exists(self.config_file):
            self.load(self.config_file)

    # Method to save the configuration parameters to a json file
    def save(self):
        import json
        with open(self.config_file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    # Method to load the configuration parameters from a json file
    def load(self, filepath):
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.__dict__.update(data)
        return self

    # Method to display the configuration parameters
    def display(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

def load_config(config_file_path):
    config = {}
    # Load existing config
    with open(config_file_path, 'r') as f:
        if config_file_path.split('.')[-1] == 'json':
            with open(config_file_path, 'r') as file:
                config = json.load(file)
        else:
            for line in f:
                key, value = line.strip().split(': ')
                # Attempt to convert value to int or float if possible
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                config[key] = value
    return config

def save_config(config_file_path, config_dict):
    """Save updated config dictionary back to the same colon-separated format."""
    with open(config_file_path, 'w') as f:
        # Save result to a file
        if config_file_path.split('.')[-1] == 'json':
            with open(config_file_path, 'w') as f:
                json.dump(config_dict, f, indent=4, separators=(',', ': '))
        else:
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")