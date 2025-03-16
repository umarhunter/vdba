import argparse
import json
import os

def load_json_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_config():
    parser = argparse.ArgumentParser(description="Retrieve app configuration")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json", 
        help="Path to the non-sensitive config file (JSON format)"
    )
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_json_config(args.config)
    else:
        print(f"Config file {args.config} not found. Using default values.")
        config = {}
    return config

if __name__ == "__main__":
    config = parse_config()
    print("Loaded configuration:", config)
